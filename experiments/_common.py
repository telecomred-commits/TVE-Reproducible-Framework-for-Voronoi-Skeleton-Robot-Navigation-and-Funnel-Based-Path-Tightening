"""Shared infrastructure of experiments E1-E7.

Every experiment follows the same chain, identical for all planners:

    ground-truth layout (scenes/) -> synthesized camera image (image_formation/)
    -> image processing (src/perception/) -> DETECTED map -> configuration space
    -> planner (src/tve, src/baselines) -> metrics on the detected map
    -> safety check against the ground truth.

All parameters are read from configs/*.json. All random generators are seeded with
``zlib.crc32`` of a string that identifies the experiment, scene, level, layout seed
(and query), see ``seeds/experiment_seeds.csv``; therefore maps, images, the points
S and T and all geometric results are identical on any machine.
"""
from __future__ import annotations

import os

# one thread per process, so that timings do not depend on library multithreading
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import csv
import json
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "statistics", ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def load_config(name: str) -> dict:
    return json.loads((ROOT / "configs" / f"{name}.json").read_text(encoding="utf-8"))


EXP = load_config("experiment")
TVE_CFG = load_config("tve")
BASE_CFG = load_config("baselines")
PERC_CFG = load_config("perception")["parameters"]
IMG_LEVELS = load_config("image_formation")["levels"]

LEVELS = tuple(EXP["levels"])
LEVEL_EN = EXP["level_names_in_paper"]
SIZE = EXP["map_side_px"]
GT_SIZE = EXP["ground_truth_side_px"]
PHOTO = tuple(EXP["camera_image_size_hw"])
ROBOT_D = EXP["robot_diameter_m"]
R_MIN = EXP["ackermann_min_turning_radius_m"]
ALGOS = EXP["planners_e1"]
NAMES = EXP["planner_names_in_paper"]


def planner_params(key: str) -> dict:
    """Exact parameters passed to a planner through the common interface."""
    if key == "tve":
        return dict(TVE_CFG["registry"])
    return dict(BASE_CFG["planners"][key])


def tve_config(**overrides):
    from tve import TVEConfig
    return TVEConfig(**{**TVE_CFG["planner"], **overrides})


def wdt_config():
    from baselines.wdt.config import PlannerConfig
    return PlannerConfig(**BASE_CFG["wdt_planner"])


def scene_names():
    from scenes.generators import SCENES
    return list(SCENES)


# --------------------------------------------------------------------------- scene -> image -> detected map
def build_case(scene: str, level: str, seed: int, size: int = SIZE):
    """Ground truth, synthesized camera image and detected map of one scene."""
    from common.environment import Environment
    from image_formation.model import random_conditions, render_photo
    from perception.pipeline import PerceptionParams, mask_metrics, object_recall, process_photo
    from scenes.generators import SCENES
    fn, width_m = SCENES[scene]
    rng = np.random.default_rng(zlib.crc32(f"paper|{scene}|{level}|{seed}".encode()))
    gt_hi = fn(seed=seed, size=GT_SIZE, width_m=width_m)
    cond = random_conditions(level, rng, levels=IMG_LEVELS)
    sp = render_photo(gt_hi, cond, rng, photo_size=PHOTO)
    err = EXP["corner_marking_error_px"]
    corners = None if sp.corners is None else sp.corners + rng.uniform(-err, err, (4, 2))
    t0 = time.perf_counter()
    res, mpp = process_photo(sp.photo, PerceptionParams(**PERC_CFG, max_side=size), corners=corners,
                             width_m=width_m, height_m=width_m)
    t_perc = time.perf_counter() - t0
    gt = cv2.resize(gt_hi.astype(np.uint8), res.mask.shape[::-1], interpolation=cv2.INTER_AREA) > 0
    env = Environment(res.mask, mpp, name=f"{scene}-{level}-{seed}")
    m = mask_metrics(res.mask, gt, tolerance_px=2)
    found, total = object_recall(res.mask, gt)
    info = dict(scene=scene, level=level, seed=seed, texture=cond.texture, perspective=float(cond.perspective),
                iou=m["iou"], precision=m["precision"], recall=m["recall"], objects_found=found,
                objects_total=total, t_perception_ms=1000 * t_perc, mpp=mpp)
    return dict(env=env, gt=gt, gt_hi=gt_hi, photo=sp.photo, corners=corners, perception=res, info=info,
                conditions=cond)


class GroundTruth:
    """Safety check of a path against the REAL obstacles (dilated by the robot radius)."""

    def __init__(self, gt: np.ndarray, radius_px: int):
        from common.cspace import configuration_space
        self.occ = configuration_space(gt, radius_px)
        self.depth = cv2.distanceTransform(self.occ.astype(np.uint8), cv2.DIST_L2, 5)
        free = (~gt).astype(np.uint8)
        free[[0, -1], :] = 0
        free[:, [0, -1]] = 0
        self.clear = cv2.distanceTransform(free, cv2.DIST_L2, 5)
        _, self.comp = cv2.connectedComponents((~self.occ).astype(np.uint8), connectivity=8)

    def check(self, path, mpp):
        from common.geometry import bilinear, densify
        pts = densify(np.asarray(path, float), 0.5)
        pen = float(bilinear(self.depth, pts).max())
        return dict(gt_penetration_px=pen, gt_safe=pen <= EXP["ground_truth_safety_max_penetration_px"],
                    gt_clearance_m=float(bilinear(self.clear, pts).min()) * mpp)


def sample_queries(case, n, rng, min_sep=None):
    """Draws S and T: free in BOTH the detected and the ground-truth configuration
    spaces, in the same connected region, separated by ``min_sep`` x map side."""
    from common import scenarios as sc
    from common.cspace import configuration_space, robot_radius_px
    from common.problem import Problem
    min_sep = EXP["query_min_separation_fraction"] if min_sep is None else min_sep
    env = case["env"]
    r = robot_radius_px(ROBOT_D, 0.0, env.meters_per_pixel)
    occ = configuration_space(env.obstacles, r)
    gt = GroundTruth(case["gt"], r)
    both = ~occ & ~gt.occ
    out = []
    for _ in range(40 * n):
        if len(out) >= n:
            break
        pts = sc.random_free_points(~both, rng, 2, min_sep=min_sep * min(env.shape))
        if pts is None:
            break
        s, t = pts
        if gt.comp[int(s[1]), int(s[0])] != gt.comp[int(t[1]), int(t[0])]:
            continue
        P = Problem.build(env, s, t, ROBOT_D, occ=occ, seed=int(rng.integers(1 << 30)))
        if P.connected():
            out.append(P)
    return out, gt


def query_rng(tag: str, scene: str, level: str, seed: int):
    return np.random.default_rng(zlib.crc32(f"{tag}|{scene}|{level}|{seed}".encode()))


def st_columns(P) -> dict:
    return dict(sx=float(P.start[0]), sy=float(P.start[1]), tx=float(P.goal[0]), ty=float(P.goal[1]))


def metrics(P, res, ref_len, gt):
    from common.metrics import evaluate
    dd = EXP["differential_drive"]
    m = evaluate(P, res, ref_len, dd["v_m_s"], dd["omega_deg_s"], R_MIN)
    row = {k: v for k, v in m.items() if not isinstance(v, str)}
    if res.success and res.path is not None:
        row.update(gt.check(res.path, P.mpp))
    return row


# --------------------------------------------------------------------------- execution
def _init_worker():
    cv2.setNumThreads(1)
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:  # noqa: BLE001
        pass


def run_pool(fn, jobs, workers, label):
    rows = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as ex:
        futs = [ex.submit(fn, *j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            rows.extend(f.result())
            if i % max(1, len(jobs) // 20) == 0 or i == len(jobs):
                print(f"  {label}: {i}/{len(jobs)} ({time.perf_counter() - t0:.0f} s)", flush=True)
    return rows


def default_workers():
    return max(1, (os.cpu_count() or 2) // 2)


def save_rows(rows, path: Path):
    import pandas as pd
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


def experiment_seeds() -> dict:
    """Layout seeds and number of queries per experiment, from seeds/experiment_seeds.csv."""
    out = {}
    with open(ROOT / "seeds" / "experiment_seeds.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r["experiment"]] = dict(
                scenes=r["scenes"].split(";") if r["scenes"] != "all" else scene_names(),
                levels=r["levels"].split(";") if r["levels"] != "all" else list(LEVELS),
                seeds=list(range(int(r["seed_first"]), int(r["seed_first"]) + int(r["layouts"]))),
                queries=int(r["queries_per_map"]), extra=r["extra"])
    return out


def cli(description: str, default_out: str):
    import argparse
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--out", default=str(ROOT / "results" / "rerun" / default_out),
                    help="output CSV (the results used in the paper are in results/raw/)")
    ap.add_argument("--quick", action="store_true", help="one layout and one query per scene/level (smoke test)")
    return ap.parse_args()
