"""GENERATE step: builds every scene of E1-E7 and exports the queries.

For each experiment, map and query it writes the points S and T (pixel coordinates of the
detected map, x = column, y = row) and the scale, to ``seeds/queries_E{n}.csv``. With
``--images`` it also writes, for every map, the ground truth, the synthesized camera
image, the rectified image and the detected obstacle mask to ``results/scenes/``.

The experiment scripts rebuild the same scenes and queries by themselves (all random
generators are seeded, see seeds/experiment_seeds.csv); this step makes them visible and
checkable without running any planner.

    python scenes/generate.py                    # all experiments, query lists only
    python scenes/generate.py --exp E1 --images  # also the images of E1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import _common as X  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402


def _save_images(case, tag):
    out = ROOT / "results" / "scenes"
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / f"{tag}_ground_truth.png"), (~case["gt_hi"]).astype(np.uint8) * 255)
    cv2.imwrite(str(out / f"{tag}_camera.jpg"), cv2.cvtColor(case["photo"], cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out / f"{tag}_rectified.png"), cv2.cvtColor(case["perception"].image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out / f"{tag}_detected_obstacles.png"), (~case["env"].obstacles).astype(np.uint8) * 255)


def _rows(exp, case, probs, size, images):
    info = case["info"]
    tag = f"{exp}_{info['scene']}_{info['level']}_{info['seed']}_{size}px"
    if images:
        _save_images(case, tag)
    return [dict(experiment=exp, scene=info["scene"], level=info["level"], seed=info["seed"], map_side_px=size,
                 query=qi, **X.st_columns(P), meters_per_pixel=info["mpp"], texture=info["texture"],
                 camera_tilt=info["perspective"], iou=info["iou"]) for qi, P in enumerate(probs)]


def job(exp, args, images):
    import importlib
    mod = {"E1": "e1_comparison", "E2": "e2_tve_vs_wdt", "E3": "e3_multiquery", "E4": "e4_scalability",
           "E5": "e5_replanning", "E6": "e6_ablation", "E7": "e7_homotopy"}[exp]
    m = importlib.import_module(mod)
    out = m.case_queries(*args)
    case, probs = out[0], out[1]
    size = args[2] if exp == "E4" else X.SIZE
    return _rows(exp, case, probs, size, images)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", nargs="*", default=["E1", "E2", "E3", "E4", "E5", "E6", "E7"])
    ap.add_argument("--images", action="store_true")
    ap.add_argument("--workers", type=int, default=X.default_workers())
    a = ap.parse_args()
    import importlib
    sys.path.insert(0, str(ROOT / "experiments"))
    for exp in a.exp:
        mod = importlib.import_module({"E1": "e1_comparison", "E2": "e2_tve_vs_wdt", "E3": "e3_multiquery",
                                       "E4": "e4_scalability", "E5": "e5_replanning", "E6": "e6_ablation",
                                       "E7": "e7_homotopy"}[exp])
        jobs = [(exp, j, a.images) for j in mod.jobs()]
        rows = X.run_pool(job, jobs, a.workers, f"generate {exp}")
        df = X.save_rows(rows, ROOT / "seeds" / f"queries_{exp}.csv")
        df = df.sort_values(["scene", "level", "seed", "map_side_px", "query"])
        df.to_csv(ROOT / "seeds" / f"queries_{exp}.csv", index=False)
        print(f"{exp}: {len(df)} queries -> seeds/queries_{exp}.csv")


if __name__ == "__main__":
    main()
