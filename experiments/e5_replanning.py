"""E5 - Replanning after a person or a cart appears on the planned route (Table 9).

7 scenes x low/medium levels x 3 layouts (seeds 400-402) x 2 queries. TVE plans the
original route; a person (disk of radius 0.3 m) or a cart (1.0 x 0.6 m, random
orientation) is placed at a random fraction 0.35-0.65 of that route, and every planner
replans from S on the updated map.

    python experiments/e5_replanning.py [--workers 8] [--quick]
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (ROBOT_D, build_case, cli, experiment_seeds, planner_params, query_rng, run_pool,  # noqa: E402
                     sample_queries, save_rows, st_columns, tve_config)

KEYS = ["tve", "teselado", "astar", "theta", "rrt", "prm", "hybrid_astar", "visibility"]
PERSON_RADIUS_M = 0.3
CART_SIZE_M = (1.0, 0.6)


def _point_at(path, frac):
    P = np.asarray(path, float)
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))])
    t = frac * s[-1]
    return np.array([np.interp(t, s, P[:, 0]), np.interp(t, s, P[:, 1])])


def case_queries(scene, level, seed, queries):
    case = build_case(scene, level, seed)
    rng = query_rng("q5", scene, level, seed)
    probs, gt = sample_queries(case, queries, rng)
    return case, probs, gt, rng


def run_case(scene, level, seed, queries):
    from common.environment import Environment
    from common.geometry import polyline_length
    from common.problem import Problem
    from registry import BY_KEY
    from tve import TVEPlanner, clear_cache
    case, probs, gt, rng = case_queries(scene, level, seed, queries)
    rows = []
    for qi, P in enumerate(probs):
        clear_cache()
        tp = TVEPlanner(P.env, tve_config())
        r0 = tp.plan(P.start, P.goal)                      # original route (kept in cache)
        if not r0.success:
            continue
        L0 = polyline_length(r0.path)
        c = _point_at(r0.path, rng.uniform(0.35, 0.65))
        obst = P.env.obstacles.copy()
        yy, xx = np.ogrid[:obst.shape[0], :obst.shape[1]]
        kind = "person" if rng.random() < 0.5 else "cart"
        if kind == "person":
            obst |= (xx - c[0]) ** 2 + (yy - c[1]) ** 2 <= (PERSON_RADIUS_M / P.mpp) ** 2
        else:
            ang = rng.uniform(0, math.pi)
            box = cv2.boxPoints(((c[0], c[1]), (CART_SIZE_M[0] / P.mpp, CART_SIZE_M[1] / P.mpp), math.degrees(ang)))
            tmp = np.zeros(obst.shape, np.uint8)
            cv2.fillPoly(tmp, [np.round(box).astype(np.int32)], 1)
            obst |= tmp.astype(bool)
        env2 = Environment(obst, P.mpp)
        P2 = Problem.build(env2, P.start, P.goal, ROBOT_D)
        if not P2.connected():
            continue
        ref = BY_KEY["visibility"].execute(P2)
        ref_len = polyline_length(ref.path) if ref.success else None
        for k in KEYS:
            if k == "tve":
                t0 = time.perf_counter()
                r = TVEPlanner(env2, tve_config()).plan(P2.start, P2.goal)
                dt = time.perf_counter() - t0
                ok, path = r.success, r.path
            else:
                res = ref if k == "visibility" else BY_KEY[k].execute(P2, **planner_params(k))
                ok, path, dt = res.success, res.path, res.time_s
            L = polyline_length(path) if ok else np.nan
            rows.append(dict(scene=scene, level=level, seed=seed, query=qi, algo=k, event=kind, t_ms=1000 * dt, ok=ok,
                             ratio=L / ref_len if ok and ref_len else np.nan, detour=L / L0 if ok else np.nan,
                             event_x=float(c[0]), event_y=float(c[1]), **st_columns(P)))
    return rows


def jobs(quick=False):
    s = experiment_seeds()["E5"]
    seeds = s["seeds"][:1] if quick else s["seeds"]
    # the same random generator draws S, T and then the events: the number of queries is
    # kept in quick mode so that the quick runs are a subset of the full experiment
    return [(sc, lv, k, s["queries"]) for sc in s["scenes"] for lv in s["levels"] for k in seeds]


def main():
    a = cli(__doc__, "e5.csv")
    df = save_rows(run_pool(run_case, jobs(a.quick), min(a.workers, 8), "E5"), Path(a.out))
    print(f"E5: {len(df)} rows -> {a.out}")


if __name__ == "__main__":
    main()
