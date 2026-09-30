"""E7 - Topological diversity of the K candidate routes (Table 12).

7 scenes x 3 levels x 4 layouts (seeds 600-603) x 3 queries. For every query the
homotopy class of each candidate route is compared with the class of the Euclidean
optimum (visibility graph). Class signature: one vertical ray upwards from an interior
point of every obstacle that does not touch the map border (the holes of the free
space); the reduced word of signed ray crossings is a complete invariant of the
homotopy class of a path with fixed extremes.

    python experiments/e7_homotopy.py [--workers 8] [--quick]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (TVE_CFG, build_case, cli, experiment_seeds, query_rng, run_pool, sample_queries,  # noqa: E402
                     save_rows, st_columns, tve_config)

KS = TVE_CFG["candidates_E7"]
RAY_X_OFFSET = 0.137            # non-integer offset: rays never pass through pixel-aligned vertices


def homotopy_rays(occ):
    from baselines.topological import analyze_obstacles
    obs = analyze_obstacles(occ, min_area=4)
    return [(k + 1, float(o["point"][0]) + RAY_X_OFFSET, float(o["point"][1])) for k, o in enumerate(obs)]


def case_queries(scene, level, seed, queries):
    case = build_case(scene, level, seed)
    probs, gt = sample_queries(case, queries, query_rng("q7", scene, level, seed))
    return case, probs, gt


def run_case(scene, level, seed, queries):
    from baselines.topological import polyline_word
    from common.geometry import polyline_length
    from registry import BY_KEY
    from tve import TVEPlanner, clear_cache
    _, probs, _ = case_queries(scene, level, seed, queries)
    rows = []
    for qi, P in enumerate(probs):
        rays = homotopy_rays(P.occ)
        ref = BY_KEY["visibility"].execute(P)
        if not ref.success:
            continue
        w_opt = polyline_word(np.asarray(ref.path, float), rays)
        L_opt = polyline_length(ref.path)
        clear_cache()
        for K in KS:
            r = TVEPlanner(P.env, tve_config(candidates=K)).plan(P.start, P.goal)
            base = dict(scene=scene, level=level, seed=seed, query=qi, K=K, **st_columns(P))
            if not r.success:
                rows.append(dict(base, ok=False))
                continue
            words = [polyline_word(np.asarray(x["polyline"], float), rays) for x in r.routes]
            classes = set(words)
            rows.append(dict(base, ok=True, n_obstacles=len(rays), n_candidates=len(r.routes),
                             n_classes=len(classes), has_opt=w_opt in classes,
                             final_opt=polyline_word(np.asarray(r.path, float), rays) == w_opt,
                             ratio=polyline_length(r.path) / L_opt))
    return rows


def jobs(quick=False):
    s = experiment_seeds()["E7"]
    seeds = s["seeds"][:1] if quick else s["seeds"]
    q = 1 if quick else s["queries"]
    return [(sc, lv, k, q) for sc in s["scenes"] for lv in s["levels"] for k in seeds]


def main():
    a = cli(__doc__, "e7.csv")
    df = save_rows(run_pool(run_case, jobs(a.quick), a.workers, "E7"), Path(a.out))
    print(f"E7: {len(df)} rows -> {a.out}")


if __name__ == "__main__":
    main()
