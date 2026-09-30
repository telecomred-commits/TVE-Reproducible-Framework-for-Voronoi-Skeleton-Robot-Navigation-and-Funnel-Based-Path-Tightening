"""E2 - Paired comparison of TVE and WDT without cache, with the time of every stage
(Figure 7, Table 7 and the paired statistics of Section 4.3).

7 scenes x 3 levels x 6 layouts (seeds 100-105) x 3 queries; the execution order of
the two methods alternates with the query index.

    python experiments/e2_tve_vs_wdt.py [--workers 8] [--quick]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (NAMES, build_case, cli, experiment_seeds, metrics, query_rng, run_pool,  # noqa: E402
                     sample_queries, save_rows, st_columns, tve_config, wdt_config)


def case_queries(scene, level, seed, queries):
    case = build_case(scene, level, seed)
    probs, gt = sample_queries(case, queries, query_rng("q2", scene, level, seed))
    return case, probs, gt


def run_case(scene, level, seed, queries):
    from baselines.wdt.planner import Planner
    from common.geometry import polyline_length
    from common.problem import AlgorithmResult
    from registry import BY_KEY
    from tve import TVEPlanner, clear_cache
    case, probs, gt = case_queries(scene, level, seed, queries)
    rows = []
    for qi, P in enumerate(probs):
        ref = BY_KEY["visibility"].execute(P)
        ref_len = polyline_length(ref.path) if ref.success else None
        order = ("tve", "teselado") if qi % 2 == 0 else ("teselado", "tve")
        for k in order:
            if k == "tve":
                clear_cache()
                t0 = time.perf_counter()
                r = TVEPlanner(P.env, tve_config()).plan(P.start, P.goal, use_cache=False)
                dt = time.perf_counter() - t0
                path, ok, tim = r.path, r.success, r.timings
            else:
                t0 = time.perf_counter()
                r = Planner(P.env, wdt_config()).plan(P.start, P.goal)
                dt = time.perf_counter() - t0
                path, ok, tim = r.final_path, r.success, r.timings
            res = AlgorithmResult(k, NAMES[k], ok, path if ok else None, dt)
            row = dict(case["info"], query=qi, algo=k, name=NAMES[k], t_total_ms=1000 * dt,
                       n_skeleton_nodes=len(r.graph.nodes) if getattr(r, "graph", None) is not None else np.nan,
                       **st_columns(P))
            row.update({f"t_{a}_ms": 1000 * b for a, b in tim.items() if a != "total"})
            row.update(metrics(P, res, ref_len, gt))
            rows.append(row)
    return rows


def jobs(quick=False):
    s = experiment_seeds()["E2"]
    seeds = s["seeds"][:1] if quick else s["seeds"]
    q = 1 if quick else s["queries"]
    return [(sc, lv, k, q) for sc in s["scenes"] for lv in s["levels"] for k in seeds]


def main():
    a = cli(__doc__, "e2.csv")
    df = save_rows(run_pool(run_case, jobs(a.quick), min(a.workers, 8), "E2"), Path(a.out))
    print(f"E2: {len(df)} rows -> {a.out}")


if __name__ == "__main__":
    main()
