"""E3 - Consecutive queries on the same map (Figure 10).

7 scenes x medium level x 2 layouts (seeds 200-201) x 25 queries (minimum separation
0.30 of the map side). TVE and WDT keep their preprocessing between queries; A*,
Theta*, PRM and the visibility graph are run per query through the common interface.

    python experiments/e3_multiquery.py [--workers 8] [--quick]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (build_case, cli, experiment_seeds, planner_params, query_rng, run_pool,  # noqa: E402
                     sample_queries, save_rows, st_columns, tve_config, wdt_config)

MIN_SEP = 0.30


def case_queries(scene, level, seed, queries):
    case = build_case(scene, level, seed)
    probs, gt = sample_queries(case, queries, query_rng("q3", scene, level, seed), min_sep=MIN_SEP)
    return case, probs, gt


def run_case(scene, level, seed, queries):
    from baselines.wdt.planner import Planner
    from registry import BY_KEY
    from tve import TVEPlanner, clear_cache
    case, probs, _ = case_queries(scene, level, seed, queries)
    clear_cache()
    tve = TVEPlanner(probs[0].env, tve_config()) if probs else None
    wdt = Planner(probs[0].env, wdt_config()) if probs else None
    rows = []
    for qi, P in enumerate(probs):
        base = dict(scene=scene, level=level, seed=seed, query=qi, **st_columns(P))
        t0 = time.perf_counter()
        ok = tve.plan(P.start, P.goal).success
        rows.append(dict(base, algo="tve", t_ms=1000 * (time.perf_counter() - t0), ok=ok))
        t0 = time.perf_counter()
        ok = wdt.plan(P.start, P.goal).success
        rows.append(dict(base, algo="teselado", t_ms=1000 * (time.perf_counter() - t0), ok=ok))
        for k in ("astar", "theta", "prm", "visibility"):
            res = BY_KEY[k].execute(P, **planner_params(k))
            rows.append(dict(base, algo=k, t_ms=1000 * res.time_s, ok=res.success))
    return rows


def jobs(quick=False):
    s = experiment_seeds()["E3"]
    seeds = s["seeds"][:1] if quick else s["seeds"]
    q = 6 if quick else s["queries"]
    return [(sc, lv, k, q) for sc in s["scenes"] for lv in s["levels"] for k in seeds]


def main():
    a = cli(__doc__, "e3.csv")
    df = save_rows(run_pool(run_case, jobs(a.quick), min(a.workers, 8), "E3"), Path(a.out))
    print(f"E3: {len(df)} rows -> {a.out}")


if __name__ == "__main__":
    main()
