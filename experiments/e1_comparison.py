"""E1 - Comparison of the fifteen planners on detected maps (Tables 4-6, 8; Figures 4-6, 8, 9).

7 scenes x 3 levels x 4 layouts (seeds 0-3) x 2 queries. Every planner receives the
same Problem (detected configuration space, S, T); the reference optimum is the
visibility graph. TVE runs through the common interface with its cache disabled.

    python experiments/e1_comparison.py [--workers 8] [--quick]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (ALGOS, NAMES, build_case, cli, experiment_seeds, metrics, planner_params,  # noqa: E402
                     query_rng, run_pool, sample_queries, save_rows, st_columns)


def case_queries(scene, level, seed, queries):
    case = build_case(scene, level, seed)
    probs, gt = sample_queries(case, queries, query_rng("q1", scene, level, seed))
    return case, probs, gt


def run_case(scene, level, seed, queries):
    from common.geometry import polyline_length
    from registry import BY_KEY
    from tve import clear_cache
    case, probs, gt = case_queries(scene, level, seed, queries)
    rows = []
    for qi, P in enumerate(probs):
        ref = BY_KEY["visibility"].execute(P)
        ref_len = polyline_length(ref.path) if ref.success else None
        for k in ALGOS:
            if k == "tve":
                clear_cache()
            res = ref if k == "visibility" else BY_KEY[k].execute(P, **planner_params(k))
            row = dict(case["info"], query=qi, algo=k, name=NAMES[k], **st_columns(P))
            row.update(metrics(P, res, ref_len, gt))
            rows.append(row)
    return rows


def jobs(quick=False):
    s = experiment_seeds()["E1"]
    seeds = s["seeds"][:1] if quick else s["seeds"]
    q = 1 if quick else s["queries"]
    return [(sc, lv, k, q) for sc in s["scenes"] for lv in s["levels"] for k in seeds]


def main():
    a = cli(__doc__, "e1.csv")
    df = save_rows(run_pool(run_case, jobs(a.quick), a.workers, "E1"), Path(a.out))
    print(f"E1: {len(df)} rows -> {a.out}")


if __name__ == "__main__":
    main()
