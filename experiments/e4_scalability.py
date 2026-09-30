"""E4 - Scalability with the map resolution (Figure 11).

Office, warehouse and plaza, medium level, layouts 300-301, 2 queries, processed at
map sides of 200-1200 px. The visibility graph is run only up to 800 px.

    python experiments/e4_scalability.py [--workers 6] [--quick]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (build_case, cli, experiment_seeds, planner_params, run_pool, sample_queries,  # noqa: E402
                     save_rows, st_columns)

SIZES = (200, 300, 400, 600, 800, 1000, 1200)
VISIBILITY_MAX_SIDE = 800


def case_queries(scene, seed, size, queries=2):
    import zlib
    case = build_case(scene, "medio", seed, size=size)
    rng = np.random.default_rng(zlib.crc32(f"q4|{scene}|{seed}".encode()))
    probs, gt = sample_queries(case, queries, rng)
    return case, probs, gt


def run_case(scene, seed, size, queries=2):
    from registry import BY_KEY
    from tve import clear_cache
    _, probs, _ = case_queries(scene, seed, size, queries)
    rows = []
    keys = ["tve", "teselado", "astar", "theta", "dijkstra", "prm"] + (["visibility"] if size <= VISIBILITY_MAX_SIDE else [])
    for qi, P in enumerate(probs):
        for k in keys:
            if k == "tve":
                clear_cache()
            res = BY_KEY[k].execute(P, **planner_params(k))
            rows.append(dict(scene=scene, seed=seed, size=size, pixels=int(np.prod(P.shape)), query=qi, algo=k,
                             t_ms=1000 * res.time_s, ok=res.success,
                             t_tess_ms=res.stats.get("t_teselado_ms", np.nan), **st_columns(P)))
    return rows


def jobs(quick=False):
    s = experiment_seeds()["E4"]
    sizes = (200, 400) if quick else SIZES
    seeds = s["seeds"][:1] if quick else s["seeds"]
    return [(sc, k, z, s["queries"]) for sc in s["scenes"] for k in seeds for z in sizes]


def main():
    a = cli(__doc__, "e4.csv")
    df = save_rows(run_pool(run_case, jobs(a.quick), min(a.workers, 6), "E4"), Path(a.out))
    print(f"E4: {len(df)} rows -> {a.out}")


if __name__ == "__main__":
    main()
