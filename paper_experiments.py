"""Master script of the RUN step: executes experiments E1-E7.

    python paper_experiments.py all                 # E1-E7 -> results/rerun/e1.csv ... e7.csv
    python paper_experiments.py E2 E7 --workers 8
    python paper_experiments.py all --quick         # smoke test (one layout, one query per case)

Each experiment is also a stand-alone script in experiments/ (e1_comparison.py ...
e7_homotopy.py). The raw results used in the paper are in results/raw/; new executions
are written to results/rerun/ so that they never overwrite them. Timings depend on the
hardware; every geometric result is deterministic (see verify.py).
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "experiments"))
import _common as X  # noqa: E402

MODULES = {"E1": ("e1_comparison", None), "E2": ("e2_tve_vs_wdt", 8), "E3": ("e3_multiquery", 8),
           "E4": ("e4_scalability", 6), "E5": ("e5_replanning", 8), "E6": ("e6_ablation", 8), "E7": ("e7_homotopy", None)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiments", nargs="+", help="E1 ... E7 or all")
    ap.add_argument("--workers", type=int, default=X.default_workers())
    ap.add_argument("--out", default=str(ROOT / "results" / "rerun"))
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    exps = list(MODULES) if "all" in [e.lower() for e in a.experiments] else [e.upper() for e in a.experiments]
    out = Path(a.out)
    for e in exps:
        name, cap = MODULES[e]
        mod = importlib.import_module(name)
        t0 = time.perf_counter()
        workers = min(a.workers, cap) if cap else a.workers
        rows = X.run_pool(mod.run_case, mod.jobs(a.quick), workers, e)
        df = X.save_rows(rows, out / f"{e.lower()}.csv")
        print(f"{e}: {len(df)} rows in {time.perf_counter() - t0:.0f} s -> {out / (e.lower() + '.csv')}", flush=True)


if __name__ == "__main__":
    main()
