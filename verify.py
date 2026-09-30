"""Compares a new execution (results/rerun) with the raw results of the paper (results/raw).

Geometric and topological results (path lengths, ratios, validity, clearances, turning
radii, safety, homotopy classes, image-processing metrics...) are deterministic and must
be identical up to floating-point rounding: boolean and categorical columns must match
exactly, numeric columns within a relative tolerance (default 1e-5; vectorized NumPy
reductions may differ in the last bits depending on memory alignment, which occasionally
propagates to ~1e-6). Computation times are not compared.

    python verify.py                     # all experiments present in results/rerun
    python verify.py --rerun other/dir --tol 1e-6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
KEYS = {"e1": ["scene", "level", "seed", "query", "algo"], "e2": ["scene", "level", "seed", "query", "algo"],
        "e3": ["scene", "level", "seed", "query", "algo"], "e4": ["scene", "seed", "size", "query", "algo"],
        "e5": ["scene", "level", "seed", "query", "algo"], "e6": ["scene", "level", "seed", "query", "variant"],
        "e7": ["scene", "level", "seed", "query", "K"]}
def _is_time(c):
    """Computation times (hardware dependent) end in _ms; travel times of the robot
    (tiempo_holonomico_s, tiempo_diferencial_s, ...) are geometric and are compared."""
    return c.endswith("_ms")


def compare(name, raw, rerun, tol, subset=False):
    a = pd.read_csv(raw).set_index(KEYS[name]).sort_index()
    b = pd.read_csv(rerun).set_index(KEYS[name]).sort_index()
    if subset:                                   # e.g. a --quick execution: compare the runs it contains
        missing = b.index.difference(a.index)
        if len(missing):
            return False, f"{len(missing)} runs of the new execution are not in the paper results"
        a = a.loc[b.index]
    elif len(a.index.symmetric_difference(b.index)):
        return False, f"different sets of runs ({len(a)} vs {len(b)} rows)"
    cols = [c for c in a.columns if c in b.columns and not _is_time(c)]
    bad, exact, max_rel, n_rounding = [], 0, 0.0, 0
    for c in cols:
        x, y = a[c], b.loc[a.index, c]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(x):
            xv, yv = x.to_numpy(float), y.to_numpy(float)
            both = np.isfinite(xv) & np.isfinite(yv)
            special_ok = (np.isnan(xv) & np.isnan(yv)) | (np.isinf(xv) & np.isinf(yv) & (np.sign(xv) == np.sign(yv)))
            rel = np.zeros(len(xv))
            rel[both] = np.abs(xv[both] - yv[both]) / np.maximum(np.abs(xv[both]), 1e-12)
            with np.errstate(invalid="ignore"):
                same = (both & ((rel <= tol) | (np.abs(xv - yv) <= 1e-9))) | special_ok
            if both.any():
                n_rounding += int((both & (xv != yv)).sum())
                max_rel = max(max_rel, float(rel[both].max()))
        else:                                    # booleans/strings must be identical; missing on both sides is equal
            same = (x.astype(str).to_numpy() == y.astype(str).to_numpy()) | (x.isna().to_numpy() & y.isna().to_numpy())
        if not same.all():
            bad.append(f"{c} ({int((~same).sum())} rows)")
        else:
            exact += 1
    if bad:
        return False, "DIFFERENT: " + ", ".join(bad)
    note = (f"; {n_rounding} numeric values differ only by floating-point rounding (max. relative difference "
            f"{max_rel:.1e})") if n_rounding else "; all values bit-identical"
    return True, f"{len(a)} runs, {exact} non-timing columns equal{note}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=str(ROOT / "results" / "raw"))
    ap.add_argument("--rerun", default=str(ROOT / "results" / "rerun"))
    ap.add_argument("--tol", type=float, default=1e-5, help="relative tolerance for floating-point rounding")
    ap.add_argument("--subset", action="store_true", help="compare only the runs present in --rerun (quick mode)")
    a = ap.parse_args()
    ok_all = True
    for name in KEYS:
        rr = Path(a.rerun) / f"{name}.csv"
        if not rr.exists():
            continue
        ok, msg = compare(name, Path(a.raw) / f"{name}.csv", rr, a.tol, a.subset)
        ok_all &= ok
        print(f"{name.upper()}: {msg}")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
