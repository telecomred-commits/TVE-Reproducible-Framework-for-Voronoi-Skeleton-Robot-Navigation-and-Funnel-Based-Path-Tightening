"""Master script of the ANALYZE and FIGURES steps.

    python paper_figures.py                       # tables, statistics and figures from results/raw
    python paper_figures.py --raw results/rerun   # the same from a new execution

It runs statistics/analyze.py (tables in results/tables/, LaTeX bodies and the macros
with every number quoted in the text in results/tables/latex/) and
figures/make_figures.py (figures/output/).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=str(ROOT / "results" / "raw"))
    ap.add_argument("--tables", default=str(ROOT / "results" / "tables"))
    ap.add_argument("--only", choices=("quantitative", "qualitative"))
    a = ap.parse_args()
    subprocess.run([sys.executable, str(ROOT / "statistics" / "analyze.py"), "--raw", a.raw, "--out", a.tables],
                   check=True)
    cmd = [sys.executable, str(ROOT / "figures" / "make_figures.py"), "--raw", a.raw]
    if a.only:
        cmd += ["--only", a.only]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
