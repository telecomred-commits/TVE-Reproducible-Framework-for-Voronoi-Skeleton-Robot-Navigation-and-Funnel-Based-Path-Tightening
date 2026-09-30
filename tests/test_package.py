"""Tests of the reproducibility package.

    python -m pytest tests -q
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import _common as X  # noqa: E402
from nonparametric import bootstrap_median, friedman, gammaincc, holm, rankdata, wilcoxon  # noqa: E402


# --------------------------------------------------------------------------- statistics
def test_gammaincc_matches_chi2_closed_form():
    # chi-square with 2 dof: survival function exp(-x/2)
    for x in (0.5, 3.0, 12.0):
        assert gammaincc(1.0, x / 2) == pytest.approx(math.exp(-x / 2), rel=1e-12)


def test_rankdata_ties():
    assert rankdata([3, 1, 3, 2]).tolist() == [3.5, 1.0, 3.5, 2.0]


def test_wilcoxon_symmetric_and_significant():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 200)
    _, z, p, r = wilcoxon(x + 1.0, x)
    assert p < 1e-10 and z > 0 and 0 < r <= 1
    _, z0, p0, _ = wilcoxon(x, x[::-1])
    assert p0 > 0.05


def test_friedman_detects_ordering():
    M = np.tile([1.0, 2.0, 3.0], (30, 1))
    chi2, p, Rm = friedman(M)
    assert Rm.tolist() == [1.0, 2.0, 3.0]
    assert chi2 == pytest.approx(60.0) and p < 1e-12


def test_holm_monotone():
    adj = holm([0.01, 0.04, 0.03])
    assert adj.tolist() == pytest.approx([0.03, 0.06, 0.06])


def test_bootstrap_median_contains_median():
    m, lo, hi = bootstrap_median(np.arange(101))
    assert lo <= m <= hi and m == 50


# --------------------------------------------------------------------------- determinism of the pipeline
def test_scene_and_queries_match_paper_raw_results():
    """The first query of E1 (office, low level, layout 0) is rebuilt and planned with
    TVE; its geometric metrics must equal the raw results of the paper."""
    import e1_comparison as E1
    from common.geometry import polyline_length
    from registry import BY_KEY
    from tve import clear_cache
    case, probs, gt = E1.case_queries("office", "facil", 0, 1)
    P = probs[0]
    ref = BY_KEY["visibility"].execute(P)
    clear_cache()
    res = BY_KEY["tve"].execute(P, **X.planner_params("tve"))
    m = X.metrics(P, res, polyline_length(ref.path), gt)
    raw = pd.read_csv(ROOT / "results" / "raw" / "e1.csv")
    row = raw[(raw.scene == "office") & (raw.level == "facil") & (raw.seed == 0) & (raw["query"] == 0)
              & (raw.algo == "tve")].iloc[0]
    for c in ("longitud_m", "relacion_optima", "holgura_min_m", "radio_giro_min_m", "gt_clearance_m"):
        assert m[c] == pytest.approx(row[c], rel=1e-9, abs=1e-12), c
    assert case["info"]["iou"] == pytest.approx(row["iou"], rel=1e-12)


def test_configs_are_complete():
    assert set(X.ALGOS) == set(X.BASE_CFG["planners"]) | {"tve"}
    assert X.TVE_CFG["planner"]["turn_radius"] == X.R_MIN
    assert X.BASE_CFG["planners"]["hybrid_astar"]["radio_giro_m"] == X.R_MIN
    assert set(X.IMG_LEVELS) == set(X.LEVELS)
