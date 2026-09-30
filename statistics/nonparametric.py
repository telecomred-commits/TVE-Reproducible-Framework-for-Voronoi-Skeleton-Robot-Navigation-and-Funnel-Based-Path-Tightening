"""Non-parametric statistics used in the paper (no SciPy dependency).

* ``wilcoxon``      Wilcoxon signed-rank test, normal approximation with tie and
                    continuity corrections; returns W+, z, two-sided p and the effect
                    size r = z / sqrt(n).
* ``friedman``      Friedman test with average ranks (lower value = better);
                    chi-square p-value from the regularized upper incomplete gamma.
* ``holm``          Holm-Bonferroni step-down adjustment of a family of p-values.
* ``bootstrap_median``  percentile bootstrap CI of the median (5000 resamples, seed 0).
* ``nemenyi_cd``    critical difference of the Nemenyi post-hoc test (alpha = 0.05).
* ``gammaincc``     regularized upper incomplete gamma Q(a, x) (Numerical Recipes).

These functions reproduce the numbers reported in the paper; ``tests/test_statistics.py``
checks them against closed-form cases.
"""
from __future__ import annotations

import math

import numpy as np

# Studentized range statistic q_{0.05} / sqrt(2) for k = 2..12 (Demsar, 2006)
_NEMENYI_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
                11: 3.219, 12: 3.268}


def rankdata(x) -> np.ndarray:
    """Average ranks (1 = smallest), ties receive the mean of their ranks."""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(1, len(x) + 1)
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    return (np.bincount(inv, ranks) / cnt)[inv]


def wilcoxon(x, y):
    """Paired Wilcoxon signed-rank test of x - y. Returns (W+, z, p_two_sided, r)."""
    d = np.asarray(x, float) - np.asarray(y, float)
    d = d[np.isfinite(d) & (d != 0)]
    n = len(d)
    if n == 0:
        return 0.0, 0.0, 1.0, 0.0
    a = np.abs(d)
    order = np.argsort(a)
    ranks = np.empty(n)
    ranks[order] = np.arange(1, n + 1)
    _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, ranks)
    ranks = sums[inv] / cnt[inv]
    wp = ranks[d > 0].sum()
    mu = n * (n + 1) / 4
    var = n * (n + 1) * (2 * n + 1) / 24 - (cnt ** 3 - cnt).sum() / 48
    z = (wp - mu - 0.5 * np.sign(wp - mu)) / math.sqrt(var)
    p = math.erfc(abs(z) / math.sqrt(2))
    return float(wp), float(z), float(p), float(z / math.sqrt(n))


def gammaincc(a, x):
    """Regularized upper incomplete gamma function Q(a, x)."""
    if x <= 0:
        return 1.0
    if x < a + 1:
        ap, s, dl = a, 1 / a, 1 / a
        for _ in range(500):
            ap += 1
            dl *= x / ap
            s += dl
            if abs(dl) < abs(s) * 1e-14:
                break
        return 1 - s * math.exp(-x + a * math.log(x) - math.lgamma(a))
    b, c, d = x + 1 - a, 1e300, 1 / (x + 1 - a)
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1 / d
        dl = d * c
        h *= dl
        if abs(dl - 1) < 1e-14:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def chi2_sf(x, dof):
    return gammaincc(dof / 2, x / 2)


def friedman(M):
    """M: (n blocks, k treatments), lower = better. Returns (chi2, p, mean ranks)."""
    M = np.asarray(M, float)
    n, k = M.shape
    R = np.vstack([rankdata(r) for r in M])
    Rm = R.mean(0)
    chi2 = 12 * n / (k * (k + 1)) * ((Rm - (k + 1) / 2) ** 2).sum()
    return float(chi2), float(chi2_sf(chi2, k - 1)), Rm


def holm(pvalues):
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    p = np.asarray(pvalues, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (m - i) * p[j]))
        adj[j] = run
    return adj


def bootstrap_median(x, n=5000, seed=0, level=0.95):
    """(median, lower, upper) percentile bootstrap confidence interval of the median."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    rng = np.random.default_rng(seed)
    b = np.median(rng.choice(x, (n, len(x))), axis=1)
    a = (1 - level) / 2
    return float(np.median(x)), float(np.quantile(b, a)), float(np.quantile(b, 1 - a))


def nemenyi_cd(k, n):
    """Critical difference of mean ranks (Nemenyi, alpha = 0.05)."""
    q = _NEMENYI_Q05.get(k)
    return float("nan") if q is None else q * math.sqrt(k * (k + 1) / (6 * n))


def effect_size_r(z, n):
    return z / math.sqrt(n) if n else 0.0
