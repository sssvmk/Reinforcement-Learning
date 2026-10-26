"""Small statistics helpers."""
import numpy as np
from scipy import stats


def bootstrap_ci(x, stat=np.mean, n=2000, alpha=0.05, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    s = [stat(x[rng.integers(0, len(x), len(x))]) for _ in range(n)]
    return float(np.quantile(s, alpha / 2)), float(np.quantile(s, 1 - alpha / 2))


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def cliffs_delta(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    gt = (a[:, None] > b[None, :]).sum()
    lt = (a[:, None] < b[None, :]).sum()
    return float((gt - lt) / (len(a) * len(b)))


def mannwhitney(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    try:
        return float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except ValueError:
        return 1.0


def wilcoxon(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    if np.allclose(d, 0):
        return 1.0
    try:
        return float(stats.wilcoxon(d).pvalue)
    except ValueError:
        return np.nan


def paired_bootstrap(d, n=4000, seed=0):
    """Mean of paired differences with a 95% CI and a two-sided bootstrap p-value (resampling windows)."""
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    m = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    p = 2 * min((m <= 0).mean(), (m >= 0).mean())
    return float(d.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975)), float(max(p, 1.0 / n))


def block_bootstrap_diff(a, b, block=20, n=2000, seed=0):
    """Moving-block bootstrap over time for the mean difference of two daily return series."""
    d = np.asarray(a, float) - np.asarray(b, float)
    T = len(d)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(T / block))
    out = np.empty(n)
    for i in range(n):
        st = rng.integers(0, max(T - block, 1), nb)
        out[i] = np.concatenate([d[s:s + block] for s in st])[:T].sum()
    return float(d.sum()), float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))
