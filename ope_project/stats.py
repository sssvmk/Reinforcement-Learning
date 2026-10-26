"""Paired statistics across repetitions."""
from __future__ import annotations
import itertools
import numpy as np
from scipy.stats import wilcoxon, friedmanchisquare


def cliffs_delta(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    gt = (a[:, None] > b[None, :]).sum()
    lt = (a[:, None] < b[None, :]).sum()
    return float((gt - lt) / (len(a) * len(b)))


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def paired_table(scores, lower_is_better=True):
    """scores: dict method -> array over repetitions (same reps, paired). Returns list of dict rows with
    Wilcoxon p, Holm-adjusted p, Cliff's delta and median paired difference for every pair."""
    names = list(scores)
    rows = []
    for a, b in itertools.combinations(names, 2):
        x, y = np.asarray(scores[a], float), np.asarray(scores[b], float)
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        d = x - y
        try:
            p = float(wilcoxon(x, y).pvalue) if len(x) >= 5 and np.any(d != 0) else np.nan
        except ValueError:
            p = np.nan
        rows.append(dict(a=a, b=b, n=int(ok.sum()), median_diff=float(np.median(d)) if len(d) else np.nan,
                         cliffs_delta=cliffs_delta(x, y) if len(x) else np.nan, p=p))
    ps = np.array([r["p"] for r in rows])
    ok = np.isfinite(ps)
    if ok.any():
        adj = holm(ps[ok])
        for r, v in zip([r for r, o in zip(rows, ok) if o], adj):
            r["p_holm"] = float(v)
    for r in rows:
        r.setdefault("p_holm", np.nan)
    return rows


def friedman(scores):
    names = [n for n in scores if np.isfinite(np.asarray(scores[n], float)).any()]
    M = np.array([scores[n] for n in names], float)
    if M.shape[1] < 3 or len(names) < 3:
        return dict(stat=np.nan, p=np.nan, n=int(M.shape[1]))
    s = friedmanchisquare(*M)
    return dict(stat=float(s.statistic), p=float(s.pvalue), n=int(M.shape[1]))
