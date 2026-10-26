"""Evaluation metrics for off-policy evaluation experiments."""
from __future__ import annotations
import numpy as np
from scipy.stats import spearmanr, kendalltau


def value_errors(est, true):
    est, true = np.asarray(est, float), np.asarray(true, float)
    spread = np.ptp(true) or 1.0
    e = est - true
    return dict(abs=float(np.mean(np.abs(e))), signed=float(np.mean(e)), rmse=float(np.sqrt(np.mean(e ** 2))),
                rel=float(np.mean(np.abs(e) / (np.abs(true) + 1e-12))), nrmse=float(np.sqrt(np.mean(e ** 2)) / spread))


def ranking_metrics(est, true, k=3, tol=0.0):
    """Spearman, Kendall, Regret@1 (raw, normalized, tie-aware), precision@k, pairwise accuracy (ties ignored)."""
    est, true = np.asarray(est, float), np.asarray(true, float)
    ok = np.isfinite(est)
    out = dict(spearman=np.nan, kendall=np.nan, regret1=np.nan, nregret1=np.nan, regret1_tie=np.nan,
               prec_k=np.nan, pair_acc=np.nan)
    if ok.sum() < 3 or np.ptp(est[ok]) == 0:
        return out
    e, t = est[ok], true[ok]
    out["spearman"] = float(spearmanr(e, t)[0])
    out["kendall"] = float(kendalltau(e, t)[0])
    best = t.max()
    pick = int(np.argmax(e))
    out["regret1"] = float(best - t[pick])
    out["nregret1"] = float((best - t[pick]) / (np.ptp(t) or 1.0))
    ties = np.flatnonzero(e >= e.max() - 1e-12)
    out["regret1_tie"] = float(best - t[ties].mean())
    kk = min(k, len(t))
    out["prec_k"] = float(len(set(np.argsort(-e)[:kk]) & set(np.argsort(-t)[:kk])) / kk)
    iu = np.triu_indices(len(t), 1)
    dt, de = (t[:, None] - t[None, :])[iu], (e[:, None] - e[None, :])[iu]
    m = (dt != 0) & (de != 0)
    out["pair_acc"] = float(np.mean(np.sign(dt[m]) == np.sign(de[m]))) if m.any() else np.nan
    return out


def msve_floor_ratio(msve, msve_oracle):
    return float(msve / max(msve_oracle, 1e-300))


def updates_to_within(curve_J, n_updates_per_ckpt, truth, frac=0.10):
    """First checkpoint update count after which |J-truth| stays within frac*|truth| (NaN if never)."""
    err = np.abs(np.asarray(curve_J) - truth) <= frac * max(abs(truth), 1e-9)
    for c in range(len(err)):
        if err[c:].all():
            return float((c + 1) * n_updates_per_ckpt)
    return np.nan


def block_bootstrap_mean(x, block_len, n_boot, rng):
    """Moving-block bootstrap distribution of the mean of a time series."""
    x = np.asarray(x, float)
    n = len(x)
    b = max(1, min(block_len, n))
    nb = int(np.ceil(n / b))
    starts = rng.integers(0, n - b + 1, (n_boot, nb))
    idx = (starts[:, :, None] + np.arange(b)).reshape(n_boot, -1)[:, :n]
    return x[idx].mean(axis=1)


def tie_matrix(window_truth, block_len=20, n_boot=500, rng=None, level=0.95):
    """window_truth: (A, W) per-window true values. Entry (i,j) = 1 if agent i beats j with block-bootstrap
    confidence >= level, so un-separable pairs are 'ties'."""
    rng = rng or np.random.default_rng(0)
    A = window_truth.shape[0]
    M = np.zeros((A, A))
    for i in range(A):
        for j in range(A):
            if i != j:
                d = window_truth[i] - window_truth[j]
                M[i, j] = float(np.mean(block_bootstrap_mean(d, block_len, n_boot, rng) > 0) >= level)
    return M


def interval_coverage(lo, hi, true):
    lo, hi, true = map(np.asarray, (lo, hi, true))
    return float(np.mean((lo <= true) & (true <= hi)))


def trading_performance(daily_ret, periods=252):
    r = np.asarray(daily_ret, float)
    if len(r) == 0:
        return {}
    eq = np.cumprod(1 + r)
    cum = float(eq[-1] - 1)
    ann = float(eq[-1] ** (periods / len(r)) - 1)
    vol = float(r.std(ddof=1) * np.sqrt(periods)) if len(r) > 1 else np.nan
    dn = r[r < 0]
    dd = float(np.sqrt(np.mean(dn ** 2)) * np.sqrt(periods)) if len(dn) else np.nan
    mdd = float(np.max(1 - eq / np.maximum.accumulate(eq)))
    gp, gl = r[r > 0].sum(), -r[r < 0].sum()
    return dict(cum_return=cum, ann_return=ann, vol=vol,
                log_return=float(np.log(eq[-1])) if eq[-1] > 0 else np.nan,
                cont_rate_pa=float(np.log(eq[-1]) * periods / len(r)) if eq[-1] > 0 else np.nan,
                profit_per_100k=float(100000 * cum),
                sharpe=float(r.mean() * periods / vol) if vol and vol > 0 else np.nan,
                sortino=float(r.mean() * periods / dd) if dd and dd > 0 else np.nan,
                max_drawdown=mdd, calmar=float(ann / mdd) if mdd > 0 else np.nan,
                win_rate=float(np.mean(r > 0)), profit_factor=float(gp / gl) if gl > 0 else np.nan)
