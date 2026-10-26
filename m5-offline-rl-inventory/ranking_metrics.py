"""Ranking-quality metrics comparing an estimator's scores with ground-truth values (tie-aware)."""
import numpy as np
from scipy import stats


def tie_matrix(returns, names, z=1.96):
    """tie[i,j] True if the paired (CRN) difference of true returns is statistically indistinguishable from 0."""
    n = len(names)
    tie = np.zeros((n, n), bool)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = np.asarray(returns[names[i]]), np.asarray(returns[names[j]])
            k = min(len(a), len(b))
            d = a[:k] - b[:k]
            se = d.std(ddof=1) / np.sqrt(k) if k > 1 else np.inf
            tie[i, j] = tie[j, i] = abs(d.mean()) < z * se
    return tie


def spearman(est, truth):
    if len(est) < 3 or np.std(est) == 0 or np.std(truth) == 0:
        return np.nan
    return float(stats.spearmanr(est, truth)[0])


def kendall(est, truth):
    if len(est) < 3 or np.std(est) == 0 or np.std(truth) == 0:
        return np.nan
    return float(stats.kendalltau(est, truth)[0])


def regret_at_1(est, truth):
    est, truth = np.asarray(est), np.asarray(truth)
    r = truth.max() - truth[int(np.argmax(est))]
    spread = truth.max() - truth.min()
    return float(r), float(r / spread) if spread > 0 else 0.0


def precision_at_k(est, truth, k):
    k = min(k, len(est))
    return len(set(np.argsort(-np.asarray(est))[:k]) & set(np.argsort(-np.asarray(truth))[:k])) / k


def pairwise_accuracy(est, truth, tie=None):
    n, ok, tot = len(est), 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            if tie is not None and tie[i, j]:
                continue
            if truth[i] == truth[j]:
                continue
            tot += 1
            ok += (est[i] - est[j]) * (truth[i] - truth[j]) > 0
    return ok / tot if tot else np.nan


def bootstrap_corr_ci(est, truth, fn=spearman, n_boot=300, seed=0):
    rng = np.random.default_rng(seed)
    est, truth = np.asarray(est), np.asarray(truth)
    v = []
    for _ in range(n_boot):
        i = rng.integers(len(est), size=len(est))
        x = fn(est[i], truth[i])
        if not np.isnan(x):
            v.append(x)
    return (float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))) if v else (np.nan, np.nan)


def top3_contains_pick(est, truth):
    return float(int(np.argmax(est)) in set(np.argsort(-np.asarray(truth))[:3]))


def metrics_for(est, truth, tie=None, ci=True):
    est, truth = np.asarray(est, float), np.asarray(truth, float)
    ok = ~np.isnan(est)
    if tie is not None:
        tie = tie[np.ix_(ok, ok)]
    est, truth = est[ok], truth[ok]
    if len(est) < 3:
        return {}
    reg, nreg = regret_at_1(est, truth)
    out = {"spearman": spearman(est, truth), "kendall": kendall(est, truth), "regret@1": reg,
           "regret@1_norm": nreg, "p@1": precision_at_k(est, truth, 1), "p@3": precision_at_k(est, truth, 3),
           "p@5": precision_at_k(est, truth, 5), "top1_in_true_top3": top3_contains_pick(est, truth),
           "pairwise_acc": pairwise_accuracy(est, truth, tie), "mae": float(np.mean(np.abs(est - truth))),
           "bias": float(np.mean(est - truth)), "n": int(len(est))}
    if ci:
        out["spearman_lo"], out["spearman_hi"] = bootstrap_corr_ci(est, truth)
    return out


def family_metrics(df, est_col, truth_col="truth", fam_col="family"):
    """Per-family value error (bias / MAE) and within-family vs across-family Spearman (fairness check)."""
    rows = []
    for f, g in df.groupby(fam_col):
        e = g[est_col].to_numpy(float)
        ok = ~np.isnan(e)
        row = {"family": f, "n": int(ok.sum())}
        if ok.any():
            row["bias"] = float(np.mean(e[ok] - g[truth_col].to_numpy()[ok]))
            row["mae"] = float(np.mean(np.abs(e[ok] - g[truth_col].to_numpy()[ok])))
            row["within_spearman"] = spearman(e[ok], g[truth_col].to_numpy()[ok]) if ok.sum() >= 4 else np.nan
        rows.append(row)
    return rows
