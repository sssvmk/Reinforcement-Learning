"""Business/performance metrics for purchase schedules, plus bootstrap and multiple-testing helpers."""
import numpy as np
import costs
from risk import cvar, hedge_ratio


def agent_metrics(P, q, idx, q_ref=None):
    """Metrics for purchases q (n,24) on days idx. Normalized costs use P.Cref (train TSO-forecast mean)."""
    idx = np.asarray(idx)
    C = P.cost(q, idx)
    Ct = C / P.Cref
    bd = costs.cost_breakdown(q, P.L[idx], P.pDA[idx], P.pA[idx], P.rho, P.pi)
    out = {"mean_eur": C.mean(), "median_eur": np.median(C), "mean_norm": Ct.mean(), "median_norm": np.median(Ct),
           "std_norm": Ct.std(), "cvar95": cvar(Ct, .95), "cvar99": cvar(Ct, .99), "worst_norm": Ct.max(),
           "hedge_ratio": hedge_ratio(q, P.Lhat[idx]),
           "purchase_eur": bd["purchase"].mean(), "shortfall_eur": bd["shortfall"].mean(),
           "surplus_credit_eur": bd["surplus_credit"].mean(),
           "shortfall_share": bd["shortfall_mwh"].sum() / P.L[idx].sum(),
           "surplus_share": bd["surplus_wasted_mwh"].sum() / P.L[idx].sum(),
           "da_share": q.sum() / P.L[idx].sum()}
    months = np.asarray(P.dates[idx].to_period("M"))
    out["max_monthly_eur"] = max(C[months == m].sum() for m in np.unique(months))
    if q_ref is not None:
        Cr = P.cost(q_ref, idx)
        out["savings_eur_day"] = (Cr - C).mean()
        out["savings_pct"] = 100 * (Cr - C).mean() / Cr.mean()
        out["savings_eur_year"] = 365 * (Cr - C).mean()
        out["share_days_cheaper"] = (C < Cr).mean()
    return out


def block_bootstrap_ci(x, block=7, n_boot=2000, seed=0, stat=np.mean, level=0.95):
    """Moving-block bootstrap over a time-ordered series x; returns (point, lo, hi)."""
    x = np.asarray(x, float)
    n = len(x)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    starts_max = max(n - block, 0)
    est = np.empty(n_boot)
    for b in range(n_boot):
        st = rng.integers(0, starts_max + 1, nb)
        idx = (st[:, None] + np.arange(block)[None]).ravel()[:n]
        est[b] = stat(x[idx])
    a = (1 - level) / 2
    return stat(x), np.quantile(est, a), np.quantile(est, 1 - a)


def bootstrap_ci(x, n_boot=2000, seed=0, stat=np.mean, level=0.95):
    return block_bootstrap_ci(x, block=1, n_boot=n_boot, seed=seed, stat=stat, level=level)


def holm(pvals):
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(run, 1.0)
    return adj
