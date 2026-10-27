"""Risk measures on normalized daily costs."""
import numpy as np


def cvar(x, alpha=0.95):
    """Rockafellar-Uryasev: min_eta eta + E[(x-eta)+]/(1-alpha); the minimizer is the alpha-quantile (VaR)."""
    x = np.asarray(x, float).ravel()
    eta = np.quantile(x, alpha, method="inverted_cdf")
    return eta + np.mean(np.maximum(x - eta, 0.0)) / (1.0 - alpha)


def cvar_tail_mean(x, alpha=0.95):
    """Average of the worst (1-alpha) share (fractional weight on the boundary observation)."""
    x = np.sort(np.asarray(x, float).ravel())[::-1]
    k = (1 - alpha) * len(x)
    n = int(np.floor(k + 1e-12))
    tot = x[:n].sum() + (k - n) * (x[n] if n < len(x) else 0.0)
    return tot / k


def var_(x):
    return float(np.var(np.asarray(x, float)))


def mean_variance(x, kappa):
    x = np.asarray(x, float)
    return float(x.mean() + kappa * x.var())


def hedge_ratio(q, Lhat):
    return float(np.sum(q) / np.sum(Lhat))


def summarize(Ct, alpha=0.95):
    Ct = np.asarray(Ct, float)
    return {"mean": Ct.mean(), "median": np.median(Ct), "std": Ct.std(), "cvar95": cvar(Ct, 0.95),
            "cvar99": cvar(Ct, 0.99), "worst": Ct.max()}
