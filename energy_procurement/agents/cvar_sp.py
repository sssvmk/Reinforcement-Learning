"""Agent 3: stochastic program with CVaR (Rockafellar-Uryasev LP), solved per delivery day with HiGHS.

Hourly cost is max of two linear pieces (shortfall / surplus), so one epigraph variable w_sh per
scenario-hour is exact for rho <= 1 and equivalent to the (sh, sp) formulation in the task text.
Costs are divided by Cref so lambda acts on normalized daily cost.
"""
import time
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
from concurrent.futures import ProcessPoolExecutor

LAMBDAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def solve_day(Ls, pAs, pDA, cap, Cref, rho, pi, lams, alpha):
    """Ls, pAs: (S,24); pDA, cap: (24,). Returns dict lam -> (q, in-sample mean, in-sample cvar, seconds)."""
    S = Ls.shape[0]
    n_q, n_w = 24, S * 24
    iq, iw, ie = 0, n_q, n_q + n_w
    iu = ie + 1
    nv = iu + S
    s_idx = np.repeat(np.arange(S), 24)
    h_idx = np.tile(np.arange(24), S)
    k = np.arange(n_w)
    # rows 0..n_w-1 : shortfall piece, n_w..2n_w-1 : surplus piece, then S rows for u
    b1 = pDA[h_idx] - pAs[s_idx, h_idx] - pi
    b2 = pDA[h_idx] - rho * pAs[s_idx, h_idx]
    rows = np.concatenate([k, k, n_w + k, n_w + k, 2 * n_w + np.repeat(np.arange(S), 24), 2 * n_w + np.arange(S), 2 * n_w + np.arange(S)])
    cols = np.concatenate([iw + k, iq + h_idx, iw + k, iq + h_idx, iw + k, np.full(S, ie), iu + np.arange(S)])
    vals = np.concatenate([-np.ones(n_w), b1, -np.ones(n_w), b2, np.full(n_w, 1.0 / Cref), -np.ones(S), -np.ones(S)])
    # constraints in <= form: -w + b q <= -(const)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(2 * n_w + S, nv))
    rhs = np.concatenate([-(pAs[s_idx, h_idx] + pi) * Ls[s_idx, h_idx], -rho * pAs[s_idx, h_idx] * Ls[s_idx, h_idx], np.zeros(S)])
    bounds = [(0.0, c) for c in cap] + [(None, None)] * n_w + [(None, None)] + [(0.0, None)] * S
    out = {}
    for lam in lams:
        c = np.zeros(nv)
        c[iw:iw + n_w] = 1.0 / (S * Cref)
        if lam > 0:
            c[ie] = lam
            c[iu:] = lam / ((1 - alpha) * S)
        t0 = time.time()
        r = linprog(c, A_ub=A, b_ub=rhs, bounds=bounds, method="highs")
        q = r.x[:n_q]
        out[lam] = (q, time.time() - t0)
    return out


def _job(args):
    return solve_day(*args)


def solve_days(P, sc, idx, lams=LAMBDAS, alpha=0.95, S=300, n_jobs=12, seed=0):
    """Return {lam: (n,24) purchases} and mean solve time per day-LP."""
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(sc.rL), min(S, len(sc.rL)), replace=False)
    jobs = []
    for d in idx:
        Ls = P.Lhat[d] * (1 + sc.cL[d] + sc.rL[sel])
        pAs = np.maximum(P.pDA[d] + sc.cg[d] + sc.rg[sel], 0.0)
        jobs.append((Ls, pAs, P.pDA[d], P.cap[d], P.Cref, P.rho, P.pi, tuple(lams), alpha))
    if n_jobs > 1:
        with ProcessPoolExecutor(n_jobs) as ex:
            res = list(ex.map(_job, jobs, chunksize=4))
    else:
        res = [_job(j) for j in jobs]
    Q = {lam: np.stack([r[lam][0] for r in res]) for lam in lams}
    t = {lam: float(np.mean([r[lam][1] for r in res])) for lam in lams}
    return Q, t
