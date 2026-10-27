"""Agent 4: mean-variance hedging on scenarios; closed-form linear version; hedge-ratio frontier."""
import numpy as np
from scipy.optimize import minimize

KAPPAS = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)


def scen_costs_and_grad(q, Ls, pAs, pDA, rho, pi):
    """Per-scenario daily cost C_s (S,) and dC_s/dq (S,24)."""
    short = np.maximum(Ls - q, 0.0)
    surplus = np.maximum(q - Ls, 0.0)
    C = (pDA * q + pAs * short - rho * pAs * surplus + pi * short).sum(1)
    dC = np.where(q < Ls, pDA - pAs - pi, pDA - rho * pAs)
    return C, dC


def mv_objective(q, Ls, pAs, pDA, rho, pi, kappa, Cref):
    C, dC = scen_costs_and_grad(q, Ls, pAs, pDA, rho, pi)
    C, dC = C / Cref, dC / Cref
    m = C.mean()
    dm = dC.mean(0)
    cen = C - m
    var = (cen ** 2).mean()
    gvar = 2 * (cen[:, None] * (dC - dm)).mean(0)
    return m + kappa * var, dC.mean(0) + kappa * gvar


def mv_day(Ls, pAs, pDA, cap, rho, pi, kappa, Cref, q0):
    r = minimize(mv_objective, q0, args=(Ls, pAs, pDA, rho, pi, kappa, Cref), jac=True, method="L-BFGS-B",
                 bounds=[(0.0, c) for c in cap], options={"maxiter": 200})
    return r.x


def mv_days(P, sc, idx, q0, kappas=KAPPAS, S=300, seed=0):
    """{kappa: (n,24)}; starts from q0 (risk-neutral SAA solution), warm-starting along the kappa grid."""
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(sc.rL), min(S, len(sc.rL)), replace=False)
    out = {k: np.zeros((len(idx), 24)) for k in kappas}
    for i, d in enumerate(idx):
        Ls = P.Lhat[d] * (1 + sc.cL[d] + sc.rL[sel])
        pAs = np.maximum(P.pDA[d] + sc.cg[d] + sc.rg[sel], 0.0)
        q = q0[i].copy()
        for k in kappas:
            q = mv_day(Ls, pAs, P.pDA[d], P.cap[d], P.rho, P.pi, k, P.Cref, q)
            out[k][i] = q
    return out


def closed_form_linear(Ls, pAs, pDA, kappa, ridge=1e-9):
    """rho=1, pi=0: C = sum pA*L - g.q with g = pA - pDA. Unconstrained minimizer of E[C] + kappa Var[C]:
    q* = Sigma_g^{-1} (cov(g, a) + E[g]/(2 kappa)),  a = sum_h pA*L. kappa=inf gives the minimum-variance hedge."""
    g = pAs - pDA
    a = (pAs * Ls).sum(1)
    Sg = np.cov(g.T) + ridge * np.eye(g.shape[1])
    cga = ((g - g.mean(0)) * (a - a.mean())[:, None]).mean(0) * len(a) / (len(a) - 1)
    rhs = cga + (g.mean(0) / (2 * kappa) if kappa > 0 else 0.0)
    return np.linalg.solve(Sg, rhs)


def hedge_ratio_frontier(P, idx, Hs=np.linspace(0.8, 1.1, 31)):
    """Buy H*Lhat each hour: returns rows (H, mean, std, cvar95) of normalized daily cost on the given days."""
    from risk import cvar
    rows = []
    for H in Hs:
        q = np.clip(H * P.Lhat[idx], 0, P.cap[idx])
        c = P.ctilde(q, idx)
        rows.append((H, c.mean(), c.std(), cvar(c, 0.95)))
    return np.array(rows)
