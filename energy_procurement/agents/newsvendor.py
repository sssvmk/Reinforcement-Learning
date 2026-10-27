"""Agent 2: newsvendor / critical fractile (Variant A: quantile rule; Variant B: joint-scenario SAA)."""
import numpy as np
from agents import forecast as F


def fractile(pDA, pA_hat, rho, pi):
    """Return tau (clipped) and regime flags: Cu<=0 -> zero purchase, Co<=0 -> cap."""
    Cu = (pA_hat - pDA) + pi
    Co = pDA - rho * pA_hat
    with np.errstate(divide="ignore", invalid="ignore"):
        tau = Cu / (Cu + Co)
    tau = np.clip(np.nan_to_num(tau, nan=0.5), 0.01, 0.99)
    return tau, Cu, Co


def newsvendor_A(P, sc, idx, tau_shift=0.0):
    """q = Lhat*(1+Qhat_eL(tau)); tau from expected pA = pDA + E[g|x]. tau_shift is a (24,) or scalar offset."""
    idx = np.asarray(idx)
    pA_hat = P.pDA[idx] + sc.cg[idx]
    tau, Cu, Co = fractile(P.pDA[idx], pA_hat, P.rho, P.pi)
    tau = np.clip(tau + tau_shift, 0.01, 0.99)
    e = F.interp_quantile(sc.Q[idx], tau)
    q = np.clip(P.Lhat[idx] * (1 + e), 0, P.cap[idx])
    q = np.where(Co <= 0, P.cap[idx], q)
    q = np.where(Cu <= 0, 0.0, q)
    return q, tau


def saa_quantity(Ls, pAs, pDA, cap, rho, pi):
    """Minimize the sample-average hourly cost over q in [0, cap]. Ls, pAs: (..., S); pDA, cap: (...)."""
    order = np.argsort(Ls, axis=-1)
    Lo = np.take_along_axis(Ls, order, -1)
    pAo = np.take_along_axis(pAs, order, -1)
    a = pDA[..., None] - rho * pAo           # slope when q exceeds L_s (surplus)
    b = pDA[..., None] - pAo - pi            # slope when q is below L_s (shortfall)
    S = Ls.shape[-1]
    # slope after passing k sorted scenarios: sum_{i<=k} a_i + sum_{i>k} b_i, k = 0..S
    csa = np.concatenate([np.zeros(a.shape[:-1] + (1,)), np.cumsum(a, -1)], -1)
    csb = np.concatenate([np.cumsum(b[..., ::-1], -1)[..., ::-1], np.zeros(b.shape[:-1] + (1,))], -1)
    slope = csa + csb
    k = np.argmax(slope >= 0, axis=-1)
    none = ~(slope >= 0).any(-1)
    k = np.where(none, S, k)
    Lpad = np.concatenate([np.zeros(Lo.shape[:-1] + (1,)), Lo], -1)  # k=0 -> q=0
    q = np.take_along_axis(Lpad, k[..., None], -1)[..., 0]
    q = np.where(none, cap, q)
    return np.clip(q, 0.0, cap)


def newsvendor_B(P, sc, idx, S=None, chunk=64, seed=0):
    """Sample-average newsvendor from whole-day scenarios (load and price drawn jointly)."""
    idx = np.asarray(idx)
    rng = np.random.default_rng(seed)
    rL, rg = sc.rL, sc.rg
    if S is not None and S < len(rL):
        sel = rng.choice(len(rL), S, replace=False)
        rL, rg = rL[sel], rg[sel]
    out = np.zeros((len(idx), 24))
    for i in range(0, len(idx), chunk):
        d = idx[i:i + chunk]
        Ls = P.Lhat[d][..., None] * (1 + sc.cL[d][..., None] + rL.T[None])
        pAs = np.maximum(P.pDA[d][..., None] + sc.cg[d][..., None] + rg.T[None], 0.0)
        out[i:i + chunk] = saa_quantity(Ls, pAs, P.pDA[d], P.cap[d], P.rho, P.pi)
    return out


def calibrate_tau_shift(P, sc, idx, steps=40, lr=0.02, eps=0.02, seed=0):
    """Decision-focused calibration: central finite-difference policy gradient on a per-hour-block tau offset,
    minimizing mean normalized training cost (common days in +/- evaluations)."""
    rng = np.random.default_rng(seed)
    blocks = np.repeat(np.arange(4), 6)  # 4 six-hour blocks
    th = np.zeros(4)
    hist = []
    for t in range(steps):
        days = rng.choice(idx, min(128, len(idx)), replace=False)
        g = np.zeros(4)
        for j in range(4):
            e = np.zeros(4); e[j] = eps
            cp = P.ctilde(newsvendor_A(P, sc, days, (th + e)[blocks])[0], days).mean()
            cm = P.ctilde(newsvendor_A(P, sc, days, (th - e)[blocks])[0], days).mean()
            g[j] = (cp - cm) / (2 * eps)
        th = np.clip(th - lr * g / (np.linalg.norm(g) + 1e-12), -0.3, 0.3)
        hist.append(P.ctilde(newsvendor_A(P, sc, days, th[blocks])[0], days).mean())
    return th[blocks], hist
