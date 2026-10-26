"""Linear off-policy value estimators (numba). Notation: x=x(S_t), xp=x(S_{t+1}), rho=pi/b, delta=R+gamma*w.xp-w.x.
All terms in one update use the pre-update w and v.
Method ids: 0 TD(0) semi-gradient, 1 GTD2, 2 TDC, 3 ETD(lambda)
vform: 0 = rho*(delta - v.x)*x   (book Sec. 11.7);  1 = rho*delta*x - (v.x)*x   (book eq. 12.30 at lambda=0)
"""
from __future__ import annotations
import numpy as np

try:
    from numba import njit, prange
    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False
    prange = range

    def njit(*a, **k):
        if a and callable(a[0]):
            return a[0]
        return lambda f: f

TD, GTD2, TDC, ETD = 0, 1, 2, 3


@njit(cache=True)
def one_step(method, w, v, x, xp, rho, R, alpha, beta, gamma, lam, vform, F_carry, e, clipF):
    """One update in place on w, v, e. Returns (delta, F_t, new F_carry). F_carry = gamma*rho_{t-1}*F_{t-1}."""
    d = w.shape[0]
    wx = 0.0
    wxp = 0.0
    xv = 0.0
    for i in range(d):
        wx += w[i] * x[i]
        wxp += w[i] * xp[i]
        xv += v[i] * x[i]
    delta = R + gamma * wxp - wx
    F = 0.0
    if method == 0:
        c = alpha * rho * delta
        for i in range(d):
            w[i] += c * x[i]
    elif method == 1:
        for i in range(d):
            w[i] += alpha * rho * (x[i] - gamma * xp[i]) * xv
        for i in range(d):
            if vform == 0:
                v[i] += beta * rho * (delta - xv) * x[i]
            else:
                v[i] += beta * (rho * delta - xv) * x[i]
    elif method == 2:
        for i in range(d):
            w[i] += alpha * rho * (delta * x[i] - gamma * xp[i] * xv)
        for i in range(d):
            if vform == 0:
                v[i] += beta * rho * (delta - xv) * x[i]
            else:
                v[i] += beta * (rho * delta - xv) * x[i]
    else:
        F = F_carry + 1.0
        if clipF > 0.0 and F > clipF:
            F = clipF
        M = lam + (1.0 - lam) * F
        for i in range(d):
            e[i] = rho * (gamma * lam * e[i] + M * x[i])
            w[i] += alpha * delta * e[i]
        F_carry = gamma * rho * F
    return delta, F, F_carry


@njit(cache=True)
def run_kernel(method, Xs, rho, R, order, alpha0, kappa, gamma, lam, vform, reset, clipF,
               decay_p, decay_k0, n_ckpt, w0):
    """Run `order.shape[0]` epochs over windows in the given orders. Xs (W,H+1,d), rho,R (W,H).
    Returns snaps (n_ckpt,d), w_final, w_tail (mean of last 20% iterates), diverged flag, n_updates."""
    n_ep, W = order.shape
    H = R.shape[1]
    d = Xs.shape[2]
    w = w0.copy()
    v = np.zeros(d)
    e = np.zeros(d)
    total = n_ep * W * H
    snaps = np.full((n_ckpt, d), np.nan)
    every = max(total // n_ckpt, 1)
    tail_start = int(0.8 * total)
    wsum = np.zeros(d)
    ntail = 0
    k = 0
    ci = 0
    F_carry = 0.0
    diverged = False
    for ep in range(n_ep):
        for oi in range(W):
            wi = order[ep, oi]
            if reset:
                F_carry = 0.0
                for i in range(d):
                    e[i] = 0.0
            for t in range(H):
                a = alpha0
                if decay_p > 0.0:
                    a = alpha0 / (1.0 + k / decay_k0) ** decay_p
                delta, F, F_carry = one_step(method, w, v, Xs[wi, t], Xs[wi, t + 1], rho[wi, t], R[wi, t],
                                             a, kappa * a, gamma, lam, vform, F_carry, e, clipF)
                k += 1
                if k > tail_start:
                    for i in range(d):
                        wsum[i] += w[i]
                    ntail += 1
                if (not np.isfinite(delta)) or abs(delta) > 1e9:
                    diverged = True
                    break
                if k % every == 0 and ci < n_ckpt:
                    nrm = 0.0
                    for i in range(d):
                        nrm += w[i] * w[i]
                    if (not np.isfinite(nrm)) or nrm > 1e12:
                        diverged = True
                        break
                    snaps[ci, :] = w
                    ci += 1
            if diverged:
                break
        if diverged:
            break
    if diverged:
        wt = np.full(d, np.nan)
        wf = np.full(d, np.nan)
    else:
        wf = w.copy()
        wt = wsum / max(ntail, 1)
    return snaps, wf, wt, diverged, k


@njit(cache=True)
def emphasis_trace(rho, gamma, reset, clipF):
    """Followon trace F_t for a log with window resets (reset=True) or concatenated (False). Returns F (W,H)."""
    W, H = rho.shape
    F = np.zeros((W, H))
    carry = 0.0
    for wi in range(W):
        if reset:
            carry = 0.0
        for t in range(H):
            f = carry + 1.0
            if clipF > 0.0 and f > clipF:
                f = clipF
            F[wi, t] = f
            carry = gamma * rho[wi, t] * f
    return F


@njit(parallel=True, cache=True)
def run_many(Xs, rho_all, R, order, agent_idx, method, alpha, kappa, lam, vform, reset, clipF,
             decay_p, decay_k0, n_ckpt, gamma):
    """Run K independent tasks in parallel (prange). Task k uses rho_all[agent_idx[k]] and its own hyper-parameters.
    Returns snaps (K,n_ckpt,d), w_final (K,d), w_tail (K,d), diverged (K,), n_updates (K,)."""
    K = agent_idx.shape[0]
    d = Xs.shape[2]
    snaps = np.full((K, n_ckpt, d), np.nan)
    wf = np.zeros((K, d))
    wt = np.zeros((K, d))
    div = np.zeros(K, dtype=np.bool_)
    nup = np.zeros(K, dtype=np.int64)
    w0 = np.zeros(d)
    for k in prange(K):
        s, f, t, dv, n = run_kernel(method[k], Xs, rho_all[agent_idx[k]], R, order, alpha[k], kappa[k], gamma,
                                    lam[k], vform[k], reset[k], clipF[k], decay_p, decay_k0, n_ckpt, w0)
        snaps[k] = s
        wf[k] = f
        wt[k] = t
        div[k] = dv
        nup[k] = n
    return snaps, wf, wt, div, nup


def make_orders(W, n_epochs, rng):
    return np.stack([rng.permutation(W) for _ in range(n_epochs)]).astype(np.int64)


def run_method(method, Xs, rho, R, alpha, kappa=10.0, gamma=0.95, lam=0.0, vform=0, reset=True, clipF=0.0,
               n_epochs=20, n_ckpt=20, decay_p=0.0, decay_k0=20000.0, w0=None, seed=0, order=None):
    d = Xs.shape[2]
    if order is None:
        order = make_orders(Xs.shape[0], n_epochs, np.random.default_rng(seed))
    if w0 is None:
        w0 = np.zeros(d)
    out = run_kernel(method, Xs, rho, R, order, float(alpha), float(kappa), float(gamma), float(lam), int(vform),
                     bool(reset), float(clipF), float(decay_p), float(decay_k0), int(n_ckpt), w0.astype(np.float64))
    return dict(snaps=out[0], w=out[1], w_tail=out[2], diverged=bool(out[3]), n_updates=int(out[4]))
