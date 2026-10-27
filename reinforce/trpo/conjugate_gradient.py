"""Conjugate gradient for (F + damping*I) x = g using only matrix-vector products."""
import numpy as np


def conjugate_gradient(fvp, g, iters=20, tol=1e-10, damping=0.0):
    """Returns (x, residual_norm_history). Stops when ||r||^2 < tol or after `iters` iterations."""
    x = np.zeros_like(g)
    r = g.copy()
    p = g.copy()
    rr = float(r @ r)
    hist = [np.sqrt(rr)]
    for _ in range(iters):
        if rr < tol:
            break
        Ap = fvp(p) + damping * p
        alpha = rr / max(float(p @ Ap), 1e-300)
        x = x + alpha * p
        r = r - alpha * Ap
        rr_new = float(r @ r)
        p = r + (rr_new / rr) * p
        rr = rr_new
        hist.append(np.sqrt(rr))
    return x, hist


def mean_diag_fisher(fvp, P):
    """Mean of the diagonal of F from P Fisher-vector products (small P), used for the damping rule d = frac * mean diag F."""
    if P <= 200:
        return float(np.mean([fvp(np.eye(P)[i])[i] for i in range(P)]))
    rng = np.random.default_rng(0)
    z = rng.choice([-1.0, 1.0], size=(8, P))                    # Hutchinson estimate of trace(F)/P for larger P
    return float(np.mean([z[i] @ fvp(z[i]) for i in range(8)]) / P)
