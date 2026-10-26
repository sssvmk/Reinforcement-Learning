"""Derivative-free baselines at equal training-rollout budget. Each returns {'u_best', 'trace': [(rollouts, u_best_so_far)]}."""
import itertools
import numpy as np
from scipy.optimize import minimize
from .optimizer import project


def grid_search(J, lo, hi, budget, rng, max_dim=2):
    n = len(lo)
    if n > max_dim or n == 0:
        return None
    r0 = J.rollouts
    n_eval = max(1, budget // J.n_paths)
    pts = max(2, int(np.floor(n_eval ** (1.0 / n))))
    axes = [np.linspace(l, h, pts) for l, h in zip(lo, hi)]
    paths = J.draw(rng)  # one shared path set for all grid points
    best, bu, trace = -np.inf, None, []
    for p in itertools.product(*axes):
        u = np.array(p)
        v = J(u, paths)
        if v > best:
            best, bu = v, u
        trace.append((J.rollouts - r0, bu.copy()))
    return {"u_best": bu, "trace": trace, "val_best": None}


def random_search(J, lo, hi, budget, rng):
    r0 = J.rollouts
    paths = J.draw(rng)
    best, bu, trace = -np.inf, None, []
    for _ in range(max(1, budget // J.n_paths)):
        u = rng.uniform(lo, hi)
        v = J(u, paths)
        if v > best:
            best, bu = v, u
        trace.append((J.rollouts - r0, bu.copy()))
    return {"u_best": bu, "trace": trace}


def nelder_mead(J, u0, lo, hi, budget, rng):
    """Nelder-Mead on the noisy objective (fresh N paths per evaluation), projected into bounds."""
    r0 = J.rollouts
    best, bu, trace = [-np.inf], [project(np.asarray(u0, float), lo, hi)], []
    maxfev = max(2, budget // J.n_paths)

    def f(x):
        x = project(x, lo, hi)
        v = J(x, J.draw(rng))
        if v > best[0]:
            best[0], bu[0] = v, x.copy()
        trace.append((J.rollouts - r0, bu[0].copy()))
        return -v

    minimize(f, u0, method="Nelder-Mead", options={"maxfev": maxfev, "xatol": 1e-4, "fatol": 1e-6})
    return {"u_best": bu[0], "trace": trace}


def grid_tune(J, axes, rng=None, n_paths=None):
    """Hand-tuned benchmark: exhaustive grid over named axes on a fixed path set (train period only). Returns (u_best, value)."""
    paths = J.draw(rng or np.random.default_rng(0))
    best, bu = -np.inf, None
    for p in itertools.product(*axes):
        u = np.array(p, dtype=float)
        v = J(u, paths)
        if v > best:
            best, bu = v, u
    return bu, best
