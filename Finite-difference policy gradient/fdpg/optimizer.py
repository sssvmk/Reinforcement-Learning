"""Finite-difference policy-gradient optimizer: projected gradient ascent (or Adam) on internal coordinates."""
from dataclasses import dataclass, asdict
import numpy as np
from .estimators import ESTIMATORS, evals_per_estimate


class Scaler:
    """Maps physical parameters to internal (unit-order) coordinates: u = p / scale."""

    def __init__(self, scale):
        self.scale = np.asarray(scale, dtype=float)

    def to_internal(self, p):
        return np.asarray(p, dtype=float) / self.scale

    def to_phys(self, u):
        return np.asarray(u, dtype=float) * self.scale


def project(u, lo, hi):
    return np.minimum(np.maximum(u, lo), hi)


@dataclass
class HParams:
    est: str = "central"
    eps: float = 0.1
    gamma: float = 0.1      # eps_k = eps / (k+1)^gamma
    a: float = 0.5          # a_k = a / (k + A)^alpha
    A: float = 10.0
    alpha: float = 0.602
    m: int = 4              # SPSA draws
    K: int = 0              # regression / gaussian perturbations (0 -> default)
    clip: float = 5.0       # gradient-norm clip
    adam: bool = False
    crn: bool = True
    budget: int = 20000     # training rollouts
    max_iter: int = 100000
    val_every: int = 1
    patience: int = 0       # 0 disables early stopping (in validation checks)
    ridge_rel: float = 1e-6

    def asdict(self):
        return asdict(self)


def run_fd(Jtrain, Jval, u0, lo, hi, hp, seed, Jdiag=None):
    """Returns dict with curve (iteration, rollouts, val, train_diag, u) and the validation-selected u."""
    rng = np.random.default_rng(seed)
    est = ESTIMATORS[hp.est]
    n = len(u0)
    u = project(np.asarray(u0, float), lo, hi)
    r0 = Jtrain.rollouts
    cost = evals_per_estimate(hp.est, n, hp.m, hp.K or None) * Jtrain.n_paths
    mom, vel = np.zeros(n), np.zeros(n)
    curve, best_val, best_u, bad, k = [], -np.inf, u.copy(), 0, 0

    def log(k):
        nonlocal best_val, best_u, bad
        v = Jval(u, Jval.fixed_paths)
        d = Jdiag(u, Jdiag.fixed_paths) if Jdiag is not None else np.nan
        curve.append((k, Jtrain.rollouts - r0, v, d, u.copy()))
        if np.isfinite(v) and v > best_val:
            best_val, best_u, bad = v, u.copy(), 0
        else:
            bad += 1

    log(0)
    while k < hp.max_iter and (Jtrain.rollouts - r0) + cost <= hp.budget:
        eps = hp.eps / (k + 1) ** hp.gamma
        g = est(Jtrain, u, rng, eps=eps, m=hp.m, K=hp.K or None, crn=hp.crn, ridge_rel=hp.ridge_rel)
        if not np.all(np.isfinite(g)):
            g = np.zeros(n)
        nrm = np.linalg.norm(g)
        if hp.clip and nrm > hp.clip:
            g = g * hp.clip / nrm
        ak = hp.a / (k + hp.A) ** hp.alpha
        if hp.adam:
            mom = 0.9 * mom + 0.1 * g
            vel = 0.999 * vel + 0.001 * g * g
            step = (mom / (1 - 0.9 ** (k + 1))) / (np.sqrt(vel / (1 - 0.999 ** (k + 1))) + 1e-8)
        else:
            step = g
        u = project(u + ak * step, lo, hi)
        k += 1
        if k % hp.val_every == 0:
            log(k)
            if hp.patience and bad >= hp.patience:
                break
    if curve[-1][0] != k:
        log(k)
    return {"u_best": best_u, "u_last": u.copy(), "val_best": best_val, "iters": k,
            "rollouts": Jtrain.rollouts - r0, "curve": curve, "hp": hp.asdict()}
