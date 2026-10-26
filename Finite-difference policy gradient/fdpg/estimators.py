"""Finite-difference gradient estimators. Each takes (J, u, rng, **kw) and returns the gradient estimate (n,)."""
import numpy as np
from .crn import PathProvider


def forward(J, u, rng, eps=0.1, crn=True, **_):
    P = PathProvider(J, rng, crn)
    j0 = P.eval(u)
    g = np.empty(len(u))
    for i in range(len(u)):
        e = np.zeros(len(u)); e[i] = eps
        g[i] = (P.eval(u + e) - j0) / eps
    return g


def central(J, u, rng, eps=0.1, crn=True, **_):
    P = PathProvider(J, rng, crn)
    g = np.empty(len(u))
    for i in range(len(u)):
        e = np.zeros(len(u)); e[i] = eps
        g[i] = (P.eval(u + e) - P.eval(u - e)) / (2 * eps)
    return g


def spsa(J, u, rng, eps=0.1, m=4, crn=True, **_):
    g = np.zeros(len(u))
    for _k in range(m):
        P = PathProvider(J, rng, crn)
        delta = rng.choice([-1.0, 1.0], size=len(u))
        g += (P.eval(u + eps * delta) - P.eval(u - eps * delta)) / (2 * eps) * delta
    return g / m


def regression(J, u, rng, eps=0.1, K=None, ridge_rel=1e-6, crn=True, D=None, **_):
    """g = (D'D + ridge I)^-1 D' dJ with rows of D ~ N(0, eps^2 I) (or an explicit D)."""
    n = len(u)
    P = PathProvider(J, rng, crn)
    if D is None:
        D = rng.normal(0.0, eps, size=(K or 2 * n, n))
    j0 = P.eval(u)
    dJ = np.array([P.eval(u + d) - j0 for d in D])
    A = D.T @ D + ridge_rel * eps ** 2 * np.eye(n)
    return np.linalg.solve(A, D.T @ dJ)


def gaussian(J, u, rng, eps=0.1, K=None, crn=True, **_):
    """Antithetic Gaussian smoothing, sigma = eps."""
    n = len(u)
    K = K or max(2, n)
    P = PathProvider(J, rng, crn)
    g = np.zeros(n)
    for _k in range(K):
        z = rng.normal(size=n)
        g += (P.eval(u + eps * z) - P.eval(u - eps * z)) / 2 * z
    return g / (K * eps)


ESTIMATORS = {"forward": forward, "central": central, "spsa": spsa, "regression": regression, "gaussian": gaussian}


def evals_per_estimate(name, n, m=4, K=None):
    """Number of J evaluations (each of N paths) one gradient estimate costs."""
    return {"forward": n + 1, "central": 2 * n, "spsa": 2 * m, "regression": (K or 2 * n) + 1,
            "gaussian": 2 * (K or max(2, n))}[name]
