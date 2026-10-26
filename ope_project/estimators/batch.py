"""Batch quantities and baselines: LSTD(0), MSPBE, emphatic PBE, WIS, linear FQE, oracle projection."""
from __future__ import annotations
import numpy as np
from .kernels import emphasis_trace


def flat_transitions(Xs):
    W, H1, d = Xs.shape
    return Xs[:, :-1].reshape(-1, d), Xs[:, 1:].reshape(-1, d)


def batch_AbC(Xs, rho, R, gamma, ridge=1e-6, weights=None):
    """A = mean(w_t rho x (x-gamma x')^T), b = mean(w_t rho r x), C = mean(w_t' x x^T) + ridge*I.
    weights=None -> behavior-weighted (plain). weights=F -> emphasis-weighted (A,b use F*rho; C uses F)."""
    X, Xp = flat_transitions(Xs)
    rr = rho.reshape(-1)
    r = R.reshape(-1)
    if weights is None:
        wt = np.ones_like(rr)
        wc = np.ones_like(rr)
    else:
        wt = weights.reshape(-1)
        wc = wt
    N = len(rr)
    A = (X * (wt * rr)[:, None]).T @ (X - gamma * Xp) / N
    b = (X * (wt * rr * r)[:, None]).sum(axis=0) / N
    C = (X * wc[:, None]).T @ X / N + ridge * np.eye(X.shape[1])
    return A, b, C


def mspbe(w, A, b, C):
    g = b - A @ w
    return float(g @ np.linalg.solve(C, g))


def lstd(Xs, rho, R, gamma, ridge):
    A, b, _ = batch_AbC(Xs, rho, R, gamma)
    d = A.shape[0]
    return np.linalg.solve(A + ridge * np.eye(d), b)


def emphatic_stats(rho, gamma, reset=True, clipF=0.0):
    F = emphasis_trace(rho, gamma, bool(reset), float(clipF))
    return F


def wis(R, logrho, gamma, clip):
    """Whole-trajectory weighted IS with clipped cumulative weights. Returns (J_hat, ESS)."""
    H = R.shape[1]
    G = (R * gamma ** np.arange(H)).sum(axis=1)
    lw = np.minimum(logrho.sum(axis=1), np.log(clip))
    w = np.exp(lw - lw.max())
    return float((w * G).sum() / w.sum()), float(w.sum() ** 2 / (w ** 2).sum())


def fqe(Xs, act, R, Ppi_next, Ppi_start, gamma, rounds, ridge, n_actions, X0=None):
    """Linear fitted-Q evaluation over (x (x) one-hot a). Ppi_next: (W,H,nA) target probs at next states,
    Ppi_start: (W,nA) at window starts. Exact expectation over pi's action probabilities."""
    X, Xp = flat_transitions(Xs)
    a = act.reshape(-1)
    r = R.reshape(-1)
    N, d = X.shape
    Pn = Ppi_next.reshape(N, n_actions)
    idx = [np.where(a == k)[0] for k in range(n_actions)]
    G = [X[i].T @ X[i] / N + ridge * np.eye(d) for i in idx]
    Gi = [np.linalg.inv(g) for g in G]
    Wq = [np.zeros(d) for _ in range(n_actions)]
    for _ in range(rounds):
        qn = np.stack([Xp @ Wq[k] for k in range(n_actions)], axis=1)
        y = r + gamma * (Pn * qn).sum(axis=1)
        Wq = [Gi[k] @ (X[idx[k]].T @ y[idx[k]] / N) for k in range(n_actions)]
    X0 = Xs[:, 0] if X0 is None else X0
    q0 = np.stack([X0 @ Wq[k] for k in range(n_actions)], axis=1)
    return float((Ppi_start * q0).sum(axis=1).mean()), Wq


def oracle_projection(X_states, v_true):
    """Least-squares fit of v_true on features (not a feasible estimator): representation-error floor."""
    w, *_ = np.linalg.lstsq(X_states, v_true, rcond=None)
    return w
