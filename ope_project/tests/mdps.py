"""Tiny MDPs and Baird's counterexample with exact (linear-algebra) answers, derived in code."""
from __future__ import annotations
import numpy as np


def tiny_exact(bR, gamma=0.9, piR=0.8):
    """States {1,2}; action L -> state 1, R -> state 2; reward 1 when next state is 2; x = (1, 0.5)."""
    X = np.array([[1.0], [0.5]])
    b = np.array([1 - bR, bR])
    pi = np.array([1 - piR, piR])
    P_b = np.tile(b, (2, 1))
    P_pi = np.tile(pi, (2, 1))
    # stationary distribution under b: next state independent of current -> rows identical
    d = b.copy()
    D = np.diag(d)
    r_pi = P_pi @ np.array([0.0, 1.0])          # expected one-step reward under pi (same for both states)
    r_pi = np.full(2, r_pi)
    A = X.T @ D @ (np.eye(2) - gamma * P_pi) @ X
    bb = X.T @ D @ r_pi
    w_td = float(np.linalg.solve(A, bb))
    f = np.linalg.solve(np.eye(2) - gamma * P_pi.T, d)
    Fm = np.diag(f)
    A_E = X.T @ Fm @ (np.eye(2) - gamma * P_pi) @ X
    b_E = X.T @ Fm @ r_pi
    w_etd = float(np.linalg.solve(A_E, b_E))
    var = gamma ** 2 * float((pi ** 2 / b).sum())
    return dict(X=X, d=d, A=A, b=bb, w_td=w_td, f=f, A_E=A_E, b_E=b_E, w_etd=w_etd, gamma2_Erho2=var,
                P_pi=P_pi, rho=pi / b, bvec=b, gamma=gamma)


def tiny_chain(bR, n, seed, gamma=0.9, piR=0.8, single_window=True):
    """Sample one long behavior trajectory. Because the next state depends only on the action, the chain is
    stationary from the start (state 1 w.p. 1-bR, 2 w.p. bR). Returns Xs (1,n+1,1), rho, R (1,n)."""
    rng = np.random.default_rng(seed)
    a = (rng.random(n + 1) < bR).astype(np.int64)      # a_t = 1 means R; s_{t+1} = 1 + a_t ; s_0 ~ mu
    s = np.empty(n + 1, dtype=np.int64)
    s[0] = a[-1]                                         # s_0 drawn like any other state
    s[1:] = a[:n]
    X = np.array([1.0, 0.5])
    Xs = X[s].reshape(1, n + 1, 1)
    pi = np.array([1 - piR, piR])
    b = np.array([1 - bR, bR])
    rho = (pi[a[:n]] / b[a[:n]]).reshape(1, n)
    R = (a[:n] == 1).astype(float).reshape(1, n)
    return Xs, rho, R


def tiny_windows(bR, W, H, seed, piR=0.8):
    """W independent windows of H transitions; window start state ~ stationary mu (as in a log with random starts)."""
    rng = np.random.default_rng(seed)
    a = (rng.random((W, H + 1)) < bR).astype(np.int64)
    s = np.empty((W, H + 1), dtype=np.int64)
    s[:, 0] = a[:, -1]
    s[:, 1:] = a[:, :H]
    X = np.array([1.0, 0.5])
    pi = np.array([1 - piR, piR])
    b = np.array([1 - bR, bR])
    rho = pi[a[:, :H]] / b[a[:, :H]]
    R = (a[:, :H] == 1).astype(float)
    return X[s][..., None], rho, R, a[:, :H], s


def baird():
    nS, d = 7, 8
    X = np.zeros((nS, d))
    for i in range(6):
        X[i, i] = 2.0
        X[i, 7] = 1.0
    X[6, 6] = 1.0
    X[6, 7] = 2.0
    return X


def baird_exact(gamma=0.99):
    X = baird()
    nS = 7
    # behavior: dashed (6/7) -> uniform over upper 6; solid (1/7) -> state 7. Stationary: uniform 1/7
    d = np.full(nS, 1.0 / 7)
    P_pi = np.zeros((nS, nS))
    P_pi[:, 6] = 1.0                                    # target always solid
    A = X.T @ np.diag(d) @ (np.eye(nS) - gamma * P_pi) @ X
    f = np.linalg.solve(np.eye(nS) - gamma * P_pi.T, d)
    A_E = X.T @ np.diag(f) @ (np.eye(nS) - gamma * P_pi) @ X
    return dict(X=X, d=d, A=A, A_E=A_E, f=f, P_pi=P_pi, gamma=gamma)


def baird_chain(n, seed, gamma=0.99):
    rng = np.random.default_rng(seed)
    X = baird()
    s = np.empty(n + 1, dtype=np.int64)
    solid = rng.random(n) < 1 / 7
    s[0] = rng.integers(0, 7)
    nxt = np.where(solid, 6, rng.integers(0, 6, n))
    s[1:] = nxt
    # action taken at s_t is solid iff the next state is 7 (dashed never leads to state 7)
    rho = np.where(solid, 7.0, 0.0).reshape(1, n)
    Xs = X[s].reshape(1, n + 1, 8)
    R = np.zeros((1, n))
    return Xs, rho, R
