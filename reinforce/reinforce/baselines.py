"""Baselines for REINFORCE. All return an advantage array A[i, t] = G[i, t] - b[i, t] (b must not depend on episode i's own actions
unless it is a leave-one-out or previous-iteration quantity, which keeps the gradient unbiased)."""
import numpy as np

BASELINES = ["none", "loo_const", "loo_time", "linear", "loo_time_std"]


def rewards_to_go(rew, gamma):
    G = np.zeros_like(rew)
    run = np.zeros(rew.shape[0])
    for t in range(rew.shape[1] - 1, -1, -1):
        run = rew[:, t] + gamma * run
        G[:, t] = run
    return G


def _vf_feats(phi, H):
    t = (np.arange(phi.shape[1]) / H)[None, :, None] * np.ones((phi.shape[0], 1, 1))
    return np.concatenate([phi, t, t ** 2], axis=2)


class BaselineState:
    """Holds the learned linear value function (fit on the previous iteration, so it is independent of the current batch)."""

    def __init__(self, ridge=1e-3):
        self.w, self.ridge = None, ridge

    def fit(self, phi, G, H):
        X = _vf_feats(phi, H).reshape(-1, phi.shape[2] + 2)
        y = G.reshape(-1)
        A = X.T @ X + self.ridge * len(y) * np.eye(X.shape[1])
        self.w = np.linalg.solve(A, X.T @ y)


def advantages(kind, G, phi, st, H):
    N = G.shape[0]
    if kind == "none":
        return G.copy()
    if kind == "loo_const":
        tot = G.sum()
        b = (tot - G.sum(axis=1, keepdims=True)) / (G.size - G.shape[1])
        return G - b
    if kind in ("loo_time", "loo_time_std"):
        b = (G.sum(axis=0, keepdims=True) - G) / (N - 1)
        A = G - b
        if kind == "loo_time_std":
            A = A / (A.std() + 1e-8)
        return A
    if kind == "linear":
        if st.w is None:
            return G.copy()
        return G - (_vf_feats(phi, H) @ st.w)
    raise ValueError(kind)
