"""Generalised advantage estimation for truncated windows (bootstrapping with V(s_H) at the last step)."""
import numpy as np


def gae(rew, V, v_last, gamma, lam):
    """rew, V: (N, H); v_last: (N,) value of the state after the last step. A_t = sum_k (gamma*lam)^k delta_{t+k},
    delta_t = r_{t+1} + gamma V(s_{t+1}) - V(s_t). Returns (A, deltas)."""
    N, H = rew.shape
    Vn = np.concatenate([V[:, 1:], v_last[:, None]], axis=1)
    delta = rew + gamma * Vn - V
    A = np.zeros_like(delta)
    run = np.zeros(N)
    for t in range(H - 1, -1, -1):
        run = delta[:, t] + gamma * lam * run
        A[:, t] = run
    return A, delta
