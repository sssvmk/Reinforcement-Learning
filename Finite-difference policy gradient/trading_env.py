"""Trading replay: one instrument, daily steps, long-only position fraction in [0, 1]."""
import numpy as np


def simulate(policy, theta, X, R, starts, H, cost, detail=False):
    """Vectorized over windows. Features at close of day t pick pos_t, which earns R[t] = close[t+1]/close[t]-1.

    reward_{t+1} = pos_t * R[t] - cost * |pos_t - pos_{t-1}| ; windows start flat (pos_{-1} = 0).
    cost is a fraction per unit of turnover (5 bps -> 5e-4).
    """
    starts = np.asarray(starts)
    N = len(starts)
    prev = np.zeros(N)
    net = np.empty((N, H))
    pos_all = np.empty((N, H)) if detail else None
    gross = np.empty((N, H)) if detail else None
    for j in range(H):
        t = starts + j
        pos = np.clip(policy.act(theta, X[t], prev), 0.0, 1.0)
        g = pos * R[t]
        net[:, j] = g - cost * np.abs(pos - prev)
        if detail:
            pos_all[:, j], gross[:, j] = pos, g
        prev = pos
    if detail:
        return {"net": net, "pos": pos_all, "gross": gross}
    return net.sum(axis=1)
