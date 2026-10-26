"""Ground truth. Trading: exact finite-horizon dynamic programming over (day, position), which is exact because
prices are exogenous and the state is only (t, pos). V_H(t,pos) = expected discounted return over the next H steps
(or until the split ends). Window-start truth J = V_H(start, flat)."""
from __future__ import annotations
import numpy as np


def policy_probs_table(G, eps, n_actions=2):
    """G: (N,2) greedy actions -> P (N,2,nA) with (1-eps)*onehot + eps/nA."""
    P = np.full(G.shape + (n_actions,), eps / n_actions)
    ii, jj = np.indices(G.shape)
    P[ii, jj, G] += 1.0 - eps
    return P


def dp_values(fwd, lo, hi, P, cost, H, gamma):
    """Backward induction H steps. Returns V (N,2); states outside [lo, hi-1) get 0."""
    N = len(fwd)
    valid = np.zeros(N)
    valid[lo:hi - 1] = 1.0
    V = np.zeros((N + 1, 2))
    for _ in range(H):
        Vn = np.zeros((N + 1, 2))
        for p in (0, 1):
            q = 0.0
            for a in (0, 1):
                Qa = a * fwd - cost * abs(a - p) + gamma * V[1:N + 1, a]
                q = q + P[:, p, a] * Qa
            Vn[:N, p] = valid * q
        V = Vn
    return V[:N]


def trading_truth(data, rng_unused, G_tables, eps_t, log, lo, hi, cost_bps, H, gamma, state_idx):
    """Return J (A,), Jw (A,W) per-window truth at the logged window starts, vs (A,S) at sampled states."""
    cost = cost_bps / 1e4
    starts = log.start.astype(int)
    A = len(G_tables)
    Jw = np.zeros((A, len(starts)))
    vs = np.zeros((A, len(state_idx[0])))
    for i, G in enumerate(G_tables):
        V = dp_values(data.fwd_ret, lo, hi, policy_probs_table(G, eps_t), cost, H, gamma)
        Jw[i] = V[starts, 0]
        vs[i] = V[state_idx[0], state_idx[1]]
    return Jw.mean(axis=1), Jw, vs


def behavior_truth(data, eps_b, lo, hi, cost_bps, H, gamma, log):
    cost = cost_bps / 1e4
    G = data.sar_up[:, None].repeat(2, axis=1).astype(np.int64)
    V = dp_values(data.fwd_ret, lo, hi, policy_probs_table(G, eps_b), cost, H, gamma)
    return float(V[log.start.astype(int), 0].mean())
