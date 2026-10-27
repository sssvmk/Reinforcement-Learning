"""Trading replay: one instrument, daily steps, long fraction in [0, 1].

State at the close of day t uses only data up to day t plus the previous position. The position chosen at close t earns
R[t] = close[t+1]/close[t] - 1.  reward_{t+1} = pos_t * R[t] - cost * |pos_t - pos_{t-1}|; windows start flat.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class TradeData:
    X: np.ndarray          # raw whitelist features (n, 5)
    R: np.ndarray          # next-day return earned by the position chosen at close t
    Z: np.ndarray          # features standardized with TRAIN statistics only
    mu: np.ndarray
    sd: np.ndarray
    cost: float            # fraction per unit of turnover
    dates: np.ndarray


def make_trade_data(data, train_range, cost):
    lo, hi = train_range
    mu, sd = data.X[lo:hi].mean(axis=0), np.maximum(data.X[lo:hi].std(axis=0), 1e-12)
    return TradeData(data.X, data.R, (data.X - mu) / sd, mu, sd, cost, data.dates)


def simulate(policy, theta, env, starts, H, rng=None, mode="mean", record=False, detail=False):
    starts = np.asarray(starts)
    N = len(starts)
    prev = np.zeros(N)
    net = np.empty((N, H))
    pos_all = np.empty((N, H)) if detail else None
    gross = np.empty((N, H)) if detail else None
    rec = None
    for j in range(H):
        t = starts + j
        st = {"t": t, "prev": prev, "j": j, "env": env}
        pos, aux = policy.act(theta, st, rng, mode)
        pos = np.clip(pos, 0.0, 1.0)
        if record:
            sc = policy.score(theta, aux)
            if rec is None:
                phi0 = policy.phi_of(aux)
                rec = {"rew": np.zeros((N, H)), "phi": np.zeros((N, H, phi0.shape[1])), "score": np.zeros((N, H, sc.shape[1])),
                       "ent": np.zeros((N, sc.shape[1])), "act": np.zeros((N, H))}
            if hasattr(policy, "act_of"):
                rec["act"][:, j] = policy.act_of(aux)
            rec["score"][:, j] = sc
            rec["phi"][:, j] = policy.phi_of(aux)
            rec["ent"] += policy.ent_grad(theta, aux)
        g = pos * env.R[t]
        r = g - env.cost * np.abs(pos - prev)
        net[:, j] = r
        if record:
            rec["rew"][:, j] = r
        if detail:
            pos_all[:, j], gross[:, j] = pos, g
        prev = pos
    out = {"total": net.sum(axis=1)}
    if record:
        out.update(rec)
    if detail:
        out.update({"net": net, "pos": pos_all, "gross": gross})
    return out
