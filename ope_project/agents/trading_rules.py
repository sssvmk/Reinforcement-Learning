"""Rule-based trading agents. raw = [day_index, position]. Only whitelist features are used.
Every parameter grid is scored on the TRAIN period only; each family contributes candidates spread over
the train-score ranking (best and worst included) so that the pool has a wide range of quality."""
from __future__ import annotations
import numpy as np
from .base import Agent, pick_spread


def replay_table(G, fwd_ret, lo, hi, cost):
    """Greedy replay from flat over days [lo, hi-1): returns daily net returns. G: (N,2) greedy actions."""
    pos = 0
    out = np.empty(hi - 1 - lo)
    for i, t in enumerate(range(lo, hi - 1)):
        a = G[t, pos]
        out[i] = a * fwd_ret[t] - cost * abs(a - pos)
        pos = a
    return out


def _cands(data, rng):
    f = data.feat
    rsi, flag, dist = f["rsi"], f["flag"], f["dist"]
    ret = {1: f["ret1"], 5: f["ret5"], 20: f["ret20"]}
    N = data.N
    u = rng.random(N)

    def mk(fn):
        return lambda raw: fn(raw[:, 0].astype(int), raw[:, 1].astype(int))

    C = {}  # family -> list of (name, fn, params)
    C["buy_hold"] = [("buy_hold", mk(lambda t, p: np.ones_like(t)), {})]
    C["flat"] = [("always_flat", mk(lambda t, p: np.zeros_like(t)), {})]
    C["random"] = [(f"random_p{p}", mk(lambda t, pp, p=p: (u[t] < p).astype(int)), {"p": p}) for p in (0.2, 0.5, 0.8)]
    C["rsi_mr"] = [
        (f"rsi_mr_L{L}_U{U}",
         mk(lambda t, p, L=L, U=U: np.where(p == 0, rsi[t] < L, rsi[t] <= U).astype(int)), {"L": L, "U": U})
        for L in (20, 25, 30, 35, 40) for U in (50, 55, 60, 65, 70, 75)]
    C["sar_trend"] = [
        (f"sar_dist_gt{th}", mk(lambda t, p, th=th: (dist[t] > th).astype(int)), {"theta": th})
        for th in (-0.03, -0.01, 0.0, 0.01, 0.02, 0.04)]
    C["sar_flag"] = [("sar_flag_long", mk(lambda t, p: (flag[t] == 0).astype(int)), {})]
    C["sar_contra"] = [("sar_contrarian", mk(lambda t, p: (flag[t] == 1).astype(int)), {})]
    C["rsi_sar"] = [
        (f"rsi_sar_L{L}_U{U}",
         mk(lambda t, p, L=L, U=U: np.where(p == 0, (flag[t] == 0) & (rsi[t] < L),
                                            (flag[t] == 0) & (rsi[t] < U)).astype(int)), {"L": L, "U": U})
        for L in (40, 50, 60, 70) for U in (60, 70, 80)]
    C["momentum"] = [(f"mom{k}", mk(lambda t, p, k=k: (ret[k][t] > 0).astype(int)), {"k": k}) for k in (1, 5, 20)]
    C["anti_mom"] = [(f"antimom{k}", mk(lambda t, p, k=k: (ret[k][t] < 0).astype(int)), {"k": k}) for k in (1, 5, 20)]
    C["rsi_inv"] = [(f"rsi_gt{L}", mk(lambda t, p, L=L: (rsi[t] > L).astype(int)), {"L": L}) for L in (50, 55, 60, 65)]
    return C


PICKS = {"buy_hold": 1, "flat": 1, "random": 3, "rsi_mr": 5, "sar_trend": 3, "sar_flag": 1, "sar_contra": 1,
         "rsi_sar": 3, "momentum": 3, "anti_mom": 2, "rsi_inv": 2}


def build_rule_agents(data, train_range, cost, rng):
    N = data.N
    C = _cands(data, rng)
    raw_all = np.stack([np.repeat(np.arange(N), 2), np.tile([0, 1], N)], axis=1).astype(float)
    lo, hi = train_range
    agents, train_scores = [], {}
    for fam, cands in C.items():
        sc = []
        for name, fn, prm in cands:
            G = np.asarray(fn(raw_all), dtype=np.int8).reshape(N, 2)
            rr = replay_table(G, data.fwd_ret, lo, hi, cost)
            sc.append(rr.mean() / (rr.std() + 1e-12))          # train Sharpe-like score
        sc = np.array(sc)
        for i in pick_spread(sc, PICKS[fam]):
            name, fn, prm = cands[i]
            agents.append(Agent(name, fam, fn, prm))
            train_scores[name] = float(sc[i])
    return agents, train_scores
