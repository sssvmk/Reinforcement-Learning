"""Epsilon-soft base-stock logging policy and logged datasets (train-period episodes only)."""
import json
from pathlib import Path

import numpy as np

from candidates.references import BaseStock
from data import Series
from discretize import Discretizer, coverage_report
from env import DemandModel, EnvParams, make_episodes, rollout


class BehaviorPolicy:
    """Mixture over order-up-to levels of epsilon-soft base-stock policies (single S = ordinary case)."""

    def __init__(self, p: EnvParams, S_levels, eps):
        self.p, self.eps = p, eps
        self.S_levels = list(S_levels)
        self.pols = [BaseStock(p, S) for S in self.S_levels]

    def probs_k(self, Sx, k):
        a = self.pols[k].greedy(Sx)
        P = np.full((len(a), self.p.n_actions), self.eps / self.p.n_actions)
        P[np.arange(len(a)), a] += 1 - self.eps
        return P

    def probs(self, Sx):
        """Marginal (over the S mixture) action probabilities: what IS estimators must use."""
        return np.mean([self.probs_k(Sx, k) for k in range(len(self.pols))], axis=0)


def tune_base_stock(series, cfg, p, dm, seed=123, n_ep=60):
    """Choose S on the train period (generative demand), by mean discounted return."""
    ep = make_episodes(series, cfg, dm, "train", np.arange(n_ep) + 900_000, seed)
    best, bestv = None, -np.inf
    for k in np.arange(0.5, 10.01, 0.5):
        S = k * p.mean_demand
        v = rollout(ep, p, lambda s, t, pol=BaseStock(p, S): pol.greedy(s))["ret"].mean()
        if v > bestv:
            best, bestv = S, v
    return float(best), float(bestv)


def generate_log(series, cfg, p, dm, disc: Discretizer, behavior: BehaviorPolicy, split, ids, seed, mode="gen"):
    ep = make_episodes(series, cfg, dm, split, ids, seed, mode)
    N, T = ep.demand.shape
    rngs = [np.random.default_rng([seed, int(i), 7]) for i in ids]
    ks = np.array([r.integers(len(behavior.pols)) for r in rngs])
    U = np.stack([r.random(T) for r in rngs])

    def act(s, t):
        P = np.zeros((N, p.n_actions))
        for k in range(len(behavior.pols)):
            m = ks == k
            if m.any():
                P[m] = behavior.probs_k(s[m], k)
        return (np.cumsum(P, 1) < U[:, t:t + 1]).sum(1).clip(0, p.n_actions - 1)

    out = rollout(ep, p, act, record=True)
    s, s2 = out["s"].reshape(-1, p.state_dim), out["s2"].reshape(-1, p.state_dim)
    a = out["a"].reshape(-1)
    bp = np.empty(len(a))
    for i in range(0, len(a), 20000):
        bp[i:i + 20000] = behavior.probs(s[i:i + 20000])[np.arange(len(a[i:i + 20000])), a[i:i + 20000]]
    trunc = (out["t"].reshape(-1) == T - 1)
    log = dict(s_cont=s, s_discrete=disc(s), a=a, r=out["r"].reshape(-1), s2_cont=s2, s2_discrete=disc(s2),
               done=np.zeros(len(a), bool), truncated=trunc, behavior_prob=bp,
               episode_id=np.repeat(np.asarray(ids), T), t=out["t"].reshape(-1), s0=out["s0"], ep_ids=np.asarray(ids))
    return log


def split_log_ids(n_total, fqe_frac, seed, id_offset=0):
    rng = np.random.default_rng(seed)
    ids = rng.permutation(n_total) + id_offset
    k = int(round(n_total * fqe_frac))
    return np.sort(ids[k:]), np.sort(ids[:k])        # train-log ids, fqe-log ids (disjoint by construction)


def subset_log(log, ids):
    m = np.isin(log["episode_id"], ids)
    out = {k: v[m] for k, v in log.items() if k not in ("s0", "ep_ids")}
    em = np.isin(log["ep_ids"], ids)
    out["s0"], out["ep_ids"] = log["s0"][em], log["ep_ids"][em]
    return out


def save_log(log, path, meta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path.with_suffix(".npz"), **log)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1, default=float), encoding="utf-8")


def log_summary(log, disc, n_actions):
    return coverage_report(disc, log["s_discrete"], log["a"], n_actions)
