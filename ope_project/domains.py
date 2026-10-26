"""Domain wrappers tying data, features, agents, logs, ground truth and deployment performance together."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from common import Log
from features import TradingFeatures
from trading_env import TradingEnv
from log_data import simulate_trading_log
from agents.trading_rules import build_rule_agents, replay_table
from agents.learned import build_learned_agents
import true_value as tv


class TradingDomain:
    name = "trading"
    n_actions = 2

    def __init__(self, data, fold, cfg):
        self.data, self.fold, self.cfg = data, fold, cfg
        self.gamma = cfg["mdp"]["gamma"]
        self.H = cfg["mdp"]["H"]
        self.cost_bps = cfg["trading"]["cost_bps"]
        self.cost = self.cost_bps / 1e4
        self.reward_scale = cfg["trading"]["reward_scale"]
        self.feats = TradingFeatures(data, fold.train)
        self.dim, self.dim_dense = self.feats.dim, self.feats.dim_dense
        self.pool = []
        self._G = {}
        self._V = {}

    # ---- ranges / features
    def rng_of(self, kind):
        return {"train": self.fold.train, "ope": self.fold.ope, "tune": self.fold.tune, "val": self.fold.val,
                "test": self.fold.test}[kind]

    def phi(self, raw):
        return self.feats.phi(raw)

    def phi_dense(self, raw):
        return self.feats.phi_dense(raw)

    def env(self):
        return TradingEnv(self.data, self.cost_bps)

    # ---- logs
    def sample_log(self, kind, n_windows, eps, rng):
        lo, hi = self.rng_of(kind)
        return simulate_trading_log(self.data, lo, hi, n_windows, eps, self.H, self.cost_bps, rng, kind)

    # ---- agents
    def build_pool(self, seed, use_torch=True, cache_dir=None):
        from agents.base import TableAgent
        key = f"{self.data.ticker}_{self.fold.name}_s{seed}_t{int(use_torch)}"
        cache = None if cache_dir is None else Path(cache_dir) / f"pool_{key}.npz"
        if cache is not None and cache.exists():
            z = np.load(cache, allow_pickle=True)
            self.pool = [TableAgent(n, f, G) for n, f, G in zip(z["names"], z["fams"], z["G"])]
            self.train_scores = z["train_scores"].item()
            return self.pool
        rng = np.random.default_rng(seed)
        agents, self.train_scores = build_rule_agents(self.data, self.fold.train, self.cost, rng)
        if use_torch:
            tr = self.sample_log("train", 300, self.cfg["behavior"]["eps_default"], np.random.default_rng(seed + 1))
            agents += build_learned_agents(self, tr, self.gamma, seed)
        self.pool = [TableAgent(a.name, a.family, self.G(a), a.params) for a in agents]
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, names=[a.name for a in self.pool], fams=[a.family for a in self.pool],
                     G=np.stack([a.G for a in self.pool]), train_scores=np.array(self.train_scores, dtype=object))
        return self.pool

    def G(self, agent):
        if agent.name not in self._G:
            N = self.data.N
            raw = np.stack([np.repeat(np.arange(N), 2), np.tile([0, 1], N)], axis=1).astype(float)
            self._G[agent.name] = np.asarray(agent.act(raw), dtype=np.int64).reshape(N, 2)
        return self._G[agent.name]

    # ---- truth (exact DP)
    def V(self, agent, eps, kind):
        key = (agent.name, eps, kind)
        if key not in self._V:
            lo, hi = self.rng_of(kind)
            self._V[key] = tv.dp_values(self.data.fwd_ret, lo, hi, tv.policy_probs_table(self.G(agent), eps),
                                        self.cost, self.H, self.gamma)
        return self._V[key]

    def truth(self, agents, log, eps_t, kind, state_sample):
        """state_sample: (S,2) int array of (window, step) indices into log.raw. Returns Jw (A,W), vs (A,S)."""
        starts = log.start.astype(int)
        t = log.raw[state_sample[:, 0], state_sample[:, 1], 0].astype(int)
        p = log.raw[state_sample[:, 0], state_sample[:, 1], 1].astype(int)
        Jw = np.stack([self.V(a, eps_t, kind)[starts, 0] for a in agents])
        vs = np.stack([self.V(a, eps_t, kind)[t, p] for a in agents])
        return Jw, vs

    def behavior_value(self, log, eps_b, kind):
        key = ("__behavior__", eps_b, kind)
        if key not in self._V:
            lo, hi = self.rng_of(kind)
            G = self.data.sar_up[:, None].repeat(2, axis=1).astype(np.int64)
            self._V[key] = tv.dp_values(self.data.fwd_ret, lo, hi, tv.policy_probs_table(G, eps_b), self.cost,
                                        self.H, self.gamma)
        return float(self._V[key][log.start.astype(int), 0].mean())

    # ---- deployment performance of the greedy agent over a period, from flat
    def daily_returns(self, agent, kind):
        lo, hi = self.rng_of(kind)
        return replay_table(self.G(agent), self.data.fwd_ret, lo, hi, self.cost)

    def turnover_and_gross(self, agent, kind):
        lo, hi = self.rng_of(kind)
        G = self.G(agent)
        pos, turn = 0, 0
        for t in range(lo, hi - 1):
            a = G[t, pos]
            turn += abs(a - pos)
            pos = a
        return turn / (hi - 1 - lo)
