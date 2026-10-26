"""Open-loop UCT decision-time planner using simulator dynamics + a demand model fitted on the TRAIN period.
The planner never sees true future demand. Decision nodes alternate with sampled-demand chance steps; values are
backed up with a single-agent max (UCB selection, visit-count action at the root)."""
import math
import random
from bisect import bisect_left

import numpy as np

from candidates.base import Policy
from env import EnvParams, unpack


def _step(on, pipe, a, d, price, p: EnvParams):
    """Pure-python copy of env.transition (equality is unit-tested)."""
    if p.lead_time == 0:
        avail, arr, newp = on + a, 0.0, pipe
    else:
        ext = list(pipe)
        ext[p.lead_time - 1] += a
        avail, arr, newp = on, ext[0], ext[1:] + [0.0]
    sales = d if d < avail else avail
    lost = d - sales
    on_end = avail - sales
    r = (price * sales - p.unit_cost * a - p.holding_cost * on_end - p.stockout_penalty_frac * price * lost
         - (p.fixed_order_cost if a > 0 else 0.0))
    return on_end + arr, newp, r


class _Node:
    __slots__ = ("N", "untried", "n", "w", "child")

    def __init__(self, nA):
        self.N = 0
        self.untried = list(range(nA))
        self.n = [0] * nA
        self.w = [0.0] * nA
        self.child = [None] * nA


class MCTSPolicy(Policy):
    family = "MCTS"

    def __init__(self, name, p: EnvParams, dm, budget=200, c=1.0, h_plan=14, S_rollout=None, max_decisions=600,
                 seed=0):
        super().__init__(name, p.n_actions, {"budget": budget, "c": c, "h_plan": h_plan, "demand": dm.kind,
                                             "max_decisions": max_decisions, "S_rollout": S_rollout})
        self.p, self.dm, self.budget, self.c, self.H = p, dm, budget, c, h_plan
        self.S_roll, self.max_dec, self.seed = S_rollout, max_decisions, seed
        self.acts = [float(a) for a in p.actions]
        self.tables = [dm.py_table(w) for w in range(7)]
        self.norm = p.r_scale * min(h_plan, 1.0 / (1 - p.gamma))
        self.cache, self.n_planned, self.n_fallback = {}, 0, 0

    def _nearest(self, target):
        a = self.acts
        i = bisect_left(a, target)
        if i == 0:
            return 0
        if i == len(a):
            return len(a) - 1
        return i if a[i] - target < target - a[i - 1] else i - 1

    def _base_stock_idx(self, on, pipe):
        return self._nearest(max(0.0, self.S_roll - on - sum(pipe)))

    def plan(self, on, pipe, wday, price, rng):
        p, g, nA = self.p, self.p.gamma, self.p.n_actions
        root = _Node(nA)
        H = self.H
        for _ in range(self.budget):
            o, pp, w = on, list(pipe), wday
            node, path, rewards = root, [], []
            expanded = False
            depth = 0
            while depth < H:
                if node.untried:
                    a = node.untried.pop(rng.randrange(len(node.untried)))
                    expanded = True
                else:
                    lnN, best, a = math.log(node.N + 1), -1e18, 0
                    for k in range(nA):
                        u = node.w[k] / node.n[k] + self.c * math.sqrt(lnN / node.n[k])
                        if u > best:
                            best, a = u, k
                tab = self.tables[w]
                d = tab[rng.randrange(len(tab))]
                o, pp, r = _step(o, pp, self.acts[a], d, price, p)
                rewards.append(r)
                path.append((node, a))
                w = (w + 1) % 7
                depth += 1
                if expanded:
                    nxt = _Node(nA)
                    node.child[a] = nxt
                    break
                node = node.child[a] if node.child[a] is not None else _Node(nA)
            while depth < H:      # rollout with the base-stock rule
                a = self._base_stock_idx(o, pp) if self.S_roll is not None else 0
                tab = self.tables[w]
                d = tab[rng.randrange(len(tab))]
                o, pp, r = _step(o, pp, self.acts[a], d, price, p)
                rewards.append(r)
                w = (w + 1) % 7
                depth += 1
            G = 0.0
            for i in range(len(rewards) - 1, -1, -1):
                G = rewards[i] + g * G
                if i < len(path):
                    nd, a = path[i]
                    nd.N += 1
                    nd.n[a] += 1
                    nd.w[a] += G / self.norm
        return max(range(nA), key=lambda k: root.n[k])

    def _decide(self, s):
        key = tuple(np.round(s, 3))
        if key in self.cache:
            return self.cache[key]
        p = self.p
        raw = s[:1 + p.lead_time] * p.mean_demand
        pos, _, wd, _, prel = unpack(s, p)
        if self.n_planned >= self.max_dec:
            self.n_fallback += 1
            a = self._base_stock_idx(float(raw[0]), [float(x) for x in raw[1:]]) if self.S_roll is not None else 0
        else:
            rng = random.Random(self.seed * 1000003 + hash(key))
            a = self.plan(float(raw[0]), [float(x) for x in raw[1:]], int(wd), float(prel * p.mean_price), rng)
            self.n_planned += 1
        self.cache[key] = a
        return a

    def greedy(self, S):
        return np.array([self._decide(s) for s in np.asarray(S)], dtype=int)

    def precompute(self, S):
        """Cache actions for a fixed set of states (used for the FQE subsample)."""
        return self.greedy(S)

    def __getstate__(self):
        d = dict(self.__dict__)
        return d
