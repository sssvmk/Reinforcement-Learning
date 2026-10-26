"""Agent base classes and the target-policy construction pi = (1-eps)*greedy + eps*uniform."""
from __future__ import annotations
import numpy as np


class Agent:
    def __init__(self, name, family, fn, params=None):
        self.name, self.family, self.fn, self.params = name, family, fn, params or {}

    def act(self, raw):
        return np.asarray(self.fn(np.asarray(raw, float)), dtype=np.int64)

    def pi_prob(self, raw, actions, eps, n_actions):
        """pi(a|s) for the given taken actions."""
        g = self.act(raw)
        return (1.0 - eps) * (g == actions) + eps / n_actions

    def __repr__(self):
        return f"Agent({self.name})"


class TableAgent(Agent):
    """Frozen greedy policy given by a lookup table G[day, position] (trading state space is (t, pos))."""

    def __init__(self, name, family, G, params=None):
        self.G = np.asarray(G, dtype=np.int64)
        super().__init__(name, family, lambda raw: self.G[raw[:, 0].astype(int), raw[:, 1].astype(int)], params)


def pick_spread(scores, n):
    """Indices of n candidates evenly spread over the score ranking (best and worst included)."""
    order = np.argsort(scores)
    if n >= len(order):
        return order
    pos = np.linspace(0, len(order) - 1, n).round().astype(int)
    return order[np.unique(pos)]
