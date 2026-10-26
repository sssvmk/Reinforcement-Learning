"""Trading MDP. State at close of day t: (t, pos). Action a in {0 flat, 1 long} = position held over t -> t+1.
Reward r = a*(close[t+1]/close[t]-1) - cost*|a - pos|. Next state (t+1, a). Prices exogenous (price taker)."""
from __future__ import annotations
import numpy as np


def reward(fwd_ret, t, pos, a, cost):
    return a * fwd_ret[t] - cost * np.abs(a - pos)


class TradingEnv:
    def __init__(self, data, cost_bps=5.0, end=None):
        self.data = data
        self.cost = cost_bps / 1e4
        self.end = data.N - 1 if end is None else end   # last usable day index (exclusive of next-day return)
        self.t = 0
        self.pos = 0

    def set_state(self, t, pos):
        self.t, self.pos = int(t), int(pos)

    def get_state(self):
        return (self.t, self.pos)

    def raw_state(self):
        return np.array([self.t, self.pos], float)

    def step(self, a):
        if self.t >= self.end:
            raise RuntimeError("no next-day price inside the allowed range")
        r = reward(self.data.fwd_ret, self.t, self.pos, a, self.cost)
        self.t += 1
        self.pos = int(a)
        return float(r)
