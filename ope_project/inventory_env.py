"""Inventory MDP (single item). Timeline per day t: state = (t, on_hand after today's arrivals, pipeline of orders still in
transit [arrive at t+1 .. t+L-1]).  Agent orders q; if L = 0 it arrives at once; else it arrives at the start of day t+L.
Demand is then revealed, sales = min(on_hand, demand).  Reward = price*sales - unit_cost*q - holding*on_hand_end
- penalty*lost - fixed*1[q>0].  Demand comes from an array (real M5 series once M5_DIR is valid, synthetic otherwise)."""
from __future__ import annotations
import numpy as np


def synthetic_demand(n, rng, mean=5.0, weekly=0.3):
    """Poisson demand with a weekly pattern; used for tests while M5 is PENDING."""
    t = np.arange(n)
    lam = mean * (1 + weekly * np.sin(2 * np.pi * t / 7))
    return rng.poisson(lam).astype(float)


class InventoryEnv:
    def __init__(self, demand, price, unit_cost, holding, penalty, fixed, lead_time, q_max):
        self.d = np.asarray(demand, float)
        self.p, self.c, self.h, self.pen, self.fix = price, unit_cost, holding, penalty, fixed
        self.L, self.qmax = int(lead_time), int(q_max)
        self.set_state(0, 0.0, np.zeros(max(self.L - 1, 0)))

    def set_state(self, t, on_hand, pipe):
        self.t, self.on_hand = int(t), float(on_hand)
        self.pipe = np.array(pipe, float).copy()
        assert len(self.pipe) == max(self.L - 1, 0)

    def raw_state(self):
        return np.concatenate([[self.t, self.on_hand], self.pipe])

    def step(self, q):
        q = float(np.clip(q, 0, self.qmax))
        on = self.on_hand + (q if self.L == 0 else 0.0)
        dem = self.d[self.t]
        sales = min(on, dem)
        lost = dem - sales
        end = on - sales
        r = self.p * sales - self.c * q - self.h * end - self.pen * lost - self.fix * (q > 0)
        if self.L == 0:
            arrive, newpipe = 0.0, self.pipe
        elif self.L == 1:
            arrive, newpipe = q, self.pipe
        else:
            arrive, newpipe = self.pipe[0], np.append(self.pipe[1:], q)
        self.on_hand, self.pipe, self.t = end + arrive, newpipe, self.t + 1
        return float(r), dict(sales=sales, lost=lost, end=end, arrived=arrive)

    def total_stock(self):
        return self.on_hand + self.pipe.sum()
