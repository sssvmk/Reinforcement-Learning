"""Non-learned reference policies (incl. clearly bad anchors) and the base-stock rule."""
import numpy as np

from candidates.base import Policy
from env import EnvParams, unpack


def nearest_action(target, actions):
    return np.abs(np.asarray(target)[..., None] - actions).argmin(-1)


class BaseStock(Policy):
    """Order up to S (units): target = max(0, S - position), rounded to the nearest grid action."""

    def __init__(self, p: EnvParams, S, name=None):
        super().__init__(name or f"basestock_S{S:g}", p.n_actions, {"S": float(S)})
        self.p, self.S = p, float(S)

    def greedy(self, Sx):
        pos, *_ = unpack(Sx, self.p)
        return nearest_action(np.maximum(0.0, self.S - pos), self.p.actions)


class sS(Policy):
    """(s, S): if position <= s order up to S, else nothing."""

    def __init__(self, p: EnvParams, s, S, name=None):
        super().__init__(name or f"sS_{s:g}_{S:g}", p.n_actions, {"s": float(s), "S": float(S)})
        self.p, self.s, self.S = p, float(s), float(S)

    def greedy(self, Sx):
        pos, *_ = unpack(Sx, self.p)
        a = nearest_action(np.maximum(0.0, self.S - pos), self.p.actions)
        return np.where(pos <= self.s, a, 0)


class FixedQty(Policy):
    def __init__(self, p: EnvParams, idx, name=None):
        super().__init__(name or f"fixed_q{p.actions[idx]:g}", p.n_actions, {"action_idx": int(idx)})
        self.idx = int(idx)

    def greedy(self, Sx):
        return np.full(len(Sx), self.idx)


def never_order(p):
    q = FixedQty(p, 0, "never_order"); return q


def order_max(p):
    return FixedQty(p, p.n_actions - 1, "order_max")


def reference_pool(p: EnvParams, S_star):
    m = p.mean_demand
    pool = [never_order(p), order_max(p)]
    for k in [1, 2, 4]:
        pool.append(FixedQty(p, int(nearest_action(k * m, p.actions))))
    for f in [0.5, 1.0, 1.5]:
        pool.append(BaseStock(p, f * S_star))
    pool.append(BaseStock(p, S_star))
    pool.append(BaseStock(p, 0.25 * S_star))
    pool.append(sS(p, 0.5 * S_star, S_star))
    pool.append(sS(p, 0.8 * S_star, 1.3 * S_star))
    out, seen = [], set()
    for q in pool:
        if q.name not in seen:
            seen.add(q.name); out.append(q)
    return out
