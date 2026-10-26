"""Rule-based inventory agents: base-stock (8), (s,S) (6), fixed quantity (3), never-order, order-max."""
from __future__ import annotations
import numpy as np
from .base import Agent

S_DAYS = [0.5, 1, 2, 3, 4, 6, 9, 14]                      # base-stock levels in days of mean demand
SS_DAYS = [(0.5, 2), (1, 3), (2, 5), (3, 8), (5, 7), (8, 14)]
FIXED_MULT = [0.5, 1.0, 1.6]


def _nearest_action(Q, q):
    return np.abs(Q[None, :] - np.asarray(q)[:, None]).argmin(axis=1)


def _ip(raw):
    return raw[:, 1] + (raw[:, 2:].sum(axis=1) if raw.shape[1] > 2 else 0.0)


def base_stock_fn(dom, S):
    return lambda raw: _nearest_action(dom.Q, np.clip(S - _ip(raw), 0, None))


def ss_fn(dom, s, S):
    return lambda raw: _nearest_action(dom.Q, np.where(_ip(raw) <= s, np.clip(S - _ip(raw), 0, None), 0.0))


def build_inventory_rule_agents(dom):
    u, out = dom.u, []
    for d in S_DAYS:
        out.append(Agent(f"base_stock_{d}d", "base_stock", base_stock_fn(dom, d * u), {"S_days": d}))
    for s, S in SS_DAYS:
        out.append(Agent(f"sS_{s}_{S}d", "sS", ss_fn(dom, s * u, S * u), {"s_days": s, "S_days": S}))
    for m in FIXED_MULT:
        a = int(_nearest_action(dom.Q, np.array([m * u]))[0])
        out.append(Agent(f"fixed_{m}", "fixed", (lambda a: lambda raw: np.full(len(raw), a))(a), {"mult": m}))
    out.append(Agent("never_order", "trivial", lambda raw: np.zeros(len(raw), dtype=np.int64)))
    out.append(Agent("order_max", "trivial", lambda raw: np.full(len(raw), len(dom.Q) - 1)))
    return out


def tune_behavior_base_stock(dom):
    """Behavior rule: order up to the S (in days) with the best deterministic profit on the TRAIN period."""
    lo, hi = dom.fold.train
    best, best_p = S_DAYS[0], -np.inf
    for d in S_DAYS:
        fn = base_stock_fn(dom, d * dom.u)
        t, on, pipe, p = np.array([lo]), np.array([dom.init_on]), np.zeros((1, dom.npipe)), 0.0
        for _ in range(hi - lo):
            a = fn(dom.raw(t, on, pipe))
            r, t, on, pipe, _, _ = dom.vec_step(t, on, pipe, dom.Q[a])
            p += float(r[0])
        if p > best_p:
            best, best_p = d, p
    dom.behavior_S_days = best
    return Agent(f"behavior_S{best}d", "behavior", base_stock_fn(dom, best * dom.u), {"S_days": best})
