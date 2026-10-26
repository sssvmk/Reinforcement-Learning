"""Inventory simulator: lost-sales, lead time L, continuous (or integer-rounded) orders, replayed exogenous demand."""
from dataclasses import dataclass
import numpy as np


@dataclass
class InvCfg:
    L: int = 2
    unit_cost: float = 0.6
    holding: float = 0.003      # per unit per day
    stockout: float = 0.2       # per lost unit (absolute, = 0.2*price in practice)
    fixed: float = 0.0
    init_inv: float = 0.0


def receive(on_hand, pipe):
    """(1) Orders placed L days ago arrive. pipe has shape (N, L); slot 0 arrives today."""
    if pipe.shape[1] == 0:
        return on_hand, pipe
    on_hand = on_hand + pipe[:, 0]
    pipe = np.concatenate([pipe[:, 1:], np.zeros((pipe.shape[0], 1))], axis=1)
    return on_hand, pipe


def place_and_sell(on_hand, pipe, order, demand):
    """(2) today's order enters the pipeline (L=0: arrives now), (3-5) demand is revealed and served."""
    if pipe.shape[1] == 0:
        on_hand = on_hand + order
    else:
        pipe = pipe.copy()
        pipe[:, -1] = order
    sales = np.minimum(on_hand, demand)
    return on_hand - sales, pipe, sales, demand - sales


def simulate(policy, theta, demand, price, ctx, starts, H, cfg, integer=False, detail=False):
    """Pure per-day loop vectorized over windows.

    demand, price: arrays over days. ctx: dict of day-indexed arrays used by the policy (wday, event, snap, price_rel).
    reward = price*sales - unit_cost*order - holding*on_hand_end - stockout*lost - fixed*1[order>0]
    """
    starts = np.asarray(starts)
    N = len(starts)
    on_hand = np.full(N, float(cfg.init_inv))
    pipe = np.zeros((N, cfg.L))
    total = np.zeros(N)
    keys = ("sales", "lost", "demand", "order", "on_hand", "revenue", "purchase", "holding", "stockout", "fixed")
    D = {k: np.zeros((N, H)) for k in keys} if detail else None
    for j in range(H):
        t = starts + j
        on_hand, pipe = receive(on_hand, pipe)
        ip = on_hand + pipe.sum(axis=1)
        order = np.maximum(0.0, policy.order(theta, ip, {k: v[t] for k, v in ctx.items()}))
        if integer:
            order = np.round(order)
        on_hand, pipe, sales, lost = place_and_sell(on_hand, pipe, order, demand[t])
        rev, pur = price[t] * sales, cfg.unit_cost * order
        hold, so, fx = cfg.holding * on_hand, cfg.stockout * lost, cfg.fixed * (order > 0)
        total += rev - pur - hold - so - fx
        if detail:
            for k, v in zip(keys, (sales, lost, demand[t], order, on_hand, rev, pur, hold, so, fx)):
                D[k][:, j] = v
    return D if detail else total
