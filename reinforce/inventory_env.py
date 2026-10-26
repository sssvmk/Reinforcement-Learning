"""Inventory simulator: lost-sales, lead time L, continuous (or integer-rounded) orders, replayed exogenous demand.

Daily timeline: (1) orders placed L days ago arrive; (2) today's order enters the pipeline (L=0: arrives now);
(3) demand is revealed; (4) sales = min(on_hand, demand), lost = demand - sales; (5) update stocks, shift pipeline.
reward = price*sales - unit_cost*order - holding*on_hand_end - stockout*lost - fixed*1[order>0]
"""
from dataclasses import dataclass, field, replace
import numpy as np


@dataclass
class InvCfg:
    L: int = 2
    unit_cost: float = 0.6
    holding: float = 0.003      # per unit per day
    stockout: float = 0.2       # per lost unit
    fixed: float = 0.0
    init_inv: float = 0.0


@dataclass
class InvData:
    demand: np.ndarray
    price: np.ndarray
    wday: np.ndarray            # 0..6
    event: np.ndarray
    snap: np.ndarray
    price_rel: np.ndarray
    mu: float
    cfg: InvCfg
    qmax: float = None          # absolute cap on one order (None = no cap)
    C: np.ndarray = field(default=None)   # context features per day: [6 weekday dummies, event, snap, price_rel]

    def __post_init__(self):
        if self.C is None:
            dummies = (self.wday[:, None] == np.arange(1, 7)[None, :]).astype(float)
            self.C = np.column_stack([dummies, self.event, self.snap, self.price_rel])

    def with_qmax(self, qmax):
        return replace(self, qmax=qmax)


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


_KEYS = ("sales", "lost", "demand", "order", "on_hand", "revenue", "purchase", "holding", "stockout", "fixed", "ip")


def simulate(policy, theta, env, starts, H, rng=None, mode="mean", record=False, integer=False, detail=False):
    """Vectorized over windows. `policy.act(theta, state, rng, mode)` returns (order, aux).

    record=True additionally stores per-step score-function terms for REINFORCE: rew (N,H), phi (N,H,F), score (N,H,P),
    ent (N,P) = summed entropy gradient. detail=True stores (N,H) arrays for business metrics.
    """
    cfg = env.cfg
    starts = np.asarray(starts)
    N = len(starts)
    on_hand = np.full(N, float(cfg.init_inv))
    pipe = np.zeros((N, cfg.L))
    total = np.zeros(N)
    D = {k: np.zeros((N, H)) for k in _KEYS} if detail else None
    rec = None
    for j in range(H):
        t = starts + j
        on_hand, pipe = receive(on_hand, pipe)
        ip = on_hand + pipe.sum(axis=1)
        st = {"ip": ip, "oh": on_hand, "t": t, "j": j, "env": env}
        order, aux = policy.act(theta, st, rng, mode)
        order = np.maximum(0.0, order)
        if env.qmax is not None:
            order = np.minimum(order, env.qmax)
        if integer:
            order = np.round(order)
        if record:
            sc = policy.score(theta, aux)
            if rec is None:
                P = sc.shape[1]
                phi0 = policy.phi_of(aux)
                rec = {"rew": np.zeros((N, H)), "phi": np.zeros((N, H, phi0.shape[1])), "score": np.zeros((N, H, P)), "ent": np.zeros((N, P))}
            rec["score"][:, j] = sc
            rec["phi"][:, j] = policy.phi_of(aux)
            rec["ent"] += policy.ent_grad(theta, aux)
        on_hand_new, pipe, sales, lost = place_and_sell(on_hand, pipe, order, env.demand[t])
        on_hand = on_hand_new
        rev, pur = env.price[t] * sales, cfg.unit_cost * order
        hold, so, fx = cfg.holding * on_hand, cfg.stockout * lost, cfg.fixed * (order > 0)
        r = rev - pur - hold - so - fx
        total += r
        if record:
            rec["rew"][:, j] = r
        if detail:
            for k, v in zip(_KEYS, (sales, lost, env.demand[t], order, on_hand, rev, pur, hold, so, fx, ip)):
                D[k][:, j] = v
    out = {"total": total}
    if record:
        out.update(rec)
    if detail:
        out.update(D)
    return out
