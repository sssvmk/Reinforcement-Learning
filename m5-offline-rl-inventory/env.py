"""Single item-store lost-sales inventory MDP: pure transition function, features, demand model, episodes."""
from collections import namedtuple
from dataclasses import dataclass

import numpy as np

from data import Series, split_bounds

HIST = 7
Day = namedtuple("Day", ["price"])


@dataclass
class EnvParams:
    lead_time: int
    gamma: float
    horizon: int
    mean_demand: float      # train-period statistics only
    mean_price: float
    unit_cost: float
    holding_cost: float
    stockout_penalty_frac: float
    fixed_order_cost: float
    actions: np.ndarray     # order-quantity grid
    r_scale: float

    @property
    def n_actions(self):
        return len(self.actions)

    @property
    def state_dim(self):
        return self.lead_time + 7


def make_params(series: Series, cfg, **over):
    """All statistics come from the train period only."""
    e = {**cfg["env"], **over}
    lo, hi = split_bounds(len(series), cfg)["train"]
    m = max(float(series.sales[lo:hi].mean()), 0.1)
    v = series.valid[lo:hi]
    mp = float(series.price[lo:hi][v].mean())
    acts = []
    for k in e["action_multiples"]:
        acts.append(0 if k == 0 else max(1, int(np.ceil(k * m))))
    acts = np.unique(acts).astype(float)
    uc = e["unit_cost_frac"] * mp
    return EnvParams(lead_time=e["lead_time"], gamma=e["gamma"], horizon=e["horizon"], mean_demand=m,
                     mean_price=mp, unit_cost=uc, holding_cost=e["holding_frac"] * uc,
                     stockout_penalty_frac=e["stockout_frac"], fixed_order_cost=e["fixed_order_frac"] * uc,
                     actions=acts, r_scale=mp * max(m, 1.0))


def init_inventory(p: EnvParams):
    """Fixed start: 2 days of demand on hand, 1 day in each pipeline slot."""
    inv = np.zeros(1 + p.lead_time)
    inv[0] = np.ceil(2 * p.mean_demand)
    inv[1:] = np.ceil(p.mean_demand)
    return inv


def transition(state, action, demand, day, p: EnvParams):
    """Pure inventory transition. state[..., 0] = on-hand after today's arrivals,
    state[..., 1:] = pipeline (slot i arrives at the start of day t+1+i).
    Timeline: (1) arrivals already in on-hand; (2) order placed (L=0: arrives at once);
    (3) demand revealed; (4) sales=min(on_hand, demand); (5) update, shift pipeline, next day's arrivals."""
    state = np.asarray(state, dtype=float)
    action = np.asarray(action, dtype=float)
    demand = np.asarray(demand, dtype=float)
    price = np.asarray(getattr(day, "price", day), dtype=float)
    on, pipe = state[..., 0], state[..., 1:]
    if p.lead_time == 0:
        avail, arr, new_pipe = on + action, 0.0, pipe
    else:
        # order placed today arrives at the start of day t+L: slot L-1 of the array that arrives from t+1 on
        slot = np.zeros(pipe.shape[:-1] + (p.lead_time,))
        slot[..., p.lead_time - 1] = action
        ext = pipe + slot
        avail, arr = on, ext[..., 0]
        new_pipe = np.concatenate([ext[..., 1:], np.zeros_like(ext[..., :1])], axis=-1)
    sales = np.minimum(avail, demand)
    lost = demand - sales
    on_end = avail - sales
    reward = (price * sales - p.unit_cost * action - p.holding_cost * on_end
              - p.stockout_penalty_frac * price * lost - p.fixed_order_cost * (action > 0))
    nxt = np.concatenate([(on_end + arr)[..., None], new_pipe], axis=-1)
    return nxt, reward, {"sales": sales, "lost": lost, "on_hand_end": on_end, "order_qty": action}


def features(inv, wday, event, snap, price, trailing, p: EnvParams):
    m = p.mean_demand
    ang = 2 * np.pi * np.asarray(wday) / 7.0
    cols = [inv / m, np.sin(ang)[..., None], np.cos(ang)[..., None], np.asarray(event, float)[..., None],
            np.asarray(snap, float)[..., None], (np.asarray(price) / p.mean_price)[..., None],
            (np.asarray(trailing) / m)[..., None]]
    return np.concatenate(cols, axis=-1)


def unpack(s, p: EnvParams):
    """From s_cont: position (units), on-hand (units), weekday idx, event flag, price (rel)."""
    s = np.asarray(s)
    L = p.lead_time
    pos = s[..., :1 + L].sum(-1) * p.mean_demand
    on = s[..., 0] * p.mean_demand
    wd = np.round(np.arctan2(s[..., 1 + L], s[..., 2 + L]) * 7 / (2 * np.pi)).astype(int) % 7
    return pos, on, wd, s[..., 3 + L], s[..., 5 + L]


class DemandModel:
    """Fit on the train period only. kind: bootstrap | negbin | mean."""

    def __init__(self, sales, wday, snap, kind="bootstrap", by_snap=False, seed=0):
        self.kind, self.by_snap = kind, by_snap
        self.vals = {}
        for w in range(7):
            for sn in ([0, 1] if by_snap else [None]):
                msk = (wday == w) if sn is None else (wday == w) & (snap == sn)
                v = sales[msk].astype(float)
                if len(v) < 20:
                    v = sales[wday == w].astype(float)
                self.vals[(w, sn)] = v
        self.mean_ = {k: v.mean() for k, v in self.vals.items()}
        rng = np.random.default_rng(seed)
        self.table = {}
        for k, v in self.vals.items():
            if kind == "mean":
                self.table[k] = [self.mean_[k]]
            elif kind == "negbin":
                mu, var = v.mean(), v.var()
                if var > mu + 1e-9:
                    r = mu * mu / (var - mu)
                    x = rng.negative_binomial(r, r / (r + mu), size=2000)
                else:
                    x = rng.poisson(mu, size=2000)
                self.table[k] = [float(t) for t in x]
            else:
                self.table[k] = [float(t) for t in v]

    def _key(self, w, sn):
        return (int(w), int(sn) if self.by_snap else None)

    def sample(self, rng, wday, snap):
        wday, snap = np.asarray(wday), np.asarray(snap)
        out = np.empty(wday.shape, float)
        for w in range(7):
            for sn in ([0, 1] if self.by_snap else [None]):
                msk = (wday == w) if sn is None else (wday == w) & (snap == sn)
                n = int(msk.sum())
                if n:
                    tab = np.asarray(self.table[(w, sn)])
                    out[msk] = tab[rng.integers(len(tab), size=n)] if self.kind != "mean" else tab[0]
        return out

    def py_table(self, w):
        return self.table[(int(w), None)] if not self.by_snap else self.table[(int(w), 0)]


@dataclass
class Episodes:
    ids: np.ndarray
    start: np.ndarray
    wday: np.ndarray     # (N, T+1)
    event: np.ndarray
    snap: np.ndarray
    price: np.ndarray
    demand: np.ndarray   # (N, T)
    hist: np.ndarray     # (N, HIST) demand before the first decision day


def valid_starts(series: Series, lo, hi, T):
    v = (~series.valid).astype(int)
    cs = np.concatenate([[0], np.cumsum(v)])
    s = np.arange(lo + HIST, hi - T)
    ok = (cs[s + T + 1] - cs[s - HIST]) == 0
    return s[ok]


def make_episodes(series: Series, cfg, dm: DemandModel, split, ids, seed, mode="gen", horizon=None):
    """Whole demand array for an episode is drawn up front from (seed, episode_id): common random numbers."""
    T = horizon or cfg["env"]["horizon"]
    lo, hi = split_bounds(len(series), cfg)[split]
    starts = valid_starts(series, lo, hi, T)
    if len(starts) == 0:
        raise ValueError(f"no valid {T}-day window in split {split}")
    N = len(ids)
    W = np.empty((N, T + 1), int); E = np.empty_like(W); S_ = np.empty_like(W)
    P = np.empty((N, T + 1)); D = np.empty((N, T)); H = np.empty((N, HIST)); st = np.empty(N, int)
    for i, eid in enumerate(ids):
        rng = np.random.default_rng([seed, int(eid)])
        s0 = int(starts[rng.integers(len(starts))])
        st[i] = s0
        sl = slice(s0, s0 + T + 1)
        W[i], E[i], S_[i], P[i] = series.wday[sl], series.event[sl], series.snap[sl], series.price[sl]
        if mode == "replay":
            D[i] = series.sales[s0:s0 + T]
            H[i] = series.sales[s0 - HIST:s0]
        else:
            D[i] = dm.sample(rng, W[i, :T], S_[i, :T])
            H[i] = dm.sample(rng, series.wday[s0 - HIST:s0], series.snap[s0 - HIST:s0])
    return Episodes(np.asarray(ids), st, W, E, S_, P, D, H)


def rollout(ep: Episodes, p: EnvParams, act_fn, record=False):
    """Vectorised over episodes. act_fn(s_cont (N,d), t) -> action indices (N,).
    Returns discounted returns (N,) and optional transition records. Horizon end = truncation."""
    N, T = ep.demand.shape
    inv = np.tile(init_inventory(p), (N, 1))
    hist = ep.hist.copy()
    ret = np.zeros(N)
    rec = {k: [] for k in ["s", "a", "r", "s2", "t"]} if record else None
    s = features(inv, ep.wday[:, 0], ep.event[:, 0], ep.snap[:, 0], ep.price[:, 0], hist.mean(1), p)
    s0 = s.copy()
    for t in range(T):
        a = np.asarray(act_fn(s, t), dtype=int)
        q = p.actions[a]
        inv, r, _ = transition(inv, q, ep.demand[:, t], Day(ep.price[:, t]), p)
        hist = np.concatenate([hist[:, 1:], ep.demand[:, t:t + 1]], axis=1)
        s2 = features(inv, ep.wday[:, t + 1], ep.event[:, t + 1], ep.snap[:, t + 1], ep.price[:, t + 1],
                      hist.mean(1), p)
        ret += p.gamma ** t * r
        if record:
            rec["s"].append(s); rec["a"].append(a); rec["r"].append(r); rec["s2"].append(s2)
            rec["t"].append(np.full(N, t))
        s = s2
    out = {"ret": ret, "s0": s0}
    if record:
        out.update({k: np.stack(v, 1) for k, v in rec.items()})
    return out


class Env:
    """Thin single-episode wrapper around `transition` (replay of an Episodes row)."""

    def __init__(self, p: EnvParams, ep: Episodes, i=0):
        self.p, self.ep, self.i = p, ep, i
        self.reset()

    def reset(self):
        self.inv = init_inventory(self.p)
        self.hist = self.ep.hist[self.i].copy()
        self.t = 0
        return self._obs(0)

    def _obs(self, t):
        e, i = self.ep, self.i
        return features(self.inv, e.wday[i, t], e.event[i, t], e.snap[i, t], e.price[i, t], self.hist.mean(), self.p)

    def step(self, a_idx):
        e, i, t = self.ep, self.i, self.t
        self.inv, r, info = transition(self.inv, self.p.actions[a_idx], e.demand[i, t], Day(e.price[i, t]), self.p)
        self.hist = np.append(self.hist[1:], e.demand[i, t])
        self.t += 1
        trunc = self.t >= e.demand.shape[1]
        return self._obs(self.t), float(r), False, trunc, info
