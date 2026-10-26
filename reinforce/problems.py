"""Problem construction: splits, windows, reward normalisation, benchmarks tuned on train (and, labelled, on validation)."""
from dataclasses import dataclass
import numpy as np
import inventory_env as ie
import trading_env as te
import policies as P
import data_trading as dt


@dataclass
class Problem:
    domain: str
    key: str
    env: object
    splits: dict
    H: int
    valid: np.ndarray        # valid[t] is True if a window may start at t
    norm: float = 1.0
    meta: dict = None

    def allowed(self, split):
        lo, hi = self.splits[split]
        s = np.arange(lo, hi - self.H)
        return s[self.valid[s]]

    def sample(self, split, n, rng):
        a = self.allowed(split)
        return a[rng.integers(0, len(a), n)]

    def grid(self, split, stride):
        return self.allowed(split)[::stride]

    def run(self, pol, theta, starts, rng=None, mode="mean", **kw):
        sim = ie.simulate if self.domain == "inventory" else te.simulate
        return sim(pol, theta, self.env, starts, self.H, rng, mode, **kw)

    def J(self, pol, theta, starts, rng=None, mode="mean"):
        """Mean undiscounted window return in normalised units."""
        return float(self.run(pol, theta, starts, rng, mode)["total"].mean() / self.norm)


def _valid_windows(ok, H, n):
    c = np.r_[0, np.cumsum(~ok)]
    v = np.zeros(n, bool)
    s = np.arange(0, n - H)
    v[s] = (c[s + H] - c[s]) == 0
    return v


def build_inventory(df, key, cfg, splits_frac):
    n = len(df)
    sp = dt.make_splits(n, *splits_frac)
    icfg = cfg["inventory"]
    H, L = icfg["horizon"], icfg["lead_time"]
    tr = slice(*sp["train"])
    d = df["sales"].to_numpy(float)
    price = df["price"].to_numpy(float)
    ok = df["price_ok"].to_numpy(bool)
    mu = float(d[tr].mean())
    mp = float(np.nanmean(price[tr][ok[tr]]))
    price = np.where(ok, price, mp)
    unit_cost = 0.6 * mp
    cost = ie.InvCfg(L=L, unit_cost=unit_cost, holding=0.005 * unit_cost, stockout=0.2 * mp, fixed=0.5 * unit_cost, init_inv=mu * (L + 1))
    env = ie.InvData(d, price, df["wday0"].to_numpy(int), df["event"].to_numpy(float), df["snap"].to_numpy(float), price / mp - 1.0,
                     mu, cost, qmax=None if icfg["q_max_mult"] is None else icfg["q_max_mult"] * mu)
    pr = Problem("inventory", key, env, sp, H, _valid_windows(ok, H, n), meta={"mu": mu, "mean_price": mp})
    cover = P.BaseStock()
    tot = pr.run(cover, np.array([L + 1.0]), pr.grid("train", 5))["total"]
    pr.norm = float(max(tot.std(), 1e-6))
    pr.meta["norm_rule"] = "std of train-window profit of the (L+1)mu cover policy"
    return pr


def build_trading(data, key, cfg, splits_frac, splits=None):
    n = len(data.close)
    sp = splits or dt.make_splits(n, *splits_frac)
    tcfg = cfg["trading"]
    H = tcfg["horizon"]
    env = te.make_trade_data(data, sp["train"], tcfg["cost_bps"] * 1e-4)
    pr = Problem("trading", key, env, sp, H, _valid_windows(np.ones(n, bool), H, n), meta={})
    lo, hi = sp["train"]
    s = pr.grid("train", 5)
    bh = np.array([data.R[a:a + H].sum() for a in s])
    pr.norm = float(max(bh.std(), 1e-6))
    pr.meta["norm_rule"] = "std of train-window buy-and-hold returns"
    return pr


# ------------------------------------------------------------------ benchmarks
def tune_grid(pr, pol, grid, split, stride):
    starts = pr.grid(split, stride)
    vals = [pr.J(pol, np.atleast_1d(np.asarray(g, float)), starts) for g in grid]
    i = int(np.argmax(vals))
    return np.atleast_1d(np.asarray(grid[i], float)), float(vals[i])


def inventory_benchmarks(pr, tune_on="train"):
    """Base-stock and (s,S), tuned on `tune_on` ('train' = standard rule, 'val' = labelled oracle-ish variant), plus the cover rule."""
    L = pr.env.cfg.L
    out = {"cover": (P.BaseStock(), np.array([L + 1.0]))}
    th, _ = tune_grid(pr, P.BaseStock(), [[x] for x in np.linspace(0, 10, 101)], tune_on, 5)
    out["base_stock"] = (P.BaseStock(), th)
    grid = [[s, np.log(np.expm1(max(g, 1e-3)))] for s in np.linspace(0, 8, 17) for g in np.linspace(0.25, 6, 13)]
    th, _ = tune_grid(pr, P.SSPolicy(), grid, tune_on, 5)
    out["sS"] = (P.SSPolicy(), th)
    return out


def trading_benchmarks(pr, tune_on="train"):
    out = {"buy_and_hold": (P.FixedPos(1.0, "buy_and_hold"), np.zeros(0)), "flat": (P.FixedPos(0.0, "flat"), np.zeros(0))}
    grid = [[a, b] for a in range(15, 46, 5) for b in range(55, 86, 5)]
    th, _ = tune_grid(pr, P.RsiRule(), grid, tune_on, 10)
    out["RSI_rule"] = (P.RsiRule(), th)
    out["SAR_rule"] = (P.SarRule(), np.zeros(0))
    return out


# ------------------------------------------------------------------ metrics
ANN = 252


def max_drawdown(eq):
    peak = np.maximum.accumulate(eq)
    return float(((eq - peak) / peak).min())


def trading_metrics(net, pos, gross):
    """net: per-day net returns of one contiguous replay."""
    eq = np.cumprod(1 + net)
    n = len(net)
    sd = net.std(ddof=1) if n > 1 else np.nan
    dn = net[net < 0]
    return {"cum_return": float(eq[-1] - 1), "ann_return": float(eq[-1] ** (ANN / n) - 1), "sharpe": float(net.mean() / sd * np.sqrt(ANN)) if sd > 0 else np.nan,
            "sortino": float(net.mean() / dn.std(ddof=1) * np.sqrt(ANN)) if len(dn) > 1 and dn.std() > 0 else np.nan, "max_drawdown": max_drawdown(eq),
            "turnover_per_year": float(np.abs(np.diff(np.r_[0, pos])).sum() / n * ANN), "exposure": float(pos.mean()),
            "hit_rate": float((net[pos > 0] > 0).mean()) if (pos > 0).any() else np.nan, "cost_drag": float((gross - net).sum())}


def inventory_metrics(D, price_mean):
    dem = D["demand"].sum(axis=1)
    sales = D["sales"].sum(axis=1)
    prof = D["revenue"].sum(1) - D["purchase"].sum(1) - D["holding"].sum(1) - D["stockout"].sum(1) - D["fixed"].sum(1)
    return {"profit": prof, "fill_rate": sales / np.maximum(dem, 1e-9), "stockout_days": (D["lost"] > 0).mean(axis=1),
            "avg_inventory": D["on_hand"].mean(axis=1), "orders_per_window": (D["order"] > 0).sum(axis=1),
            "order_cv": D["order"].std(axis=1) / np.maximum(D["order"].mean(axis=1), 1e-9), "bullwhip": D["order"].var(axis=1) / np.maximum(D["demand"].var(axis=1), 1e-9),
            "holding": D["holding"].sum(1), "stockout_cost": D["stockout"].sum(1), "fixed_cost": D["fixed"].sum(1),
            "purchase": D["purchase"].sum(1), "revenue": D["revenue"].sum(1)}
