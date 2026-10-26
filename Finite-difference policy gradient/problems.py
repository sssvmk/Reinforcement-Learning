"""Builds optimization problems (environment + policy + splits + objectives) for both domains."""
import numpy as np
from fdpg.crn import Objective, PathSampler
from fdpg.optimizer import Scaler
from fdpg.baselines import grid_tune
import policies as P
import trading_env
import inventory_env
from data_trading import make_splits


class Problem:
    def __init__(self, domain, name, policy, run, H, N, splits, lam=0.0, valid=None, extra=None, vs=15, ds=50):
        self.domain, self.name, self.policy, self.run = domain, name, policy, run
        self.H, self.N, self.splits, self.lam, self.valid = H, N, splits, lam, valid
        self.vs, self.ds = vs, ds   # strides of the fixed validation / train-diagnostic window grids
        self.extra = extra or {}
        self.norm = 1.0
        self.norm = self._norm()

    def _norm(self):
        raise NotImplementedError

    def objective(self, policy=None, split="train", fixed_stride=None, N=None):
        policy = policy or self.policy
        lo, hi = self.splits[split]
        samp = PathSampler(lo, hi, self.H, self.valid)
        fixed = samp.grid(fixed_stride) if fixed_stride else None
        sc = Scaler(policy.scale) if policy.n else Scaler(np.zeros(0))
        return Objective(lambda th, s: self.run(policy, th, s), samp, N or self.N, to_phys=sc.to_phys,
                         norm=self.norm, lam=self.lam, fixed_paths=fixed)

    def objectives(self):
        """Fresh (counted train, fixed validation, fixed train diagnostic) objectives for one run."""
        return (self.objective(split="train"), self.objective(split="val", fixed_stride=self.vs),
                self.objective(split="train", fixed_stride=self.ds))

    def bounds(self, policy=None):
        p = policy or self.policy
        return p.ilo, p.ihi

    def window_returns(self, policy, theta_phys, split, stride=10):
        lo, hi = self.splits[split]
        starts = PathSampler(lo, hi, self.H, self.valid).grid(stride)
        return self.run(policy, theta_phys, starts)


class TradingProblem(Problem):
    def _norm(self):
        starts = PathSampler(*self.splits["train"], self.H).grid(10)
        bh = self.run(P.Fixed(1.0, "bh"), np.zeros(0), starts)
        return float(max(bh.std(), 1e-6))

    def detail(self, policy, theta_phys, lo, hi):
        """One contiguous replay from index lo to hi-1 (starts flat); returns daily net/gross/pos and dates."""
        d = self.extra["data"]
        H = hi - 1 - lo
        out = trading_env.simulate(policy, theta_phys, d.X, d.R, np.array([lo]), H, self.extra["cost"], detail=True)
        out = {k: v[0] for k, v in out.items()}
        out["R"] = d.R[lo:lo + H]
        out["dates"] = d.dates[lo:lo + H]
        return out


class InventoryProblem(Problem):
    def _norm(self):
        starts = PathSampler(*self.splits["train"], self.H, self.valid).grid(10)
        r = self.run(P.I1(self.extra["mu"]), np.array([self.extra["cfg"].L + 1.0]), starts)
        return float(max(r.std(), 1e-6))

    def detail(self, policy, theta_phys, split, integer=False, stride=10):
        e = self.extra
        lo, hi = self.splits[split]
        starts = PathSampler(lo, hi, self.H, self.valid).grid(stride)
        return inventory_env.simulate(policy, theta_phys, e["demand"], e["price"], e["ctx"], starts, self.H, e["cfg"],
                                      integer=integer, detail=True)


def build_trading_problem(data, policy_name, tcfg, splits=None, name="TICKER"):
    splits = splits or make_splits(data.n)
    tr0, tr1 = splits["train"]
    mu, sd = data.X[tr0:tr1].mean(axis=0), data.X[tr0:tr1].std(axis=0)
    policy = P.T1(mu, sd) if policy_name == "T1" else P.T2()
    cost = tcfg["cost_bps"] * 1e-4
    H = tcfg["horizon"]
    run = lambda pol, th, starts: trading_env.simulate(pol, th, data.X, data.R, starts, H, cost)
    return TradingProblem("trading", name, policy, run, H, tcfg["n_paths"], splits, lam=tcfg.get("lam", 0.0),
                          extra={"data": data, "cost": cost})


def build_inventory_problem(df, policy_name, icfg, name="SERIES"):
    """df from data_m5.load_series (or any frame with sales, price, price_ok, wday0, event, snap)."""
    n = len(df)
    splits = make_splits(n)
    tr0, tr1 = splits["train"]
    demand = df["sales"].to_numpy(float)
    price = df["price"].to_numpy(float)
    mu = float(demand[tr0:tr1].mean())
    mp = float(price[tr0:tr1].mean())
    L, H = icfg["lead_time"], icfg["horizon"]
    cfg = inventory_env.InvCfg(L=L, unit_cost=0.6 * mp, holding=0.005 * 0.6 * mp, stockout=0.2 * mp,
                               fixed=0.5 * 0.6 * mp, init_inv=mu * (L + 1))
    ctx = {"wday": df["wday0"].to_numpy(int), "event": df["event"].to_numpy(float), "snap": df["snap"].to_numpy(float),
           "price_rel": price / mp - 1.0}
    bad = np.r_[0, np.cumsum(~df["price_ok"].to_numpy())]
    valid = np.zeros(n, bool)
    s = np.arange(0, n - H)
    valid[s] = (bad[s + H] - bad[s]) == 0
    pol = {"I1": P.I1, "I2": P.I2, "I3": P.I3}[policy_name](mu)
    run = lambda p, th, starts, integer=False: inventory_env.simulate(p, th, demand, price, ctx, starts, H, cfg, integer)
    return InventoryProblem("inventory", name, pol, run, H, icfg["n_paths"], splits, valid=valid, vs=5, ds=20,
                            extra={"mu": mu, "cfg": cfg, "demand": demand, "price": price, "ctx": ctx})


def trading_benchmarks(prob):
    """Hand-tuned benchmarks (train only): returns {name: (policy, theta_phys)}."""
    out = {"buy_and_hold": (P.Fixed(1.0, "buy_and_hold"), np.zeros(0)), "flat": (P.Fixed(0.0, "flat"), np.zeros(0)),
           "SAR_rule": (P.SarRule(), np.zeros(0))}
    rsi = P.RsiRule()
    J = prob.objective(rsi, "train", fixed_stride=5)
    axes = [np.arange(15, 46, 5) / 50.0, np.arange(55, 86, 5) / 50.0]
    u, _ = grid_tune(J, axes)
    out["RSI_rule"] = (rsi, u * rsi.scale)
    return out


def inventory_benchmarks(prob):
    mu, L = prob.extra["mu"], prob.extra["cfg"].L
    i1, i2 = P.I1(mu), P.I2(mu)
    u1, _ = grid_tune(prob.objective(i1, "train", fixed_stride=5), [np.linspace(0, 10, 101)])
    u2, _ = grid_tune(prob.objective(i2, "train", fixed_stride=5), [np.linspace(0, 8, 17), np.linspace(-3, 3, 13)])
    return {"tuned_base_stock": (i1, u1), "tuned_sS": (i2, u2), "cover_(L+1)mu": (i1, np.array([L + 1.0]))}
