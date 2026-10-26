import numpy as np
import pandas as pd
import pytest
from scipy.stats import poisson
import inventory_env as ie
import policies as P
from fdpg.optimizer import HParams, run_fd
from problems import build_inventory_problem


class Const:
    n = 0

    def __init__(self, q):
        self.q = q

    def order(self, th, ip, c):
        return np.full_like(ip, self.q)


def test_hand_calculated_reward():
    cfg = ie.InvCfg(L=0, unit_cost=1.0, holding=0.1, stockout=0.5, fixed=0.2, init_inv=0.0)
    r = ie.simulate(Const(3.0), None, np.array([2.0, 5.0]), np.array([10.0, 10.0]), {}, np.array([0]), 2, cfg)
    assert r[0] == pytest.approx(16.7 + 36.3)


@pytest.mark.parametrize("L", [0, 1, 2])
def test_timeline_and_conservation(L):
    rng = np.random.default_rng(0)
    T = 60
    dem = rng.poisson(4, T).astype(float)
    cfg = ie.InvCfg(L=L, init_inv=3.0)
    pol = P.I1(4.0)
    D = ie.simulate(pol, np.array([2.5]), dem, np.ones(T), {}, np.array([0, 5]), 40, cfg, detail=True)
    assert (D["on_hand"] >= -1e-12).all()
    for i in range(2):
        orders, sales = D["order"][i], D["sales"][i]
        for t in range(40):
            arrived = orders[: t - L + 1].sum() if t - L + 1 > 0 else 0.0
            assert D["on_hand"][i, t] == pytest.approx(3.0 + arrived - sales[: t + 1].sum())  # order placed on s arrives on s+L
            pipeline = orders[max(0, t - L + 1): t + 1].sum()
            assert 3.0 + orders[: t + 1].sum() - sales[: t + 1].sum() == pytest.approx(D["on_hand"][i, t] + pipeline)


def test_lead_time_arrival_days():
    cfg = ie.InvCfg(L=2)
    D = ie.simulate(Const(1.0), None, np.zeros(10), np.ones(10), {}, np.array([0]), 5, cfg, detail=True)
    np.testing.assert_allclose(D["on_hand"][0], [0, 0, 1, 2, 3])


def _poisson_df(lam, n, const=False, seed=0):
    rng = np.random.default_rng(seed)
    s = np.full(n, lam) if const else rng.poisson(lam, n)
    return pd.DataFrame({"sales": s, "price": 10.0, "price_ok": True, "wday0": np.arange(n) % 7, "event": 0.0, "snap": 0.0})


def _fit(prob, seed=0, budget=60000):
    Jt, Jv, Jd = prob.objectives()
    hp = HParams(est="central", eps=0.2, a=1.0, A=10, clip=5.0, budget=budget)
    return run_fd(Jt, Jv, np.array([3.0]), prob.policy.ilo, prob.policy.ihi, hp, seed)["u_best"][0]


def test_known_optimum_constant_demand():
    prob = build_inventory_problem(_poisson_df(5, 4000, const=True), "I1", {"lead_time": 0, "horizon": 90, "n_paths": 32})
    assert _fit(prob, budget=30000) == pytest.approx(1.0, abs=0.15)   # S* = demand


def test_newsvendor_critical_ratio():
    prob = build_inventory_problem(_poisson_df(5, 30000), "I1", {"lead_time": 0, "horizon": 90, "n_paths": 128})
    c = prob.extra["cfg"]
    ratio = (10 + c.stockout - c.unit_cost) / (10 + c.stockout - c.unit_cost + c.holding)
    target = poisson.ppf(ratio, 5) / 5.0
    th = _fit(prob, budget=250000)
    assert abs(th - target) < 0.45   # tolerance: tail-quantile estimate from finite samples (documented in README)


def test_fractional_smooth_integer_piecewise():
    prob = build_inventory_problem(_poisson_df(5, 3000), "I1", {"lead_time": 0, "horizon": 90, "n_paths": 32})
    starts = np.arange(0, 2000, 40)
    grid = np.arange(0.5, 3.0, 0.01)
    frac = np.array([prob.run(prob.policy, np.array([t]), starts).mean() for t in grid])
    integ = np.array([prob.run(prob.policy, np.array([t]), starts, True).mean() for t in grid])
    jf, ji = np.abs(np.diff(frac)).max(), np.abs(np.diff(integ)).max()
    assert ji > 5 * jf
    assert (np.abs(np.diff(integ)) < 1e-9).mean() > 0.5   # integer J is piecewise constant over long stretches
