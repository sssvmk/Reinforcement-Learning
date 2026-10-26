"""Tests 7-11: environments."""
import numpy as np
import inventory_env as ie
import trading_env as te
import policies as P
from synth import make_inv, make_trade


class Const:
    stochastic = False

    def __init__(self, q):
        self.q = q

    def act(self, theta, st, rng, mode):
        return np.full(len(st["ip"]), self.q), None


def test_inventory_hand_computed_timeline():
    pr = make_inv(L=1, H=3)
    env = pr.env
    env.demand[:6] = [5, 4, 20, 1, 1, 1]
    env.cfg.init_inv, env.cfg.fixed = 10.0, 2.0
    out = ie.simulate(Const(6.0), None, env, np.array([0]), 3, detail=True)
    c = env.cfg
    # day0: arrivals none (pipe zero), order 6 -> pipe; sales 5; on_hand 5.  day1: +0 arrives? pipe slot holds 6 from day0 -> on_hand 11, sales 4 -> 7
    # day2: +6 -> 13, demand 20 -> sales 13, lost 7
    sales = np.array([5, 4, 13.0])
    lost = np.array([0, 0, 7.0])
    on_hand = np.array([5, 7, 0.0])
    exp = (5.0 * sales).sum() - c.unit_cost * 18 - c.holding * on_hand.sum() - c.stockout * lost.sum() - c.fixed * 3
    np.testing.assert_allclose(out["sales"][0], sales)
    np.testing.assert_allclose(out["lost"][0], lost)
    np.testing.assert_allclose(out["total"][0], exp)


def test_inventory_conservation():
    pr = make_inv(H=40)
    pol = P.BaseStock()
    out = ie.simulate(pol, np.array([3.0]), pr.env, pr.grid("train", 7), 40, detail=True)
    np.testing.assert_allclose(out["sales"] + out["lost"], out["demand"])
    assert (out["on_hand"] >= -1e-9).all() and (out["lost"] >= -1e-12).all()
    # inventory balance: init + arrivals - sales = on_hand_end ; arrivals = sum of orders placed >= L days earlier
    N = out["order"].shape[0]
    L = pr.env.cfg.L
    for i in range(N):
        arr = np.r_[np.zeros(L), out["order"][i, :-L]] if L else out["order"][i]
        np.testing.assert_allclose(pr.env.cfg.init_inv + arr.cumsum() - out["sales"][i].cumsum(), out["on_hand"][i], atol=1e-8)


def test_qmax_and_integer_orders():
    pr = make_inv(qmax=10.0)
    out = ie.simulate(Const(50.0), None, pr.env, np.array([0, 3]), 10, detail=True)
    assert out["order"].max() <= 10.0 + 1e-12
    out = ie.simulate(P.BaseStock(), np.array([3.3]), pr.env, np.array([0]), 20, detail=True, integer=True)
    assert np.allclose(out["order"], np.round(out["order"]))


def test_trading_hand_computed_and_cost():
    pr = make_trade(H=3)
    env = pr.env
    env.R[:3] = [0.01, -0.02, 0.03]
    seq = iter([1.0, 1.0, 0.0])

    class Seq:
        stochastic = False

        def act(self, theta, st, rng, mode):
            return np.full(len(st["t"]), next(seq)), None
    out = te.simulate(Seq(), None, env, np.array([0]), 3, detail=True)
    exp = np.array([0.01 - env.cost, -0.02, 0 - env.cost])
    np.testing.assert_allclose(out["net"][0], exp)


def test_trading_no_lookahead():
    """Actions at day t must not change when R[t:] (the future) is altered."""
    pr = make_trade(H=30)
    pol = P.BernoulliT()
    th = np.random.default_rng(0).standard_normal(7)
    s = np.array([10])
    a = te.simulate(pol, th, pr.env, s, 30, None, "det", detail=True)["pos"].copy()
    R0 = pr.env.R.copy()
    pr.env.R[15:] += 0.5
    b = te.simulate(pol, th, pr.env, s, 30, None, "det", detail=True)["pos"]
    pr.env.R[:] = R0
    np.testing.assert_array_equal(a[0, :16], b[0, :16])
