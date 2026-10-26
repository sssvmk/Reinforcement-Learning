import numpy as np
import pytest

from candidates.mcts_policy import _step
from conftest import make_series
from env import Day, Env, EnvParams, Episodes, make_params, make_episodes, rollout, transition, DemandModel


def params(L=2, **kw):
    base = dict(lead_time=L, gamma=0.99, horizon=10, mean_demand=5.0, mean_price=2.0, unit_cost=1.2,
                holding_cost=0.01, stockout_penalty_frac=0.2, fixed_order_cost=0.3,
                actions=np.array([0.0, 5.0, 10.0]), r_scale=10.0)
    base.update(kw)
    return EnvParams(**base)


def test_hand_calculated_reward():
    p = params(L=2)
    s = np.array([4.0, 3.0, 2.0])          # on-hand 4, pipeline [3, 2]
    nxt, r, info = transition(s, 10.0, 6.0, Day(2.0), p)
    # sales=min(4,6)=4, lost=2, on_hand_end=0
    exp = 2.0 * 4 - 1.2 * 10 - 0.01 * 0 - 0.2 * 2.0 * 2 - 0.3
    assert r == pytest.approx(exp)
    assert info["sales"] == 4 and info["lost"] == 2
    assert np.allclose(nxt, [0 + 3, 2 + 10, 0])   # slot0 arrives tomorrow, order (L=2) lands in slot L-1


@pytest.mark.parametrize("L", [0, 1, 2])
def test_timeline_lead_time(L):
    p = params(L=L)
    s0 = np.zeros(1 + L)
    if L == 0:      # order arrives before demand is revealed, same day
        nxt, _, info = transition(s0, 10.0, 7.0, Day(2.0), p)
        assert info["sales"] == 7 and nxt[0] == 3
        return
    # order placed on day 0 is on hand at the START of day L (not earlier); demand on day 0 sees nothing
    _, _, info = transition(s0, 10.0, 7.0, Day(2.0), p)
    assert info["sales"] == 0 and info["lost"] == 7
    st, _, _ = transition(s0, 10.0, 0.0, Day(2.0), p)
    for k in range(1, L + 1):
        assert st[0] == (10.0 if k == L else 0.0), (L, k, st)
        st, _, _ = transition(st, 0.0, 0.0, Day(2.0), p)


@pytest.mark.parametrize("L", [0, 1, 2])
def test_conservation_and_nonnegative(L):
    p = params(L=L)
    rng = np.random.default_rng(1)
    s = np.concatenate([[5.0], np.full(L, 5.0)])
    total0, cum_sales, cum_orders = s.sum(), 0.0, 0.0
    for _ in range(200):
        a = float(rng.choice(p.actions))
        d = float(rng.poisson(5))
        s, _, info = transition(s, a, d, Day(2.0), p)
        cum_sales += info["sales"]; cum_orders += a
        assert s.min() >= 0
        assert s.sum() + cum_sales - cum_orders == pytest.approx(total0)
        assert info["lost"] >= 0 and info["on_hand_end"] >= 0


def test_batch_equals_scalar_and_mcts_step():
    p = params(L=2)
    rng = np.random.default_rng(0)
    for _ in range(50):
        s = rng.integers(0, 10, 3).astype(float)
        a, d = float(rng.choice(p.actions)), float(rng.integers(0, 12))
        n1, r1, _ = transition(s, a, d, Day(2.0), p)
        on, pipe, r2 = _step(s[0], list(s[1:]), a, d, 2.0, p)
        assert np.allclose(n1, [on] + pipe) and r1 == pytest.approx(r2)


def test_determinism_and_env_wrapper(cfg, series):
    p = make_params(series, cfg)
    dm = DemandModel(series.sales[:600], series.wday[:600], series.snap[:600])
    ep = make_episodes(series, cfg, dm, "train", np.arange(4), 3)
    ep2 = make_episodes(series, cfg, dm, "train", np.arange(4), 3)
    assert np.array_equal(ep.demand, ep2.demand) and np.array_equal(ep.start, ep2.start)
    act = lambda s, t: np.full(len(s), 2)
    assert np.array_equal(rollout(ep, p, act)["ret"], rollout(ep2, p, act)["ret"])
    env = Env(p, ep, 0)
    ret, s, tr = 0.0, env.reset(), False
    t = 0
    while not tr:
        s, r, done, tr, _ = env.step(2)
        ret += p.gamma ** t * r
        t += 1
        assert not done
    assert ret == pytest.approx(rollout(ep, p, act)["ret"][0])
    assert tr and t == cfg["env"]["horizon"]
