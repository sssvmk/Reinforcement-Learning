import copy, numpy as np, pytest
from common import load_config, make_folds
import data_m5
from domains_inventory import InvDomain
from inventory_env import InventoryEnv

CFG = load_config()
C2 = copy.deepcopy(CFG)
C2["splits"] = CFG["inventory"]["splits"]


@pytest.fixture(scope="module")
def dom():
    ser = data_m5.load_inventory_series(CFG, "outputs")["primary"]
    return InvDomain(ser, make_folds(ser.N, C2)["primary"], C2)


@pytest.mark.parametrize("L", [0, 1, 2, 4])
def test_vec_step_matches_scalar_env(dom, L):
    d = InvDomain(dom.s, dom.fold, C2, lead_time=L)
    rng = np.random.default_rng(L)
    env = InventoryEnv(d.s.demand, 0.0, d.uc, d.hold, 0.0, d.fix, L, 10 ** 6)
    on, pipe = np.array([d.init_on]), np.zeros((1, d.npipe))
    env.set_state(100, d.init_on, np.zeros(d.npipe))
    tot_in = d.init_on
    tot_sales = 0.0
    for k in range(60):
        t = 100 + k
        q = float(d.Q[rng.integers(0, 8)])
        env.p, env.pen = d.price[t], d.pen_frac * d.price[t]
        r_env, info = env.step(q)
        r, t2, on, pipe, sales, lost = d.vec_step(t, on, pipe, np.array([q]))
        assert np.isclose(r[0], r_env) and np.isclose(on[0], env.on_hand)
        assert np.allclose(pipe[0], env.pipe) and np.isclose(sales[0], info["sales"])
        # conservation: initial + ordered = sold + on-hand + pipeline
        tot_in += q
        tot_sales += sales[0]
        assert np.isclose(tot_in, tot_sales + on[0] + pipe.sum())


def test_hand_reward(dom):
    d = dom
    t = 200
    on, pipe = np.array([5.0]), np.zeros((1, d.npipe))
    dem = d.s.demand[t]
    r, *_ = d.vec_step(t, on, pipe, np.array([0.0]))
    s = min(5.0, dem)
    exp = d.price[t] * s - d.hold * (5 - s) - d.pen_frac * d.price[t] * (dem - s)
    assert np.isclose(r[0], exp)


def test_splits_disjoint_and_ordered(dom):
    f = dom.fold
    assert f.train[1] <= f.val[0] and f.val[1] <= f.test[0]
    assert f.ope[1] <= f.tune[0] and f.tune[1] <= f.val[1] + 1


def test_behavior_probabilities_valid(dom):
    pool = dom.build_pool(1)
    rng = np.random.default_rng(0)
    log = dom.sample_log("ope", 50, 0.3, rng)
    pb = np.asarray(log.pb) if hasattr(log, "pb") else None
    if pb is not None:
        assert (pb > 0).all() and (pb <= 1).all()
    assert len(pool) >= 20


def test_truth_deterministic(dom):
    pool = dom.build_pool(1)[:3]
    rng = np.random.default_rng(1)
    log = dom.sample_log("ope", 20, 0.3, rng)
    sidx = np.stack([rng.integers(0, 20, 10), rng.integers(0, dom.state_step_max, 10)], 1)
    a = dom.truth(pool, log, 0.1, "ope", sidx)
    b = dom.truth(pool, log, 0.1, "ope", sidx)
    assert np.allclose(a[0], b[0]) and np.allclose(a[1], b[1])
