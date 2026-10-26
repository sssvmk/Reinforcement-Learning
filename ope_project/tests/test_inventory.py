"""Part 3 test 1: inventory environment, synthetic demand only (M5 is PENDING)."""
import numpy as np, pytest
from inventory_env import InventoryEnv, synthetic_demand
from common import check_m5, load_config


def env(L, d):
    return InventoryEnv(d, price=10, unit_cost=6, holding=0.1, penalty=2, fixed=1, lead_time=L, q_max=50)


def test1_lead_time_zero_arrives_immediately():
    e = env(0, [4, 4, 4])
    r, i = e.step(10)           # arrives now, sells 4, ends with 6
    assert i["sales"] == 4 and e.on_hand == 6
    assert r == pytest.approx(10 * 4 - 6 * 10 - 0.1 * 6 - 0 - 1)


def test1_lead_time_one_and_two():
    e = env(1, [3, 3, 3, 3])
    r, i = e.step(5)            # nothing on hand: all demand lost, order arrives start of next day
    assert i["lost"] == 3 and e.on_hand == 5
    assert r == pytest.approx(-6 * 5 - 2 * 3 - 1)
    e.step(0)
    assert e.on_hand == 2
    e = env(2, [3, 3, 3, 3])
    e.step(5)
    assert e.on_hand == 0 and e.pipe.tolist() == [5]    # still in transit
    e.step(0)
    assert e.on_hand == 5 and e.pipe.tolist() == [0]


def test1_conservation_and_nonnegativity():
    rng = np.random.default_rng(0)
    for L in (0, 1, 2, 3):
        d = synthetic_demand(300, rng)
        e = env(L, d)
        ordered = sales = 0.0
        for t in range(300):
            q = float(rng.integers(0, 12))
            before = e.total_stock()
            _, i = e.step(q)
            ordered += q
            sales += i["sales"]
            assert e.on_hand >= 0 and (e.pipe >= 0).all()
            # stock identity: new total = old total + order - sales
            assert e.total_stock() == pytest.approx(before + q - i["sales"])
            assert i["sales"] + i["lost"] == pytest.approx(d[t])


def test1_set_state_replays_exactly():
    rng = np.random.default_rng(1)
    d = synthetic_demand(60, rng)
    e1 = env(2, d)
    qs = rng.integers(0, 10, 30)
    traj = []
    for q in qs:
        traj.append(e1.raw_state())
        e1.step(q)
    k = 12
    e2 = env(2, d)
    e2.set_state(int(traj[k][0]), traj[k][1], traj[k][2:])
    r1 = [env_r for env_r in []]
    e1b = env(2, d)
    e1b.set_state(int(traj[k][0]), traj[k][1], traj[k][2:])
    a = [e2.step(q)[0] for q in qs[k:]]
    b = [e1b.step(q)[0] for q in qs[k:]]
    assert a == b
    assert np.allclose(e2.raw_state(), e1.raw_state())


def test1_pending_status_while_braces():
    cfg = load_config()
    assert check_m5(cfg)["status"] in ("PENDING", "READY")
