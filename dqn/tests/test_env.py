"""Sanity checks #1, #2, #7 from the task spec: conservation, determinism,
and a hand-calculated reward example."""
import numpy as np

from env import InventoryEnv, CostConfig
from baselines import fixed_order_policy, base_stock_policy


def test_conservation_and_nonnegative_inventory(synthetic_bundle):
    env = InventoryEnv(synthetic_bundle, split="train", window_len=60, lead_time=2, seed=1)
    obs, info = env.reset(seed=1)
    policy = base_stock_policy(env.bundle.mean_demand * 3)

    initial_on_hand = env.on_hand
    cum_orders = 0.0
    cum_sales = 0.0
    done = False
    while not done:
        a = policy(env)
        order_qty_idx = a
        obs, r, term, trunc, step_info = env.step(a)
        done = term or trunc
        cum_orders += step_info["order_qty"]
        cum_sales += step_info["sales"]

        assert env.on_hand >= -1e-9, "on-hand inventory went negative"
        lhs = env.on_hand + env.pipeline.sum()
        rhs = initial_on_hand + cum_orders - cum_sales
        assert np.isclose(lhs, rhs, atol=1e-6), f"conservation violated: {lhs} != {rhs}"


def test_determinism_same_seed_same_policy(synthetic_bundle):
    policy = fixed_order_policy(synthetic_bundle.mean_demand)

    def run(seed):
        env = InventoryEnv(synthetic_bundle, split="train", window_len=50, lead_time=2, seed=seed)
        obs, info = env.reset(seed=seed)
        total_reward = 0.0
        done = False
        while not done:
            a = policy(env)
            obs, r, term, trunc, step_info = env.step(a)
            done = term or trunc
            total_reward += r
        return total_reward

    r1 = run(123)
    r2 = run(123)
    assert r1 == r2


def test_reward_hand_calculation(conftest_module):
    # Build a 2-day bundle with known demand/price so the reward can be
    # computed by hand.
    bundle = conftest_module.make_synthetic_bundle(
        n_days=20, demand_fn=lambda n: np.full(n, 3.0), mean_price=10.0
    )
    cost = CostConfig(unit_cost_frac=0.6, holding_cost_frac=0.005, stockout_penalty_frac=0.2, fixed_order_cost=1.0)
    env = InventoryEnv(bundle, split="train", window_len=5, lead_time=2,
                        init_inventory_days=3.0, cost_config=cost, seed=0)
    obs, info = env.reset(seed=0)

    # init on_hand = round(3 * mean_demand) = round(3*3) = 9
    assert env.on_hand == 9.0

    order_idx = int(np.argmin(np.abs(env.order_grid - 0)))  # order 0 this step
    obs, reward, term, trunc, step_info = env.step(order_idx)

    unit_cost = 0.6 * 10.0          # = 6.0
    holding_cost = 0.005 * unit_cost  # = 0.03
    demand_t = 3.0
    sales = min(9.0, demand_t)       # = 3.0
    lost_sales = demand_t - sales    # = 0.0
    on_hand_end = 9.0 - sales        # = 6.0
    expected_reward = (
        10.0 * sales - unit_cost * 0.0 - holding_cost * on_hand_end
        - 0.2 * 10.0 * lost_sales - 1.0 * 0.0
    )
    assert np.isclose(reward, expected_reward, atol=1e-6)
    assert np.isclose(env.on_hand, on_hand_end)
