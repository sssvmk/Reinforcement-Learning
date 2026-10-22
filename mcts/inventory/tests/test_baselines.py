"""Isolated unit tests for baselines.py (never_order, fixed_quantity,
base_stock, (s,S), newsvendor, and the train-split tuning helpers), using
synthetic constant-demand data -- no M5 files needed."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inventory_env import InventoryEnv, InventoryConfig, InventoryState, make_action_grid  # noqa: E402
from baselines import (  # noqa: E402
    never_order_policy, fixed_quantity_policy, base_stock_policy, s_S_policy,
    newsvendor_order_up_to_level, tune_base_stock, tune_s_S, build_all_baselines,
)


def _constant_demand_df(n=120, demand=5.0, price=2.0):
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "d": [f"d_{i+1}" for i in range(n)],
        "date": dates,
        "sales": np.full(n, demand),
        "sell_price": np.full(n, price),
        "wday": [(d.weekday() + 1) for d in dates],
        "weekday": [d.day_name() for d in dates],
        "month": dates.month,
        "event_flag": np.zeros(n, dtype=int),
        "snap": np.zeros(n, dtype=int),
    })


def test_never_order_always_returns_zero_action():
    grid = [0.0, 2.0, 5.0, 10.0]
    policy = never_order_policy(grid)
    state = InventoryState(on_hand=3.0, pipeline=(1.0, 2.0), day_idx=0)
    assert grid[policy(state)] == 0.0


def test_fixed_quantity_always_returns_nearest_grid_qty():
    grid = [0.0, 2.0, 5.0, 10.0]
    policy = fixed_quantity_policy(grid, 4.5)
    state = InventoryState(on_hand=0.0, pipeline=(0.0,), day_idx=0)
    assert grid[policy(state)] == 5.0  # nearest to 4.5


def test_base_stock_orders_up_to_S_minus_inventory_position():
    grid = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    policy = base_stock_policy(grid, S=10.0)
    state = InventoryState(on_hand=3.0, pipeline=(2.0,), day_idx=0)  # position = 5
    assert grid[policy(state)] == 5.0  # order up to 10


def test_base_stock_orders_zero_when_above_S():
    grid = [0.0, 1.0, 2.0, 5.0]
    policy = base_stock_policy(grid, S=5.0)
    state = InventoryState(on_hand=10.0, pipeline=(0.0,), day_idx=0)
    assert grid[policy(state)] == 0.0


def test_s_S_policy_only_orders_when_at_or_below_reorder_point():
    grid = [0.0, 1.0, 2.0, 5.0, 8.0, 10.0]
    policy = s_S_policy(grid, s=3.0, S=10.0)
    above = InventoryState(on_hand=4.0, pipeline=(0.0,), day_idx=0)  # position=4 > s=3
    at_or_below = InventoryState(on_hand=2.0, pipeline=(0.0,), day_idx=0)  # position=2 <= s=3
    assert grid[policy(above)] == 0.0
    assert grid[policy(at_or_below)] == 8.0  # nearest to (10-2)=8


def test_newsvendor_level_matches_known_quantile():
    # Uniform-ish discrete demand so the empirical quantile is checkable.
    demand_samples = np.arange(0, 101, dtype=float)  # 0..100, uniform
    price = 10.0
    unit_cost = 6.0
    stockout_frac = 0.2
    co_under = (price - unit_cost) + stockout_frac * price  # 4 + 2 = 6
    co_over = unit_cost  # 6
    expected_cr = co_under / (co_under + co_over)  # 0.5
    level = newsvendor_order_up_to_level(demand_samples, unit_cost, price, stockout_frac)
    assert abs(level - np.quantile(demand_samples, expected_cr)) < 1e-9


def test_newsvendor_cr_clamped_to_0_1():
    # Degenerate costs that would push CR outside [0,1] must be clamped.
    demand_samples = np.arange(0, 21, dtype=float)
    level = newsvendor_order_up_to_level(demand_samples, unit_cost=0.0, price=10.0, stockout_penalty_frac=1.0)
    assert level == np.quantile(demand_samples, 1.0)


def test_tune_base_stock_prefers_S_near_mean_demand_for_constant_demand():
    df = _constant_demand_df(n=60, demand=5.0, price=2.0)
    cfg = InventoryConfig(lead_time_L=1, episode_days=20)
    grid = make_action_grid(5.0, cfg)

    def env_factory():
        return InventoryEnv(df, cfg=cfg, seed=0, mean_price=2.0, mean_demand=5.0)

    best_S, best_r = tune_base_stock(env_factory, grid, [0, 5, 10, 20, 40], n_episodes=5, seed=0)
    # For constant demand=5/day, a modest S (close to daily demand) should
    # dominate grossly over/under-stocked levels like 0 or 40.
    assert best_S in (5, 10)
    policy_bad = base_stock_policy(grid, 40)
    from baselines import _simulate_mean_reward
    bad_r = _simulate_mean_reward(env_factory, policy_bad, n_episodes=5, seed=0)
    assert best_r > bad_r


def test_tune_s_S_respects_S_greater_than_s_constraint():
    df = _constant_demand_df(n=60, demand=5.0, price=2.0)
    cfg = InventoryConfig(lead_time_L=1, episode_days=20)
    grid = make_action_grid(5.0, cfg)

    def env_factory():
        return InventoryEnv(df, cfg=cfg, seed=0, mean_price=2.0, mean_demand=5.0)

    best_s, best_S, best_r = tune_s_S(env_factory, grid, [0, 5, 10], [5, 10, 20], n_episodes=3, seed=0)
    assert best_S > best_s
    assert np.isfinite(best_r)


def test_build_all_baselines_returns_all_expected_policy_names():
    df = _constant_demand_df(n=60, demand=5.0, price=2.0)
    cfg = InventoryConfig(lead_time_L=1, episode_days=20)
    grid = make_action_grid(5.0, cfg)

    def env_factory():
        return InventoryEnv(df, cfg=cfg, seed=0, mean_price=2.0, mean_demand=5.0)

    policies, tuned = build_all_baselines(
        env_factory, grid, mean_demand=5.0, mean_price=2.0, cfg=cfg,
        train_demand_samples=np.full(50, 5.0), n_tune_episodes=3, seed=0,
    )
    expected_names = {"never_order", "fixed_quantity", "base_stock", "s_S", "newsvendor"}
    assert expected_names.issubset(policies.keys())
    for name, policy in policies.items():
        state = InventoryState(on_hand=5.0, pipeline=(0.0,), day_idx=0)
        a = policy(state)
        assert 0 <= a < len(grid)
    assert "base_stock_S" in tuned and "s_S" in tuned and "newsvendor_level" in tuned


def test_policies_never_return_out_of_range_action_index():
    grid = [0.0, 1.0, 2.0, 5.0, 10.0]
    state = InventoryState(on_hand=100.0, pipeline=(50.0, 50.0), day_idx=0)
    for policy in [
        never_order_policy(grid), fixed_quantity_policy(grid, 3.0),
        base_stock_policy(grid, 5.0), s_S_policy(grid, 2.0, 8.0),
    ]:
        a = policy(state)
        assert 0 <= a < len(grid)
