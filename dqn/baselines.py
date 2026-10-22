"""
baselines.py -- classical inventory-control policies evaluated in the same
InventoryEnv, plus simulation-based tuning routines for their parameters.

All policies are expressed as callables `policy(env) -> action_index`, so
they can be run through identical episode loops as the DQN agent
(see evaluate.py). Continuous order quantities chosen by a policy are
snapped to the nearest value on the environment's discrete order grid.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from env import InventoryEnv

Policy = Callable[[InventoryEnv], int]


def _snap_to_grid(env: InventoryEnv, qty: float) -> int:
    idx = int(np.argmin(np.abs(env.order_grid - qty)))
    return idx


def never_order_policy() -> Policy:
    def policy(env: InventoryEnv) -> int:
        return _snap_to_grid(env, 0.0)

    return policy


def fixed_order_policy(qty: float) -> Policy:
    def policy(env: InventoryEnv) -> int:
        return _snap_to_grid(env, qty)

    return policy


def base_stock_policy(S: float) -> Policy:
    """Order-up-to policy: order = max(0, S - inventory_position) each day."""

    def policy(env: InventoryEnv) -> int:
        order = max(0.0, S - env.inventory_position())
        return _snap_to_grid(env, order)

    return policy


def s_S_policy(s: float, S: float) -> Policy:
    """(s, S) policy: if inventory position < s, order up to S, else order nothing."""

    def policy(env: InventoryEnv) -> int:
        pos = env.inventory_position()
        order = (S - pos) if pos < s else 0.0
        return _snap_to_grid(env, max(0.0, order))

    return policy


def newsvendor_policy(target_level: float) -> Policy:
    """Newsvendor / critical-ratio policy: order up to the empirical-quantile
    based target level each day (same mechanics as base-stock, separate
    constructor so tuning/reporting can distinguish the two)."""
    return base_stock_policy(target_level)


# --------------------------------------------------------------------------- #
# Simulation helpers for tuning policy parameters on the train split
# --------------------------------------------------------------------------- #
def simulate_policy(env: InventoryEnv, policy: Policy, n_episodes: int = 10, seed: int = 0):
    """Run `policy` for n_episodes windows, return dict of aggregate metrics."""
    rng = np.random.default_rng(seed)
    returns, fill_rates, stockout_rates, avg_inventory, n_orders = [], [], [], [], []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        done = False
        ep_return = 0.0
        fills, stockouts, invs = [], [], []
        orders = 0
        while not done:
            a = policy(env)
            obs, reward, terminated, truncated, step_info = env.step(a)
            done = terminated or truncated
            ep_return += reward
            fills.append(step_info["fill_rate"])
            stockouts.append(step_info["stockout"])
            invs.append(step_info["on_hand"])
            if step_info["order_qty"] > 0:
                orders += 1
        returns.append(ep_return)
        fill_rates.append(np.mean(fills))
        stockout_rates.append(np.mean(stockouts))
        avg_inventory.append(np.mean(invs))
        n_orders.append(orders)

    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_fill_rate": float(np.mean(fill_rates)),
        "mean_stockout_rate": float(np.mean(stockout_rates)),
        "mean_inventory": float(np.mean(avg_inventory)),
        "mean_n_orders": float(np.mean(n_orders)),
    }


def tune_base_stock(env: InventoryEnv, mean_demand: float, lead_time: int, review_period: int = 1,
                     n_episodes: int = 10, seed: int = 0, safety_multiples=(0, 0.5, 1.0, 1.5, 2.0, 3.0)):
    """Grid-search safety stock on top of the analytic base-stock level
    S = mean_demand * (L + review_period) + safety_stock, maximizing mean
    simulated return on the (train-split) env."""
    best = None
    best_S = None
    for k in safety_multiples:
        S = mean_demand * (lead_time + review_period) + k * mean_demand
        metrics = simulate_policy(env, base_stock_policy(S), n_episodes=n_episodes, seed=seed)
        if best is None or metrics["mean_return"] > best["mean_return"]:
            best, best_S = metrics, S
    return best_S, best


def tune_s_S(env: InventoryEnv, mean_demand: float, s_multiples, S_multiples,
             n_episodes: int = 10, seed: int = 0):
    best = None
    best_params = None
    for sm in s_multiples:
        for Sm in S_multiples:
            s_val = sm * mean_demand
            S_val = Sm * mean_demand
            if S_val <= s_val:
                continue
            metrics = simulate_policy(env, s_S_policy(s_val, S_val), n_episodes=n_episodes, seed=seed)
            if best is None or metrics["mean_return"] > best["mean_return"]:
                best, best_params = metrics, (s_val, S_val)
    return best_params, best


def tune_newsvendor(env: InventoryEnv, demand_history: np.ndarray, quantiles, lead_time: int,
                     review_period: int = 1, n_episodes: int = 10, seed: int = 0):
    """Target level = empirical demand quantile over (L + review_period) days,
    approximated as quantile(daily demand) * (L + review_period)."""
    best = None
    best_q = None
    horizon = lead_time + review_period
    for q in quantiles:
        level = float(np.quantile(demand_history, q)) * horizon
        metrics = simulate_policy(env, newsvendor_policy(level), n_episodes=n_episodes, seed=seed)
        if best is None or metrics["mean_return"] > best["mean_return"]:
            best, best_q = metrics, q
    return best_q, best
