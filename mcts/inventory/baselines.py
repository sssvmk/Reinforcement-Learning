"""Classical inventory-control baselines, evaluated in the same
InventoryEnv on the same test windows as MCTS.

- never_order: always order 0.
- fixed_quantity: order a fixed quantity every day (tuned to a sensible
  round value from the action grid, e.g. mean daily demand).
- base_stock (order-up-to S): order up to a level S each day:
  order = max(0, S - inventory_position), inventory_position = on_hand + pipeline.
  S tuned on the TRAINING split by simulation (grid search maximizing mean
  training-episode reward).
- (s, S): order up to S only when inventory position falls at/below
  reorder point s; both tuned on train by simulation.
- newsvendor / critical ratio: order-up-to level set analytically from the
  empirical training demand quantile at the critical ratio
  CR = (price - unit_cost) / (price - unit_cost + stockout_penalty_frac*price),
  i.e. a single-period critical-fractile newsvendor level (applied daily as
  an order-up-to policy, which is the standard way to use a newsvendor
  quantile in a repeated-period inventory simulation).

All policies are stateless functions of (on_hand, pipeline, action_grid) so
they can run inside InventoryEnv or MCTS rollouts without caring which.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

from inventory_env import InventoryEnv, InventoryConfig, InventoryState


def _nearest_grid_idx(grid: List[float], target: float) -> int:
    return int(np.argmin([abs(g - target) for g in grid]))


def never_order_policy(grid: List[float]) -> Callable:
    zero_idx = _nearest_grid_idx(grid, 0.0)
    return lambda state: zero_idx


def fixed_quantity_policy(grid: List[float], qty: float) -> Callable:
    idx = _nearest_grid_idx(grid, qty)
    return lambda state: idx


def base_stock_policy(grid: List[float], S: float) -> Callable:
    def policy(state: InventoryState) -> int:
        inv_position = state.on_hand + sum(state.pipeline)
        order = max(0.0, S - inv_position)
        return _nearest_grid_idx(grid, order)
    return policy


def s_S_policy(grid: List[float], s: float, S: float) -> Callable:
    def policy(state: InventoryState) -> int:
        inv_position = state.on_hand + sum(state.pipeline)
        if inv_position <= s:
            order = max(0.0, S - inv_position)
        else:
            order = 0.0
        return _nearest_grid_idx(grid, order)
    return policy


def newsvendor_order_up_to_level(demand_samples: np.ndarray, unit_cost: float,
                                  price: float, stockout_penalty_frac: float) -> float:
    """Critical-ratio newsvendor level from the empirical training-demand
    quantile. CR = Co_under / (Co_under + Co_over) where under-stocking cost
    (lost margin+penalty) = (price - unit_cost) + stockout_penalty_frac*price,
    over-stocking cost = unit_cost (holding/spoilage proxy)."""
    co_under = (price - unit_cost) + stockout_penalty_frac * price
    co_over = unit_cost
    cr = co_under / (co_under + co_over)
    cr = min(max(cr, 0.0), 1.0)
    return float(np.quantile(demand_samples, cr))


def _simulate_mean_reward(env_factory: Callable[[], InventoryEnv], policy: Callable,
                           n_episodes: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    totals = []
    for ep in range(n_episodes):
        env = env_factory()
        env.reset(seed=int(rng.integers(0, 1_000_000)))
        total = 0.0
        done = False
        while not done:
            a = policy(env.state)
            obs, r, term, trunc, info = env.step(a)
            total += r
            done = term or trunc
        totals.append(total)
    return float(np.mean(totals))


def tune_base_stock(env_factory: Callable[[], InventoryEnv], grid: List[float],
                     candidate_S: List[float], n_episodes: int = 10, seed: int = 0) -> Tuple[float, float]:
    """Grid search over candidate order-up-to levels S on the TRAINING
    split, maximizing mean simulated episode reward. Returns (best_S, best_mean_reward)."""
    best_S, best_r = None, -np.inf
    for S in candidate_S:
        policy = base_stock_policy(grid, S)
        r = _simulate_mean_reward(env_factory, policy, n_episodes, seed)
        if r > best_r:
            best_S, best_r = S, r
    return best_S, best_r


def tune_s_S(env_factory: Callable[[], InventoryEnv], grid: List[float],
             candidate_s: List[float], candidate_S: List[float],
             n_episodes: int = 10, seed: int = 0) -> Tuple[float, float, float]:
    best_s, best_S, best_r = None, None, -np.inf
    for s in candidate_s:
        for S in candidate_S:
            if S <= s:
                continue
            policy = s_S_policy(grid, s, S)
            r = _simulate_mean_reward(env_factory, policy, n_episodes, seed)
            if r > best_r:
                best_s, best_S, best_r = s, S, r
    return best_s, best_S, best_r


def build_all_baselines(env_factory: Callable[[], InventoryEnv], grid: List[float],
                         mean_demand: float, mean_price: float, cfg: InventoryConfig,
                         train_demand_samples: np.ndarray,
                         n_tune_episodes: int = 10, seed: int = 0) -> dict:
    """Returns a dict {name: policy_fn(state)->action_idx}, tuning
    base-stock/(s,S)/newsvendor on the training split via `env_factory`
    (which must produce envs restricted to TRAIN days only)."""
    policies = {}
    policies["never_order"] = never_order_policy(grid)
    policies["fixed_quantity"] = fixed_quantity_policy(grid, mean_demand)

    max_level = max(grid) * 2 if max(grid) > 0 else mean_demand * 10
    candidate_S = sorted(set(round(x) for x in np.linspace(0, max_level, 12)))
    best_S, _ = tune_base_stock(env_factory, grid, candidate_S, n_tune_episodes, seed)
    policies["base_stock"] = base_stock_policy(grid, best_S)

    candidate_s = sorted(set(round(x) for x in np.linspace(0, max_level * 0.6, 6)))
    best_s, best_S2, _ = tune_s_S(env_factory, grid, candidate_s, candidate_S, n_tune_episodes, seed)
    policies["s_S"] = s_S_policy(grid, best_s, best_S2)

    unit_cost = cfg.unit_cost_frac_of_price * mean_price
    nv_level = newsvendor_order_up_to_level(
        train_demand_samples, unit_cost, mean_price, cfg.stockout_penalty_frac_of_price
    )
    policies["newsvendor"] = base_stock_policy(grid, nv_level)

    tuned_params = {"base_stock_S": best_S, "s_S": (best_s, best_S2), "newsvendor_level": nv_level}
    return policies, tuned_params
