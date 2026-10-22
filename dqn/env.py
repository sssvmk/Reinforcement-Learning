"""
env.py -- Gymnasium-compatible single-item/single-store inventory MDP driven
by exogenous M5 demand history.

MDP summary (see README.md for full derivation and cost assumptions):
  state:  [on_hand, pipeline_sum, trailing7_mean, trailing28_mean,
           sin(wday), cos(wday), event_flag, snap_flag, price_ratio]  (normalized)
  action: index into a discrete order-quantity grid derived from mean demand
  reward: price*sales - unit_cost*order - holding_cost*on_hand_end
          - stockout_penalty*lost_sales - fixed_order_cost*1[order>0]

Dynamics per step, in order: (1) receive arriving pipeline order, (2) place
the new order (joins the pipeline, arrives after `lead_time` days), (3)
observe today's historical demand, (4) sales = min(on_hand, demand), lost
sales = demand - sales (lost-sales, no backorders), (5) update on_hand.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from data import SeriesBundle


@dataclass
class CostConfig:
    unit_cost_frac: float = 0.6
    holding_cost_frac: float = 0.005
    stockout_penalty_frac: float = 0.2
    fixed_order_cost: float = 1.0


class InventoryEnv(gym.Env):
    """Lost-sales, periodic-review inventory control over one M5 item-store series."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        bundle: SeriesBundle,
        split: str = "train",
        window_len: int = 90,
        lead_time: int = 2,
        init_inventory_days: float = 3.0,
        order_grid_multiples=(0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0),
        cost_config: Optional[CostConfig] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.bundle = bundle
        self.split = split
        self.window_len = window_len
        self.lead_time = lead_time
        self.init_inventory_days = init_inventory_days
        self.cost = cost_config or CostConfig()

        mean_d = max(bundle.mean_demand, 1e-3)
        grid = sorted(set(int(round(m * mean_d)) for m in order_grid_multiples))
        if grid[0] != 0:
            grid = [0] + grid
        self.order_grid = np.array(grid, dtype=np.int64)

        self.unit_cost = self.cost.unit_cost_frac * bundle.mean_price
        self.holding_cost = self.cost.holding_cost_frac * self.unit_cost
        self.stockout_penalty_coeff = self.cost.stockout_penalty_frac  # * sell_price at t
        self.fixed_order_cost = self.cost.fixed_order_cost

        self.action_space = spaces.Discrete(len(self.order_grid))
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(9,), dtype=np.float32
        )

        split_range = {
            "train": bundle.train_range,
            "val": bundle.val_range,
            "test": bundle.test_range,
        }[split]
        self._split_start, self._split_end = split_range

        self._rng = np.random.default_rng(seed)
        self._max_on_hand = max(mean_d * 30.0, float(self.order_grid.max()) * 3.0, 1.0)

        # episode-local state, set in reset()
        self._t0 = 0
        self._t = 0
        self.on_hand = 0.0
        self.pipeline = np.zeros(self.lead_time, dtype=np.float64)
        self._cum_sales = 0.0
        self._cum_orders = 0.0
        self._cum_lost = 0.0
        self._n_orders_placed = 0

    # ------------------------------------------------------------------ #
    def seed(self, seed: Optional[int] = None):
        self._rng = np.random.default_rng(seed)

    def _demand_at(self, global_idx: int) -> float:
        return float(self.bundle.demand[global_idx])

    def _trailing_mean(self, global_idx: int, window: int) -> float:
        lo = max(0, global_idx - window)
        hist = self.bundle.demand[lo:global_idx]
        if len(hist) == 0:
            return self.bundle.mean_demand
        return float(np.mean(hist))

    def _build_obs(self) -> np.ndarray:
        g = self._t0 + self._t  # global index into bundle arrays, "today" (not yet observed)
        mean_d = max(self.bundle.mean_demand, 1e-3)
        mean_p = max(self.bundle.mean_price, 1e-3)

        trailing7 = self._trailing_mean(g, 7) / mean_d
        trailing28 = self._trailing_mean(g, 28) / mean_d
        wday = self.bundle.wday[g]
        sin_w = np.sin(2 * np.pi * wday / 7.0)
        cos_w = np.cos(2 * np.pi * wday / 7.0)
        event_flag = float(self.bundle.event_flag[g])
        snap_flag = float(self.bundle.snap_flag[g])
        price_ratio = self.bundle.sell_price[g] / mean_p

        obs = np.array(
            [
                self.on_hand / mean_d,
                self.pipeline.sum() / mean_d,
                trailing7,
                trailing28,
                sin_w,
                cos_w,
                event_flag,
                snap_flag,
                price_ratio,
            ],
            dtype=np.float32,
        )
        return obs

    # ------------------------------------------------------------------ #
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        super().reset(seed=seed)

        max_start = self._split_end - self.window_len
        if max_start <= self._split_start:
            # split too short for a full window: shrink window, start at split start
            self._t0 = self._split_start
            self.window_len = max(1, self._split_end - self._split_start)
        else:
            self._t0 = int(self._rng.integers(self._split_start, max_start + 1))

        self._t = 0
        self.on_hand = float(round(self.init_inventory_days * self.bundle.mean_demand))
        self.pipeline = np.zeros(self.lead_time, dtype=np.float64)
        self._cum_sales = 0.0
        self._cum_orders = 0.0
        self._cum_lost = 0.0
        self._n_orders_placed = 0

        obs = self._build_obs()
        info = {"start_idx": self._t0}
        return obs, info

    def step(self, action: int):
        order_qty = float(self.order_grid[int(action)])
        g = self._t0 + self._t

        # (1) receive arriving order placed `lead_time` days ago
        arriving = self.pipeline[0]
        self.on_hand += arriving
        # shift pipeline, (2) place new order at the end (arrives after lead_time days)
        self.pipeline = np.concatenate([self.pipeline[1:], [order_qty]])

        # (3) observe today's historical demand (exogenous)
        demand_t = self._demand_at(g)
        price_t = float(self.bundle.sell_price[g])

        # (4) sales / lost sales
        sales = min(self.on_hand, demand_t)
        lost_sales = demand_t - sales
        self.on_hand -= sales
        self.on_hand = min(self.on_hand, self._max_on_hand)  # safety cap, documented in README

        reward = (
            price_t * sales
            - self.unit_cost * order_qty
            - self.holding_cost * self.on_hand
            - self.stockout_penalty_coeff * price_t * lost_sales
            - self.fixed_order_cost * (1.0 if order_qty > 0 else 0.0)
        )

        self._cum_sales += sales
        self._cum_orders += order_qty
        self._cum_lost += lost_sales
        if order_qty > 0:
            self._n_orders_placed += 1

        self._t += 1
        terminated = False
        truncated = self._t >= self.window_len

        obs = self._build_obs()
        info = {
            "demand": demand_t,
            "sales": sales,
            "lost_sales": lost_sales,
            "order_qty": order_qty,
            "on_hand": self.on_hand,
            "price": price_t,
            "fill_rate": (sales / demand_t) if demand_t > 0 else 1.0,
            "stockout": 1.0 if lost_sales > 0 else 0.0,
        }
        return obs, float(reward), terminated, truncated, info

    def inventory_position(self) -> float:
        """on-hand + pipeline, used by base-stock / (s,S) baselines."""
        return self.on_hand + float(self.pipeline.sum())

    def render(self):  # pragma: no cover - not needed for this task
        pass
