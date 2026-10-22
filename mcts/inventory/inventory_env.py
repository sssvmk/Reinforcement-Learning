"""Gymnasium-compatible InventoryEnv for one M5 item-store series.

MDP per the task spec:
- State: on-hand inventory, on-order pipeline (units arriving in 1..L days),
  trailing 7/28-day mean demand, day-of-week (cyclical), event flag, SNAP
  flag, and relative sell_price -- all normalized.
- Action: discrete order quantity from a grid {0, q1, ..., Q_max} derived
  from the series' TRAINING-period demand statistics.
- Dynamics each day: (1) receive arriving orders, (2) place new order,
  (3) observe demand d_t from history (exogenous), (4) sales = min(on_hand,
  d_t), (5) lost sales = d_t - sales (lost-sales, no backorders),
  (6) update inventory.
- Reward: sell_price*sales - unit_cost*order_qty - holding_cost*on_hand_end
  - stockout_penalty*lost_sales - fixed_order_cost*1[order_qty>0].

Cost assumptions (M5 has no cost data -- documented, configurable):
  unit_cost = 0.6 * mean(sell_price over training period)
  holding_cost = 0.005 * unit_cost per unit per day
  stockout_penalty = 0.2 * sell_price (that day's price)
  fixed_order_cost: small constant (config)

Episode: a window of N consecutive days sampled from an allowed split
period (train/val/test, chronological, see data_split()). Initial
inventory ~= `initial_inventory_days_of_mean_demand` * mean training demand.

If `gymnasium` is not installed this still works: the same reset()/step()
interface is implemented in plain Python (gymnasium is optional and only
used for the Env base class / spaces, which are not required for our own
MCTS/DQN code to function).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    _HAS_GYMNASIUM = True
except ImportError:
    gym = None
    spaces = None
    _HAS_GYMNASIUM = False


@dataclass(frozen=True)
class InventoryState:
    """Pure, hashable-ish (via tuple pipeline) simulator state, independent
    of the day-history array. Used directly by step_with_demand() and by
    the MCTS planner for simulated rollouts."""
    on_hand: float
    pipeline: Tuple[float, ...]  # pipeline[0] arrives in 1 day, pipeline[-1] in L days
    day_idx: int                # index into the series DataFrame (the "real" calendar day)
    cum_sales: float = 0.0
    cum_orders: float = 0.0
    cum_lost_sales: float = 0.0


@dataclass
class InventoryConfig:
    lead_time_L: int = 2
    episode_days: int = 90
    unit_cost_frac_of_price: float = 0.6
    holding_cost_frac_of_unit_cost_per_day: float = 0.005
    stockout_penalty_frac_of_price: float = 0.2
    fixed_order_cost: float = 0.5
    initial_inventory_days_of_mean_demand: float = 3.0
    action_grid_days_of_mean_demand: Tuple[float, ...] = (0, 0.5, 1, 1.5, 2, 3, 4, 5, 7)


def step_with_demand(state: InventoryState, action_qty: float, demand: float,
                      day_idx: int, price: float, mean_price: float,
                      cfg: InventoryConfig) -> Tuple[InventoryState, float, Dict]:
    """Pure function: one day of inventory dynamics given a SAMPLED or
    REAL demand value. Used identically by InventoryEnv.step() (real
    demand) and by the MCTS planner (sampled demand), guaranteeing they
    agree by construction (see test_env.py::test_step_with_demand_matches_env_step).

    Order of operations (per spec):
      1. receive arriving order (pipeline[0])
      2. place the new order (append to pipeline end)
      3. observe demand (`demand`, supplied by caller)
      4. sales = min(on_hand, demand)
      5. lost_sales = demand - sales
      6. update inventory (on_hand -= sales; holding cost on resulting on_hand)
    """
    unit_cost = cfg.unit_cost_frac_of_price * mean_price
    holding_cost_per_unit = cfg.holding_cost_frac_of_unit_cost_per_day * unit_cost

    # 1. receive arriving order
    on_hand = state.on_hand + (state.pipeline[0] if state.pipeline else 0.0)
    new_pipeline = list(state.pipeline[1:]) if state.pipeline else []

    # 2. place new order (arrives after L days -> goes to the back of the pipe)
    while len(new_pipeline) < cfg.lead_time_L:
        new_pipeline.append(0.0)
    if cfg.lead_time_L > 0:
        new_pipeline[cfg.lead_time_L - 1] += action_qty
        on_hand_after_order = on_hand  # order doesn't change physical stock yet
    else:
        on_hand += action_qty  # L=0: instantaneous delivery
        on_hand_after_order = on_hand

    # 3-5. demand realization
    sales = min(on_hand_after_order, demand)
    lost_sales = demand - sales

    # 6. update inventory
    on_hand_end = on_hand_after_order - sales

    reward = (
        price * sales
        - unit_cost * action_qty
        - holding_cost_per_unit * on_hand_end
        - cfg.stockout_penalty_frac_of_price * price * lost_sales
        - cfg.fixed_order_cost * (1.0 if action_qty > 0 else 0.0)
    )

    new_state = InventoryState(
        on_hand=on_hand_end,
        pipeline=tuple(new_pipeline),
        day_idx=day_idx + 1,
        cum_sales=state.cum_sales + sales,
        cum_orders=state.cum_orders + action_qty,
        cum_lost_sales=state.cum_lost_sales + lost_sales,
    )
    info = {
        "sales": sales, "lost_sales": lost_sales, "on_hand_end": on_hand_end,
        "demand": demand, "order_qty": action_qty, "price": price,
    }
    return new_state, reward, info


def make_action_grid(mean_demand: float, cfg: InventoryConfig) -> List[float]:
    """Order-quantity grid derived from training-period mean demand, rounded
    to whole units (orders are discrete units of a product)."""
    grid = sorted(set(max(0, round(mean_demand * d)) for d in cfg.action_grid_days_of_mean_demand))
    return [float(g) for g in grid]


class InventoryEnv:
    """reset()/step() interface for one item-store series. Subclasses
    gymnasium.Env when available (so it can be used with gymnasium
    wrappers/vector envs); otherwise behaves identically as plain Python.
    """

    metadata = {"render_modes": []}

    def __init__(self, series_df, cfg: Optional[InventoryConfig] = None,
                 day_indices: Optional[np.ndarray] = None, seed: Optional[int] = None,
                 mean_price: Optional[float] = None, mean_demand: Optional[float] = None):
        """
        series_df: output of data.load_series() (full series or a
            train/val/test slice -- see data_split()).
        day_indices: the integer positional indices into series_df that are
            allowed as episode start/content days (for a train/val/test
            split). If None, all rows of series_df are usable.
        mean_price / mean_demand: if given, use these (e.g. computed on the
            TRAINING split only) to derive costs/action-grid, so the
            validation/test envs don't leak future statistics. If None,
            computed from series_df itself (only correct for the training
            env).
        """
        self.series_df = series_df.reset_index(drop=True)
        self.cfg = cfg or InventoryConfig()
        self.rng = np.random.default_rng(seed)
        self.day_indices = (
            np.arange(len(self.series_df)) if day_indices is None else np.asarray(day_indices)
        )
        self.mean_price = float(mean_price if mean_price is not None else self.series_df["sell_price"].mean())
        self.mean_demand = float(mean_demand if mean_demand is not None else self.series_df["sales"].mean())
        self.action_grid = make_action_grid(self.mean_demand, self.cfg)
        self.n_actions = len(self.action_grid)

        self._state: Optional[InventoryState] = None
        self._episode_start: Optional[int] = None
        self._episode_end: Optional[int] = None
        self._t: int = 0  # days elapsed within episode

        if _HAS_GYMNASIUM:
            self.action_space = spaces.Discrete(self.n_actions)
            self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(10,), dtype=np.float32)

    # ---- state featurization -------------------------------------------------
    def _trailing_mean(self, idx: int, window: int) -> float:
        lo = max(0, idx - window)
        vals = self.series_df["sales"].iloc[lo:idx]
        return float(vals.mean()) if len(vals) > 0 else self.mean_demand

    def _featurize(self, state: InventoryState) -> np.ndarray:
        idx = state.day_idx
        row = self.series_df.iloc[min(idx, len(self.series_df) - 1)]
        on_hand_norm = state.on_hand / (self.mean_demand + 1e-6) / 10.0
        pipeline_norm = sum(state.pipeline) / (self.mean_demand + 1e-6) / 10.0
        tm7 = self._trailing_mean(idx, 7) / (self.mean_demand + 1e-6)
        tm28 = self._trailing_mean(idx, 28) / (self.mean_demand + 1e-6)
        wday = int(row["wday"])
        dow_sin = np.sin(2 * np.pi * wday / 7.0)
        dow_cos = np.cos(2 * np.pi * wday / 7.0)
        event_flag = float(row["event_flag"])
        snap = float(row["snap"])
        price = row["sell_price"]
        price_rel = (price / self.mean_price - 1.0) if self.mean_price > 0 else 0.0
        return np.array([
            on_hand_norm, pipeline_norm, tm7, tm28, dow_sin, dow_cos,
            event_flag, snap, price_rel, 0.0,
        ], dtype=np.float32)

    # ---- gym-style API --------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, start_idx: Optional[int] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        max_start = len(self.day_indices) - self.cfg.episode_days - self.cfg.lead_time_L - 1
        if max_start < 0:
            raise ValueError("series/split window too short for the requested episode_days")
        if start_idx is None:
            start_pos = int(self.rng.integers(0, max_start + 1))
        else:
            start_pos = start_idx
        self._episode_start = int(self.day_indices[start_pos])
        self._episode_end = self._episode_start + self.cfg.episode_days
        self._t = 0

        init_inv = self.cfg.initial_inventory_days_of_mean_demand * self.mean_demand
        self._state = InventoryState(
            on_hand=init_inv, pipeline=tuple(0.0 for _ in range(self.cfg.lead_time_L)),
            day_idx=self._episode_start,
        )
        obs = self._featurize(self._state)
        info = {"episode_start": self._episode_start}
        return obs, info

    def step(self, action_idx: int):
        assert self._state is not None, "call reset() first"
        order_qty = self.action_grid[action_idx]
        day_idx = self._state.day_idx
        row = self.series_df.iloc[day_idx]
        demand = float(row["sales"])
        price = float(row["sell_price"])
        new_state, reward, info = step_with_demand(
            self._state, order_qty, demand, day_idx, price, self.mean_price, self.cfg
        )
        self._state = new_state
        self._t += 1
        terminated = False
        truncated = self._t >= self.cfg.episode_days or new_state.day_idx >= self._episode_end
        obs = self._featurize(new_state)
        info.update({"t": self._t, "inventory": new_state.on_hand, "pipeline": new_state.pipeline})
        return obs, reward, terminated, truncated, info

    @property
    def state(self) -> InventoryState:
        return self._state


def data_split(series_df, train_frac: float, val_frac: float, test_frac: float):
    """Chronological split: earliest train_frac days = train, next val_frac
    = validation, last test_frac = test. Returns (train_idx, val_idx, test_idx)
    arrays of positional indices into series_df, guaranteed disjoint and in
    calendar order."""
    n = len(series_df)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    train_idx = np.arange(0, n_train)
    val_idx = np.arange(n_train, n_train + n_val)
    test_idx = np.arange(n_train + n_val, n)
    return train_idx, val_idx, test_idx
