"""Open-loop MCTS planner for the stochastic inventory-control MDP.

We implement OPEN-LOOP MCTS: tree statistics are keyed on the ACTION
SEQUENCE (not on sampled demand outcomes), and demand is resampled fresh
from the demand model on every simulation. This avoids the state-space
blowup of a closed-loop chance-node tree (which would need progressive
widening over continuous/large demand outcomes) while still letting the
tree concentrate simulations on promising order sequences. This matches
the task's instruction: "Use open-loop MCTS ... as the default."

Decision-time planning: at each real day, run MCTS from the current real
state with a simulation budget, execute the best FIRST action (robust
child, i.e. the most-visited first action), observe the ACTUAL demand,
advance the real environment, and re-plan next day (receding horizon).

Value backup is SINGLE-AGENT (maximizing), not the zero-sum sign-flipping
used in connectx/mcts_core.py -- there is only one decision-maker here, so
each node simply accumulates (sum of) simulated reward-to-go.

Rollout policy beyond the tree (depth > horizon H during a simulation):
a base-stock policy using a simple order-up-to level, or a uniform-random
policy as an ablation.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from inventory_env import InventoryState, InventoryConfig, step_with_demand
from demand_model import DemandModel, DayFeatures


@dataclass
class MCTSInventoryConfig:
    horizon_H: int = 14
    c: float = 1.0
    n_sims: int = 200
    discount: float = 1.0
    rollout_policy: str = "base_stock"  # "base_stock" or "random"
    rollout_base_stock_S: Optional[float] = None
    reward_scale: float = 1.0           # normalizes value so `c` is meaningful
    seed: Optional[int] = None


class _Node:
    __slots__ = ("N", "W", "children", "action_taken")

    def __init__(self, action_taken: Optional[int] = None):
        self.N = 0
        self.W = 0.0  # sum of simulated returns-to-go through this node
        self.children: Dict[int, "_Node"] = {}
        self.action_taken = action_taken

    def q(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0


def _day_features(series_df, day_idx: int) -> DayFeatures:
    idx = min(day_idx, len(series_df) - 1)
    row = series_df.iloc[idx]
    return DayFeatures(wday=int(row["wday"]), event_flag=int(row["event_flag"]), snap=int(row["snap"]))


def _price_at(series_df, day_idx: int) -> float:
    idx = min(day_idx, len(series_df) - 1)
    return float(series_df.iloc[idx]["sell_price"])


def _rollout_action(state: InventoryState, grid: List[float], cfg: MCTSInventoryConfig,
                     rng: np.random.Generator) -> int:
    if cfg.rollout_policy == "random":
        return int(rng.integers(0, len(grid)))
    # base_stock: order up to cfg.rollout_base_stock_S (falls back to a
    # sensible default if not provided: 2x the smallest nonzero grid step).
    S = cfg.rollout_base_stock_S if cfg.rollout_base_stock_S is not None else (max(grid) / 2 if grid else 0.0)
    inv_position = state.on_hand + sum(state.pipeline)
    order = max(0.0, S - inv_position)
    diffs = [abs(g - order) for g in grid]
    return int(np.argmin(diffs))


def _simulate_from(state: InventoryState, depth_remaining: int, series_df, grid: List[float],
                    env_cfg: InventoryConfig, mcts_cfg: MCTSInventoryConfig,
                    demand_model: DemandModel, mean_price: float, rng: np.random.Generator) -> float:
    """Rollout (outside the tree) for `depth_remaining` days using the
    configured rollout policy and the demand MODEL (never true future
    demand)."""
    total = 0.0
    discount = 1.0
    s = state
    for _ in range(depth_remaining):
        a_idx = _rollout_action(s, grid, mcts_cfg, rng)
        order_qty = grid[a_idx]
        feats = _day_features(series_df, s.day_idx)
        demand = demand_model.sample(feats, rng)
        price = _price_at(series_df, s.day_idx)
        s, r, _ = step_with_demand(s, order_qty, demand, s.day_idx, price, mean_price, env_cfg)
        total += discount * r
        discount *= mcts_cfg.discount
    return total


def _uct_select(node: _Node, n_actions: int, c: float) -> int:
    best_a, best_score = None, -math.inf
    for a in range(n_actions):
        child = node.children.get(a)
        if child is None or child.N == 0:
            return a  # expand unvisited action first
        exploit = child.q()
        explore = c * math.sqrt(math.log(max(node.N, 1)) / child.N)
        score = exploit + explore
        if score > best_score:
            best_score, best_a = score, a
    return best_a


def plan(root_state: InventoryState, series_df, grid: List[float], env_cfg: InventoryConfig,
         mcts_cfg: MCTSInventoryConfig, demand_model: DemandModel, mean_price: float,
         rng: Optional[np.random.Generator] = None) -> Tuple[int, Dict]:
    """Runs open-loop MCTS from root_state for mcts_cfg.n_sims simulations,
    returns (best_action_idx, stats). Demand is resampled fresh from
    `demand_model` on every simulation (open-loop); never true future demand."""
    rng = rng or np.random.default_rng(mcts_cfg.seed)
    n_actions = len(grid)
    root = _Node()
    t0 = time.time()

    for _ in range(mcts_cfg.n_sims):
        # --- selection + expansion (within the tree, up to horizon H) ---
        # `trajectory` stores (child_node, reward_of_the_edge_into_child)
        # in forward order, so backprop can attribute each edge's own
        # reward to the CHILD it leads to (the node whose Q is actually
        # read by its parent's UCT selection) -- not to the node the edge
        # leaves from. Getting this attribution right is essential: it is
        # what makes the immediate cost of a large order actually show up
        # in the Q-value used to choose between actions.
        trajectory: List[Tuple[_Node, float]] = []
        node = root
        # The root has no incoming edge, so it never appears as a "child"
        # in `trajectory` and _backprop would never increment its N. But
        # _uct_select(root, ...) needs root.N as the parent-visit count
        # for the UCB explore term of root's own children -- without this,
        # log(root.N) would stay stuck at log(1)=0 forever, permanently
        # killing exploration at the root (an action that looks bad after
        # just one or two unlucky samples would never be revisited).
        root.N += 1
        s = root_state
        depth = 0
        while depth < mcts_cfg.horizon_H:
            a = _uct_select(node, n_actions, mcts_cfg.c)
            child = node.children.get(a)
            is_new = child is None
            if is_new:
                child = _Node(action_taken=a)
                node.children[a] = child
            order_qty = grid[a]
            feats = _day_features(series_df, s.day_idx)
            demand = demand_model.sample(feats, rng)
            price = _price_at(series_df, s.day_idx)
            s, r, _ = step_with_demand(s, order_qty, demand, s.day_idx, price, mean_price, env_cfg)
            trajectory.append((child, r / mcts_cfg.reward_scale))
            node = child
            depth += 1
            if is_new:
                # one new node expanded this simulation; roll out the rest.
                # `leaf_value` is the return-to-go FROM the leaf's own
                # resulting state onward (the leaf's incoming edge reward
                # is already recorded in `trajectory`, not double-counted).
                leaf_value = _simulate_from(s, mcts_cfg.horizon_H - depth, series_df, grid,
                                            env_cfg, mcts_cfg, demand_model, mean_price, rng)
                _backprop(trajectory, leaf_value / mcts_cfg.reward_scale, mcts_cfg.discount)
                break
        else:
            # reached full horizon within the tree without expanding a new
            # node (tree already this deep) -- no rollout beyond it.
            _backprop(trajectory, 0.0, mcts_cfg.discount)

    best_a = max(range(n_actions), key=lambda a: root.children[a].N if a in root.children else -1)
    stats = {
        "n_sims": mcts_cfg.n_sims,
        "seconds": time.time() - t0,
        "visit_counts": {a: (root.children[a].N if a in root.children else 0) for a in range(n_actions)},
        "q_values": {a: (root.children[a].q() if a in root.children else 0.0) for a in range(n_actions)},
    }
    return best_a, stats


def _backprop(trajectory: List[Tuple[_Node, float]], leaf_state_value: float, discount: float) -> None:
    """Single-agent (maximizing) backup -- no zero-sum sign flipping, since
    there is only one decision-maker (the planner) acting against an
    exogenous demand model.

    `trajectory` is the forward list of (child_node, reward_into_child).
    Walking it backward, each node's Q = reward_into_it + discount * (Q of
    the next node in the trajectory, recursively grounded by the rollout
    value at the leaf). This correctly folds the immediate cost of each
    action into the Q used by its PARENT's UCT selection -- unlike naively
    storing "value of the resulting state" on each node, which would hide
    how expensive it was to reach that state.
    """
    cur_value = leaf_state_value
    for node, r_edge in reversed(trajectory):
        cur_value = r_edge + discount * cur_value
        node.N += 1
        node.W += cur_value


class ReceedingHorizonPlanner:
    """Wraps `plan()` for repeated decision-time planning: at each real
    day, call `act(state)` to get the next order quantity (grid index),
    then advance the REAL environment with the REAL demand (the caller is
    responsible for this -- the planner never sees it in advance)."""

    def __init__(self, series_df, grid: List[float], env_cfg: InventoryConfig,
                 mcts_cfg: MCTSInventoryConfig, demand_model: DemandModel, mean_price: float):
        self.series_df = series_df
        self.grid = grid
        self.env_cfg = env_cfg
        self.mcts_cfg = mcts_cfg
        self.demand_model = demand_model
        self.mean_price = mean_price
        self.rng = np.random.default_rng(mcts_cfg.seed)
        self.last_stats: Optional[Dict] = None

    def act(self, state: InventoryState) -> int:
        a, stats = plan(state, self.series_df, self.grid, self.env_cfg, self.mcts_cfg,
                         self.demand_model, self.mean_price, self.rng)
        self.last_stats = stats
        return a
