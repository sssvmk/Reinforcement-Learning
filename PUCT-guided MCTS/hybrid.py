"""STRETCH (kept separate from the core deliverables): hybrid of alpha-beta search and the trained network.

HybridAgent = the iterative-deepening alpha-beta of baselines.NegamaxAgent, but
  * non-terminal leaves at the depth horizon are scored with the network VALUE head (scaled to +-VALUE_SCALE,
    far below the exact win score), instead of the window-count heuristic;
  * the root move order is the network POLICY (highest prior first) rather than centre-first (interior nodes keep
    centre-first ordering).
Exact wins/losses/draws are still found by the search, only the horizon evaluation is learned.
"""
from __future__ import annotations

import numpy as np

from baselines import NegamaxAgent
from game import State

VALUE_SCALE = 2000


class HybridAgent(NegamaxAgent):
    def __init__(self, evaluator, depth: int = 42, time_limit: float | None = None, seed: int = 0,
                 policy_root_order: bool = True, tt_max: int = 2_000_000):
        super().__init__(depth, time_limit, seed, tt_max)
        self.ev = evaluator
        self.policy_root_order = policy_root_order
        self.vcache: dict = {}
        self.name = f"hybrid(d={depth}" + (f",t={time_limit}" if time_limit else "") + ")"

    def _eval(self, cur: int, mask: int) -> int:
        key = (mask, cur)
        v = self.vcache.get(key)
        if v is None:
            sh = self.g.cell_shift
            planes = np.stack([(cur >> sh) & 1, ((cur ^ mask) >> sh) & 1]).astype(np.float32)[None]
            legal = np.ones((1, self.g.cols), dtype=bool)
            _, val = self.ev.evaluate(planes, legal)
            v = int(round(float(val[0]) * VALUE_SCALE))
            if len(self.vcache) > 500_000:
                self.vcache.clear()
            self.vcache[key] = v
        return v

    def select(self, state: State) -> int:
        if self.policy_root_order:
            g = state.g
            planes = state.to_planes(getattr(self.ev, "n_planes", 2))[None]
            legal = np.zeros((1, g.cols), dtype=bool)
            legal[0, state.legal_moves()] = True
            pri, _ = self.ev.evaluate(planes, legal)
            # root-only ordering by policy; interior nodes keep centre-first ordering
            self.root_order = [int(c) for c in np.argsort(-pri[0], kind="stable")]
        return super().select(state)
