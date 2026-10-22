"""ConnectX agents: random, greedy, negamax (alpha-beta), and MCTS, each
exposed both as a plain callable(state) -> action and as a Kaggle-compatible
`agent(observation, configuration) -> action` wrapper.
"""
from __future__ import annotations

import math
import random
from typing import Callable, Optional

from connectx_state import ConnectXState
from mcts_core import MCTSConfig, run_mcts


# ---------------------------------------------------------------------------
# Random
# ---------------------------------------------------------------------------
def random_policy(state: ConnectXState, rng: Optional[random.Random] = None) -> int:
    rng = rng or random
    return rng.choice(state.legal_actions())


# ---------------------------------------------------------------------------
# Greedy: win / block / else center-preferring
# ---------------------------------------------------------------------------
def greedy_policy(state: ConnectXState, rng: Optional[random.Random] = None) -> int:
    rng = rng or random
    me = state.current_player()
    opp = 2 if me == 1 else 1
    actions = state.legal_actions()

    for a in actions:
        s2 = state.apply(a)
        if s2.is_terminal() and s2.reward(me) > 0:
            return a

    # Block: does the OPPONENT have an immediate winning move on the
    # CURRENT board (before I move)? If so, occupy that column myself.
    opp_view = state.copy()
    opp_view.mark = opp
    for b in opp_view.legal_actions():
        s2 = opp_view.apply(b)
        if s2.is_terminal() and s2.reward(opp) > 0:
            return b

    center = state.columns // 2
    actions_sorted = sorted(actions, key=lambda a: abs(a - center))
    best_dist = abs(actions_sorted[0] - center)
    ties = [a for a in actions_sorted if abs(a - center) == best_dist]
    return rng.choice(ties)


# ---------------------------------------------------------------------------
# Negamax with alpha-beta pruning, depth-limited
# ---------------------------------------------------------------------------
def _score_heuristic(state: ConnectXState, me: int) -> float:
    """Cheap positional heuristic for depth-cutoff leaves: center-column
    preference (classic Connect-4 heuristic proxy)."""
    center = state.columns // 2
    score = 0.0
    for r in range(state.rows):
        for c in range(state.columns):
            v = state.board[c + r * state.columns]
            if v == me:
                score += 1.0 - 0.1 * abs(c - center)
            elif v != 0:
                score -= 1.0 - 0.1 * abs(c - center)
    return score


def negamax_policy(state: ConnectXState, depth: int, rng: Optional[random.Random] = None) -> int:
    rng = rng or random
    me = state.current_player()

    def negamax(s: ConnectXState, d: int, alpha: float, beta: float, player: int) -> float:
        if s.is_terminal():
            return s.reward(player)
        if d == 0:
            return _score_heuristic(s, player) / 1000.0  # scaled small vs. +-1 terminal values
        best = -float("inf")
        for a in s.legal_actions():
            s2 = s.apply(a)
            val = -negamax(s2, d - 1, -beta, -alpha, 2 if player == 1 else 1)
            best = max(best, val)
            alpha = max(alpha, val)
            if alpha >= beta:
                break
        return best

    best_action, best_val = None, -float("inf")
    actions = list(state.legal_actions())
    rng.shuffle(actions)
    for a in actions:
        s2 = state.apply(a)
        val = -negamax(s2, depth - 1, -float("inf"), float("inf"), 2 if me == 1 else 1)
        if val > best_val:
            best_val, best_action = val, a
    return best_action


def make_negamax(depth: int) -> Callable:
    return lambda state, rng=None: negamax_policy(state, depth, rng)


# ---------------------------------------------------------------------------
# MCTS agent
# ---------------------------------------------------------------------------
def make_mcts_agent(n_sims: int = 200, c: float = 1.41, rollout: str = "heuristic",
                     robust_child: bool = True, time_budget_seconds: Optional[float] = None,
                     seed: Optional[int] = None) -> Callable:
    def policy(state: ConnectXState, rng: Optional[random.Random] = None) -> int:
        cfg = MCTSConfig(
            c=c,
            n_simulations=n_sims,
            time_budget_seconds=time_budget_seconds,
            rollout=rollout,
            robust_child=robust_child,
            rng=rng or random.Random(seed),
        )
        result = run_mcts(state, cfg)
        return result["action"]
    return policy


# ---------------------------------------------------------------------------
# Kaggle-compatible wrappers: agent(observation, configuration) -> int
# ---------------------------------------------------------------------------
def kaggle_wrap(policy: Callable, seed: Optional[int] = None) -> Callable:
    rng_holder = {"rng": random.Random(seed)}

    def agent(observation, configuration):
        state = ConnectXState.from_observation(observation, configuration)
        return int(policy(state, rng_holder["rng"]))
    return agent


def random_agent(observation, configuration):
    return kaggle_wrap(random_policy)(observation, configuration)


def greedy_agent(observation, configuration):
    return kaggle_wrap(greedy_policy)(observation, configuration)


def negamax_agent(depth: int = 4):
    return kaggle_wrap(make_negamax(depth))


def mcts_agent(n_sims: int = 200, c: float = 1.41, rollout: str = "heuristic", seed: Optional[int] = None):
    return kaggle_wrap(make_mcts_agent(n_sims=n_sims, c=c, rollout=rollout, seed=seed), seed=seed)
