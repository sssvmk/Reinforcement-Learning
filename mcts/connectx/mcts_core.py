"""Generic UCT Monte Carlo Tree Search over an abstract two-player,
zero-sum game-state interface.

Required state interface (duck-typed, see connectx_state.ConnectXState):
    legal_actions() -> list[int]
    apply(action) -> new_state
    is_terminal() -> bool
    reward(perspective_mark) -> float   # +1/-1/0 from perspective_mark's view
    current_player() -> int             # mark of the player to move

Selection uses UCT: Q/N + c*sqrt(ln(N_parent)/N). Backpropagation flips the
sign of the simulated value at every ply (zero-sum, two-player minimax
convention): a value good for the player who just moved is bad (negated)
for the player to move one ply up.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


class Node:
    __slots__ = ("state", "parent", "action_from_parent", "children", "untried_actions",
                 "N", "W", "player_to_move")

    def __init__(self, state, parent=None, action_from_parent=None):
        self.state = state
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.children: Dict[int, "Node"] = {}
        self.untried_actions = list(state.legal_actions()) if not state.is_terminal() else []
        self.N = 0
        self.W = 0.0  # total value from the perspective of player_to_move at this node
        self.player_to_move = state.current_player()

    def q(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    def best_uct_child(self, c: float) -> "Node":
        # Each child's Q is stored from the CHILD's own player-to-move
        # perspective (the opponent of this node's mover), so the
        # exploitation term here must be negated to reflect this node's
        # perspective (negamax-style value flip).
        log_n = math.log(self.N)
        best, best_score = None, -float("inf")
        for child in self.children.values():
            exploit = -child.q()
            explore = c * math.sqrt(log_n / child.N) if child.N > 0 else float("inf")
            score = exploit + explore
            if score > best_score:
                best, best_score = child, score
        return best


@dataclass
class MCTSConfig:
    c: float = 1.41
    n_simulations: Optional[int] = 200
    time_budget_seconds: Optional[float] = None
    rollout: str = "heuristic"  # "random" or "heuristic"
    robust_child: bool = True   # False -> pick max-value (highest Q) child
    rng: random.Random = field(default_factory=random.Random)


def _default_rollout(state, rng: random.Random, rollout_type: str, heuristic_fn: Optional[Callable] = None):
    """Rolls out to a terminal state, returning that terminal state."""
    s = state
    while not s.is_terminal():
        actions = s.legal_actions()
        action = None
        if rollout_type == "heuristic" and heuristic_fn is not None:
            action = heuristic_fn(s, actions, rng)
        if action is None:
            action = rng.choice(actions)
        s = s.apply(action)
    return s


def heuristic_rollout_policy(state, actions, rng: random.Random):
    """Take an immediate win; else block an immediate opponent win; else None
    (caller falls back to random). Works for any state exposing
    legal_actions/apply/is_terminal/reward/current_player."""
    me = state.current_player()
    opp = 2 if me == 1 else 1
    # Immediate win
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


def run_mcts(root_state, config: MCTSConfig) -> Dict[str, object]:
    """Runs MCTS from `root_state`. Returns a dict with the chosen action and
    diagnostics (visit counts, root value estimate, simulations run)."""
    root = Node(root_state)
    rng = config.rng

    def heuristic_fn(s, actions, r):
        return heuristic_rollout_policy(s, actions, r)

    start = time.time()
    sims_run = 0

    def should_continue():
        if config.time_budget_seconds is not None:
            return (time.time() - start) < config.time_budget_seconds
        return sims_run < (config.n_simulations or 0)

    # Special case: single legal action, no need to search.
    if len(root.untried_actions) == 1 and not root.children:
        only = root.untried_actions[0]
        return {"action": only, "visits": {only: 1}, "root_value": 0.0, "simulations": 0}

    while should_continue():
        node = root
        # 1. Selection
        while node.is_fully_expanded() and node.children and not node.state.is_terminal():
            node = node.best_uct_child(config.c)
        # 2. Expansion
        if not node.state.is_terminal() and node.untried_actions:
            action = node.untried_actions.pop(rng.randrange(len(node.untried_actions)))
            child_state = node.state.apply(action)
            child = Node(child_state, parent=node, action_from_parent=action)
            node.children[action] = child
            node = child
        # 3. Simulation (rollout)
        terminal_state = _default_rollout(node.state, rng, config.rollout, heuristic_fn)
        # 4. Backpropagation (value from the perspective of the player to move at `node`)
        value = terminal_state.reward(node.player_to_move)
        cur = node
        while cur is not None:
            cur.N += 1
            cur.W += value
            value = -value  # flip sign going up one ply (opponent's perspective)
            cur = cur.parent
        sims_run += 1

    if not root.children:
        # Terminal root edge case.
        action = root_state.legal_actions()[0] if root_state.legal_actions() else None
        return {"action": action, "visits": {}, "root_value": 0.0, "simulations": sims_run}

    if config.robust_child:
        best_action = max(root.children.items(), key=lambda kv: kv[1].N)[0]
    else:
        # max-value child: child.q() is from the CHILD's own perspective, so
        # the best action for the root's mover maximizes -child.q().
        best_action = max(root.children.items(), key=lambda kv: -kv[1].q())[0]

    visits = {a: c.N for a, c in root.children.items()}
    root_value = root.q()
    return {"action": best_action, "visits": visits, "root_value": root_value, "simulations": sims_run}
