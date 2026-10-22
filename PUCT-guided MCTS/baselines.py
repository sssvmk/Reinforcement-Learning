"""Baseline agents: random, greedy, negamax/alpha-beta (bitboard, iterative deepening, TT) and plain UCT-MCTS.

Every agent exposes `select(state) -> column` and a `name`. All randomness comes from a seeded RNG.
"""
from __future__ import annotations

import math
import random
import time

from game import State, has_four


class RandomAgent:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.name = "random"

    def select(self, state: State) -> int:
        return self.rng.choice(state.legal_moves())


class GreedyAgent:
    """Win if possible, else block an immediate loss, else the most central column (random tie-break)."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.name = "greedy"

    def select(self, state: State) -> int:
        legal = state.legal_moves()
        for c in legal:
            if state.wins_with(c):
                return c
        for c in legal:
            if state.opp_wins_with(c):
                return c
        mid = (state.cols - 1) / 2
        best = min(abs(c - mid) for c in legal)
        return self.rng.choice([c for c in legal if abs(c - mid) == best])


# --------------------------------------------------------------------------------------------------
# Negamax / alpha-beta
# --------------------------------------------------------------------------------------------------
WIN = 100_000
_EXACT, _LOWER, _UPPER = 0, 1, 2


class _Timeout(Exception):
    pass


class NegamaxAgent:
    """Fail-hard alpha-beta with iterative deepening, centre-first ordering (previous best / TT move first),
    a transposition table, and a window-count heuristic at depth 0.
    Scores: +-(WIN - ply) for wins/losses (so faster wins are preferred), 0 draw."""

    def __init__(self, depth: int = 4, time_limit: float | None = None, seed: int = 0, tt_max: int = 2_000_000):
        self.max_depth = depth
        self.time_limit = time_limit
        self.tt_max = tt_max
        self.tt: dict = {}
        self.nodes = 0
        self.root_order = None   # optional root-only move order (used by the hybrid agent)
        self.last_depth = 0
        self.last_score = 0
        self.name = f"negamax(d={depth}" + (f",t={time_limit}" if time_limit else "") + ")"
        self._deadline = None

    # ---- heuristic
    def _eval(self, cur: int, mask: int) -> int:
        g = self.g
        opp = cur ^ mask
        wt = self.wt
        s = 0
        for w in g.windows:
            a, b = w & cur, w & opp
            if not b:
                if a:
                    s += wt[a.bit_count()]
            elif not a:
                s -= wt[b.bit_count()]
        return s

    def _search(self, cur, mask, moves, depth, alpha, beta):
        g = self.g
        self.nodes += 1
        if not (self.nodes & 1023) and self._deadline is not None and time.perf_counter() > self._deadline:
            raise _Timeout
        if moves == g.size:
            return 0
        top, bottom, colmask = g.top, g.bottom, g.colmask
        moves_list = []
        for c in self.order:
            if not (mask & top[c]):
                nb = (mask + bottom[c]) & colmask[c]
                if has_four(cur | nb, g):
                    return WIN - (moves + 1)
                moves_list.append((c, nb))
        if depth <= 0:
            return self._eval(cur, mask)
        key = (mask << self.keyshift) | cur
        ent = self.tt.get(key)
        tt_move = -1
        if ent is not None:
            d, flag, val, tt_move = ent
            if d >= depth:
                if flag == _EXACT:
                    return val
                if flag == _LOWER and val > alpha:
                    alpha = val
                elif flag == _UPPER and val < beta:
                    beta = val
                if alpha >= beta:
                    return val
        if tt_move >= 0:
            moves_list.sort(key=lambda x: x[0] != tt_move)  # stable: TT move first, rest keep centre order
        a0 = alpha
        best_move = -1
        opp = cur ^ mask
        for c, nb in moves_list:
            v = -self._search(opp, mask | nb, moves + 1, depth - 1, -beta, -alpha)
            if v > alpha:
                alpha = v
                best_move = c
                if alpha >= beta:
                    break
        flag = _UPPER if alpha <= a0 else (_LOWER if alpha >= beta else _EXACT)
        if len(self.tt) < self.tt_max:
            self.tt[key] = (depth, flag, alpha, best_move if best_move >= 0 else tt_move)
        return alpha

    def select(self, state: State) -> int:
        g = self.g = state.g
        K = g.inarow
        self.wt = [0] + [4 ** (n - 1) for n in range(1, K + 1)]
        self.order = g.center_order
        self.keyshift = g.cols * g.H + 1
        self.nodes = 0
        if len(self.tt) >= self.tt_max:
            self.tt.clear()
        legal = state.legal_moves()
        if len(legal) == 1:
            return legal[0]
        self._deadline = time.perf_counter() + self.time_limit if self.time_limit else None
        best_move = [c for c in (self.root_order or self.order) if c in legal][0]
        cur, mask, moves = state.cur, state.mask, state.moves
        for depth in range(1, self.max_depth + 1):
            try:
                mv, score = self._root(cur, mask, moves, depth, best_move)
            except _Timeout:
                break
            best_move, self.last_score, self.last_depth = mv, score, depth
            if abs(score) >= WIN - g.size - 1:   # proven win/loss, deeper search cannot change it
                break
        return best_move

    def _root(self, cur, mask, moves, depth, first):
        g = self.g
        cand = []
        for c in (self.root_order or self.order):
            if not (mask & g.top[c]):
                cand.append((c, (mask + g.bottom[c]) & g.colmask[c]))
        cand.sort(key=lambda x: x[0] != first)
        alpha, best = -WIN * 2, cand[0][0]
        opp = cur ^ mask
        for c, nb in cand:
            if has_four(cur | nb, g):
                return c, WIN - (moves + 1)
            v = -self._search(opp, mask | nb, moves + 1, depth - 1, -WIN * 2, -alpha)
            if v > alpha:
                alpha, best = v, c
        return best, alpha


# --------------------------------------------------------------------------------------------------
# Plain UCT MCTS
# --------------------------------------------------------------------------------------------------
class _Node:
    __slots__ = ("state", "parent", "move", "children", "untried", "N", "W", "terminal")

    def __init__(self, state: State, parent=None, move=-1, rng=None):
        self.state, self.parent, self.move = state, parent, move
        self.children: list = []
        self.terminal = state.is_terminal()
        self.untried = [] if self.terminal else state.legal_moves()
        if rng is not None:
            rng.shuffle(self.untried)
        self.N = 0
        self.W = 0.0   # total value from the perspective of the player who moved INTO this node


class MCTSAgent:
    """UCT with random or heuristic (win / block / random) rollouts, robust-child (max visits) move choice.
    Budget: `sims` simulations and/or `time_limit` seconds (whichever hits first)."""

    def __init__(self, sims: int | None = 200, time_limit: float | None = None, c: float = 1.4,
                 rollout: str = "random", seed: int = 0):
        assert rollout in ("random", "heuristic")
        assert sims or time_limit
        self.sims, self.time_limit, self.c, self.rollout = sims, time_limit, c, rollout
        self.rng = random.Random(seed)
        self.name = f"mcts-{rollout}(" + (f"sims={sims}" if sims else "") + (f",t={time_limit}" if time_limit else "") + ")"
        self.total_sims = 0

    def _rollout(self, state: State) -> int:
        """Value (+1/0/-1) for the player to move at `state`."""
        s = state.copy()
        k = 0
        rng = self.rng
        heur = self.rollout == "heuristic"
        while not s.is_terminal():
            legal = s.legal_moves()
            mv = None
            if heur:
                for c in legal:
                    if s.wins_with(c):
                        mv = c
                        break
                if mv is None:
                    for c in legal:
                        if s.opp_wins_with(c):
                            mv = c
                            break
            if mv is None:
                mv = rng.choice(legal)
            s.play(mv)
            k += 1
        if s.won:
            return 1 if k % 2 == 1 else -1
        return 0

    def select(self, state: State) -> int:
        legal = state.legal_moves()
        if len(legal) == 1:
            return legal[0]
        root = _Node(state.copy(), rng=self.rng)
        deadline = time.perf_counter() + self.time_limit if self.time_limit else None
        n = 0
        while True:
            if self.sims and n >= self.sims:
                break
            if deadline is not None and n > 0 and time.perf_counter() >= deadline:
                break
            node = root
            while not node.untried and node.children and not node.terminal:
                lnN = math.log(node.N)
                c = self.c
                node = max(node.children, key=lambda ch: ch.W / ch.N + c * math.sqrt(lnN / ch.N))
            if node.untried:
                mv = node.untried.pop()
                node = _Node(node.state.child(mv), node, mv, self.rng)
                node.parent.children.append(node)
            if node.terminal:
                v = -1 if node.state.won else 0       # value for the player to move at the leaf
            else:
                v = self._rollout(node.state)
            while node is not None:                   # value for the mover into `node` is -v
                v = -v
                node.N += 1
                node.W += v
                node = node.parent
            n += 1
        self.total_sims += n
        return max(root.children, key=lambda ch: ch.N).move
