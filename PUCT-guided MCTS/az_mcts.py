"""PUCT-guided MCTS (AlphaZero style). Torch-free: it talks to any evaluator with

    evaluator.evaluate(planes[B,P,R,C] float32, legal[B,C] bool) -> (priors[B,C] float, values[B] float)

where priors are a softmax over LEGAL columns only (exactly 0 elsewhere) and values are from the perspective
of the player to move (+1 win, -1 loss). The same file is embedded in the NumPy-only submission.

Formulation
    select  a* = argmax_a  Q(a) + c_puct * P(a) * sqrt(max(1, sum_b N(b))) / (1 + N(a)),   Q = W/N (0 if N=0)
    leaf    network value (no rollouts); terminal leaf: previous mover has won -> -1 for the player to move
    backup  value flips sign every ply (zero-sum)
"""
from __future__ import annotations

import math
import time

import numpy as np

from game import State


class BaseEvaluator:
    n_planes = 2

    def __init__(self):
        self.n_evals = 0
        self.n_calls = 0
        self.eval_seconds = 0.0

    def _forward(self, planes, legal):
        raise NotImplementedError

    def evaluate(self, planes, legal):
        t = time.perf_counter()
        out = self._forward(planes, legal)
        self.eval_seconds += time.perf_counter() - t
        self.n_evals += len(planes)
        self.n_calls += 1
        return out


class UniformEvaluator(BaseEvaluator):
    """Uniform priors over legal moves, value 0 (an 'untrained, uninformative' evaluator for tests)."""

    def _forward(self, planes, legal):
        p = legal.astype(np.float64)
        return p / p.sum(axis=1, keepdims=True), np.zeros(len(planes))


class Node:
    __slots__ = ("P", "N", "W", "children", "legal", "total")

    def __init__(self, priors, legal):
        n = len(priors)
        self.P = priors
        self.legal = legal
        self.N = np.zeros(n, dtype=np.float64)
        self.W = np.zeros(n, dtype=np.float64)
        self.children = [None] * n
        self.total = 0


class Search:
    """One MCTS tree rooted at `root_state`. Step-wise API so many searches can share batched NN calls:
        req = search.begin_sim()     # None when the simulation resolved itself (terminal / cache hit)
        search.finish_sim(priors, v) # when begin_sim returned a state to evaluate
    """

    def __init__(self, root_state: State, c_puct=1.5, cache=None, noise_rng=None, alpha=1.0, eps=0.25,
                 root: Node | None = None, cache_max=200_000):
        self.root_state = root_state
        self.c = c_puct
        self.cache = cache if cache is not None else {}
        self.cache_max = cache_max
        self.noise_rng, self.alpha, self.eps = noise_rng, alpha, eps   # noise only if noise_rng is not None
        self.root = None
        self.sims = 0
        self._pending = None
        self.root_P_clean = None
        if root is not None:
            self.root = root
            self._apply_noise()

    # ---- root noise (self-play only)
    def _apply_noise(self):
        if self.noise_rng is None:
            return
        r = self.root
        self.root_P_clean = r.P
        idx = np.flatnonzero(r.legal)
        noise = self.noise_rng.dirichlet([self.alpha] * len(idx))
        P = r.P.copy()
        P[idx] = (1 - self.eps) * r.P[idx] + self.eps * noise
        r.P = P

    def _select(self, node: Node) -> int:
        sq = math.sqrt(max(1, node.total))
        q = node.W / np.maximum(node.N, 1)
        score = q + self.c * node.P * sq / (1.0 + node.N)
        score[~node.legal] = -np.inf
        return int(np.argmax(score))

    def _store(self, key, entry):
        if len(self.cache) >= self.cache_max:
            self.cache.clear()
        self.cache[key] = entry

    def _backup(self, path, v):
        for node, a in reversed(path):
            v = -v
            node.N[a] += 1
            node.W[a] += v
            node.total += 1

    def begin_sim(self):
        if self.root is None:
            st = self.root_state
            key = st.to_key()
            hit = self.cache.get(key)
            if hit is not None:
                self.root = Node(hit[0], hit[2])
                self._apply_noise()
                return None
            self._pending = (None, st, key)
            return st
        node = self.root
        state = self.root_state.copy()
        path = []
        size = state.g.size
        while True:
            a = self._select(node)
            path.append((node, a))
            state.play(a)
            if state.won:
                v = -1.0
                break
            if state.moves == size:
                v = 0.0
                break
            child = node.children[a]
            if child is None:
                key = state.to_key()
                hit = self.cache.get(key)
                if hit is not None:
                    node.children[a] = Node(hit[0], hit[2])
                    v = hit[1]
                    break
                self._pending = (path, state, key)
                return state
            node = child
        self._backup(path, v)
        self.sims += 1
        return None

    def finish_sim(self, priors, value):
        path, state, key = self._pending
        self._pending = None
        legal = np.zeros(len(priors), dtype=bool)
        legal[state.legal_moves()] = True
        priors = np.asarray(priors, dtype=np.float64)
        value = float(value)
        self._store(key, (priors, value, legal))
        if path is None:
            self.root = Node(priors, legal)
            self._apply_noise()
            return
        node, a = path[-1]
        node.children[a] = Node(priors, legal)
        self._backup(path, value)
        self.sims += 1

    # ---- results
    def visits(self):
        return self.root.N.copy()

    def root_value(self):
        """Mean value of the root for the player to move."""
        t = self.root.N.sum()
        return float(self.root.W.sum() / t) if t > 0 else 0.0

    def best_move(self) -> int:
        r = self.root
        if r.N.sum() == 0:                                   # policy-only
            return int(np.argmax(np.where(r.legal, r.P, -1.0)))
        top = r.N.max()
        cand = np.flatnonzero(r.N == top)
        q = r.W[cand] / np.maximum(r.N[cand], 1)
        return int(cand[int(np.argmax(q))])

    def sample_move(self, rng, temperature=1.0) -> int:
        n = self.root.N
        if n.sum() == 0:
            p = np.where(self.root.legal, self.root.P, 0.0)
        else:
            p = n ** (1.0 / temperature)
        p = p / p.sum()
        return int(rng.choice(len(p), p=p))

    def policy_target(self):
        n = self.root.N
        if n.sum() == 0:
            p = np.where(self.root.legal, self.root.P, 0.0)
            return p / p.sum()
        return n / n.sum()

    def subtree(self, *actions):
        """Return the Node reached by playing `actions` from the root, or None if not expanded."""
        node = self.root
        for a in actions:
            if node is None:
                return None
            node = node.children[a]
        return node


def _eval_one(evaluator, state: State):
    planes = state.to_planes(getattr(evaluator, "n_planes", 2))[None]
    legal = np.zeros((1, state.cols), dtype=bool)
    legal[0, state.legal_moves()] = True
    pri, val = evaluator.evaluate(planes, legal)
    return pri[0], val[0]


class AZPlayer:
    """Sequential (one game at a time) PUCT agent with a simulation and/or wall-clock budget.
    sims=0 -> policy only (argmax of the network prior). Noise/temperature are OFF by default (evaluation mode)."""

    def __init__(self, evaluator, sims: int | None = 200, time_limit: float | None = None, c_puct: float = 1.5,
                 noise: bool = False, alpha: float = 1.0, eps: float = 0.25, temp_moves: int = 0,
                 reuse: bool = False, seed: int = 0, cache_max: int = 200_000, max_sims: int | None = None):
        self.ev, self.sims, self.time_limit, self.c = evaluator, sims, time_limit, c_puct
        self.noise, self.alpha, self.eps, self.temp_moves, self.reuse = noise, alpha, eps, temp_moves, reuse
        self.rng = np.random.default_rng(seed)
        self.cache: dict = {}
        self.cache_max = cache_max
        self.max_sims = max_sims
        self.last_search: Search | None = None
        self.last_sims = 0
        self.name = f"az(sims={sims}" + (f",t={time_limit}" if time_limit else "") + f",c={c_puct})"
        self.total_sims = 0

    def new_game(self):
        self.last_search = None
        self.cache.clear()

    def _reuse_root(self, state: State):
        """Find `state` among descendants (<=2 plies) of the previous search tree."""
        ls = self.last_search
        if not (self.reuse and ls is not None and ls.root is not None):
            return None
        key = state.to_key()
        base = ls.root_state
        for a in base.legal_moves():
            s1 = base.child(a)
            if s1.to_key() == key:
                return ls.subtree(a)
            if s1.is_terminal():
                continue
            for b in s1.legal_moves():
                if s1.child(b).to_key() == key:
                    return ls.subtree(a, b)
        return None

    def search(self, state: State) -> Search:
        root = self._reuse_root(state)
        s = Search(state.copy(), self.c, self.cache, self.rng if self.noise else None, self.alpha, self.eps,
                   root=root, cache_max=self.cache_max)
        deadline = time.perf_counter() + self.time_limit if self.time_limit else None
        target = self.sims if self.sims is not None else (self.max_sims or 10 ** 9)
        if self.max_sims is not None:
            target = min(target, self.max_sims)
        while s.root is None or s.sims < target:
            if s.root is not None and deadline is not None and s.sims >= 1 and time.perf_counter() >= deadline:
                break
            req = s.begin_sim()
            if req is not None:
                p, v = _eval_one(self.ev, req)
                s.finish_sim(p, v)
        self.last_search = s
        self.last_sims = s.sims
        self.total_sims += s.sims
        return s

    def select(self, state: State) -> int:
        legal = state.legal_moves()
        if len(legal) == 1:
            self.last_search = None
            return legal[0]
        s = self.search(state)
        if self.temp_moves and state.moves < self.temp_moves:
            return s.sample_move(self.rng, 1.0)
        return s.best_move()

    __call__ = select


# --------------------------------------------------------------------------------------------------
# Batched multi-game driver (self-play and arena)
# --------------------------------------------------------------------------------------------------
class PlayerSpec:
    def __init__(self, evaluator, sims=100, c_puct=1.5, noise=False, alpha=1.0, eps=0.25, temp_moves=0,
                 reuse=False, cache_max=200_000):
        self.ev, self.sims, self.c, self.noise = evaluator, sims, c_puct, noise
        self.alpha, self.eps, self.temp_moves, self.reuse, self.cache_max = alpha, eps, temp_moves, reuse, cache_max


class GameResult:
    def __init__(self):
        self.moves = []          # all plies incl. opening
        self.winner = 0          # 0 draw, 1, 2
        self.planes = []         # recorded positions (to-move perspective)
        self.pis = []
        self.movers = []         # player number (1/2) to move at each recorded position
        self.sims = 0

    def values(self):
        """Value target per recorded position, from the perspective of the player to move there."""
        if self.winner == 0:
            return [0.0] * len(self.movers)
        return [1.0 if m == self.winner else -1.0 for m in self.movers]


class _Game:
    def __init__(self, players, seed, opening, rows, cols, inarow, record):
        self.players = players          # (spec for player 1, spec for player 2)
        self.rng = np.random.default_rng(seed)
        self.state = State(rows, cols, inarow)
        self.res = GameResult()
        self.record = record
        self.search = None
        self.caches = {}
        self.prev = [None, None]
        self.done = False
        for a in opening:
            self._play(a)

    def _play(self, a):
        mover = self.state.current_player
        self.state.play(a)
        self.res.moves.append(int(a))
        if self.state.won:
            self.res.winner, self.done = mover, True
        elif self.state.moves == self.state.g.size:
            self.done = True

    def _start(self):
        idx = self.state.current_player - 1
        spec = self.players[idx]
        cache = self.caches.setdefault(id(spec.ev), {})
        root = self._reuse(self.prev[idx]) if spec.reuse else None
        self.search = Search(self.state.copy(), spec.c, cache, self.rng if spec.noise else None, spec.alpha,
                             spec.eps, root=root, cache_max=spec.cache_max)
        self.prev[idx] = self.search

    def _reuse(self, prev):
        """Re-root this player's previous tree at the current position (found <=2 plies below its root)."""
        if prev is None or prev.root is None:
            return None
        key = self.state.to_key()
        base = prev.root_state
        for a in base.legal_moves():
            s1 = base.child(a)
            if s1.to_key() == key:
                return prev.subtree(a)
            if s1.is_terminal():
                continue
            for b in s1.legal_moves():
                if s1.child(b).to_key() == key:
                    return prev.subtree(a, b)
        return None

    def advance(self):
        """Run until a network evaluation is needed (returns (evaluator, n_planes, state)) or the game ends."""
        while not self.done:
            legal = self.state.legal_moves()
            if len(legal) == 1:                     # forced move, nothing to learn / search
                self.search = None
                self._play(legal[0])
                continue
            if self.search is None or self.search.root_state.moves != self.state.moves:
                self._start()
            s = self.search
            spec = self.players[self.state.current_player - 1]
            while s.root is None or s.sims < spec.sims:
                req = s.begin_sim()
                if req is not None:
                    return spec.ev, req
            self._finish_move(s, spec)
        return None

    def _finish_move(self, s, spec):
        st = self.state
        if self.record:
            self.res.planes.append(st.to_planes(getattr(spec.ev, "n_planes", 2)))
            self.res.pis.append(s.policy_target().astype(np.float32))
            self.res.movers.append(st.current_player)
        self.res.sims += s.sims
        a = s.sample_move(self.rng, 1.0) if (spec.temp_moves and st.moves < spec.temp_moves) else s.best_move()
        self._play(a)


def run_games_batched(pairs, seeds, openings=None, rows=6, cols=7, inarow=4, record=True):
    """Play len(pairs) games in lockstep. pairs[i] = (PlayerSpec for player 1, PlayerSpec for player 2).
    Each round every unfinished game contributes at most one leaf, and leaves are grouped per evaluator into one
    batched forward pass. Returns a list of GameResult."""
    openings = openings or [[] for _ in pairs]
    games = [_Game(p, s, o, rows, cols, inarow, record) for p, s, o in zip(pairs, seeds, openings)]
    active = [g for g in games if not g.done]
    while active:
        groups: dict = {}
        nxt = []
        for g in active:
            req = g.advance()
            if req is None:
                continue
            ev, st = req
            groups.setdefault(id(ev), (ev, []))[1].append((g, st))
            nxt.append(g)
        for ev, items in groups.values():
            npl = getattr(ev, "n_planes", 2)
            planes = np.zeros((len(items), npl, items[0][1].rows, items[0][1].cols), dtype=np.float32)
            legal = np.zeros((len(items), items[0][1].cols), dtype=bool)
            for i, (g, st) in enumerate(items):
                st.to_planes(npl, out=planes[i])
                legal[i, st.legal_moves()] = True
            pri, val = ev.evaluate(planes, legal)
            for i, (g, st) in enumerate(items):
                g.search.finish_sim(pri[i], val[i])
        active = nxt
    return [g.res for g in games]
