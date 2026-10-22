"""Bitboard ConnectX engine (any rows/columns/inarow) plus a slow reference implementation.

Bit layout: column c, row r (r=0 is the BOTTOM row) is bit  c*(rows+1) + r.  The extra top bit of each
column is a sentinel that is never set, which stops runs wrapping between columns.

State fields: `cur` = checkers of the player TO MOVE, `mask` = all checkers, `moves` = plies played,
`won` = True iff the player who just moved has completed `inarow` (so the player to move has lost).

Network encoding (`to_planes`), always from the perspective of the player to move, row 0 = TOP row
(same orientation as the Kaggle `board`):
    plane 0: own checkers (player to move)      plane 1: opponent checkers
    plane 2 (only when n_planes=3): constant 1.0 if the player to move is player 1 (moved first), else 0.0
"""
from __future__ import annotations

import numpy as np


class Geometry:
    _cache: dict = {}

    def __init__(self, rows: int, cols: int, inarow: int):
        self.rows, self.cols, self.inarow = rows, cols, inarow
        H = rows + 1
        self.H = H
        self.bottom = [1 << (c * H) for c in range(cols)]
        self.colmask = [((1 << rows) - 1) << (c * H) for c in range(cols)]
        self.top = [1 << (c * H + rows - 1) for c in range(cols)]
        self.full_mask = sum(self.colmask)
        self.shifts = (1, H, H + 1, H - 1)
        # bit shift per (top-first row, col) cell for plane extraction
        self.cell_shift = np.array([[c * H + (rows - 1 - r) for c in range(cols)] for r in range(rows)], dtype=np.int64)
        self.center_order = sorted(range(cols), key=lambda c: (abs(c - (cols - 1) / 2), c))
        self.size = rows * cols
        self.windows = self._windows()

    def _windows(self):
        R, C, K, H = self.rows, self.cols, self.inarow, self.H
        wins = []
        for c in range(C):
            for r in range(R):
                for dc, dr in ((1, 0), (0, 1), (1, 1), (1, -1)):
                    ec, er = c + dc * (K - 1), r + dr * (K - 1)
                    if 0 <= ec < C and 0 <= er < R:
                        m = 0
                        for i in range(K):
                            m |= 1 << ((c + dc * i) * H + r + dr * i)
                        wins.append(m)
        return wins

    @classmethod
    def get(cls, rows, cols, inarow):
        k = (rows, cols, inarow)
        g = cls._cache.get(k)
        if g is None:
            g = cls._cache[k] = Geometry(rows, cols, inarow)
        return g


def has_four(b: int, g: Geometry) -> bool:
    """True if bitboard b contains `inarow` consecutive bits in any direction."""
    k = g.inarow
    for d in g.shifts:
        if k == 4:
            m = b & (b >> d)
            if m & (m >> (2 * d)):
                return True
        else:
            t = b
            for i in range(1, k):
                t &= b >> (i * d)
            if t:
                return True
    return False


class State:
    __slots__ = ("g", "cur", "mask", "moves", "won")

    def __init__(self, rows: int = 6, cols: int = 7, inarow: int = 4):
        self.g = Geometry.get(rows, cols, inarow)
        self.cur = 0
        self.mask = 0
        self.moves = 0
        self.won = False

    # ---- basic properties
    @property
    def rows(self): return self.g.rows

    @property
    def cols(self): return self.g.cols

    @property
    def inarow(self): return self.g.inarow

    @property
    def current_player(self) -> int:
        return 1 if self.moves % 2 == 0 else 2

    def copy(self) -> "State":
        s = State.__new__(State)
        s.g, s.cur, s.mask, s.moves, s.won = self.g, self.cur, self.mask, self.moves, self.won
        return s

    def legal_moves(self) -> list:
        if self.won:
            return []
        m, top = self.mask, self.g.top
        return [c for c in range(self.g.cols) if not (m & top[c])]

    def is_legal(self, col: int) -> bool:
        return 0 <= col < self.g.cols and not self.won and not (self.mask & self.g.top[col])

    def play(self, col: int) -> "State":
        """Mutating move for the player to move. Raises ValueError on an illegal column."""
        if not self.is_legal(col):
            raise ValueError(f"illegal move {col}")
        g = self.g
        new_bit = (self.mask + g.bottom[col]) & g.colmask[col]
        self.cur ^= self.mask                 # swap sides: cur becomes the next player to move
        self.mask |= new_bit
        self.won = has_four(self.cur ^ self.mask, g)   # (cur^mask) = checkers of the mover
        self.moves += 1
        return self

    def child(self, col: int) -> "State":
        return self.copy().play(col)

    def is_win(self) -> bool:
        """True iff the player who just moved has won."""
        return self.won

    def is_draw(self) -> bool:
        return (not self.won) and self.moves == self.g.size

    def is_terminal(self) -> bool:
        return self.won or self.moves == self.g.size

    def wins_with(self, col: int) -> bool:
        """Would the player to move win by playing col? (col must be legal)"""
        g = self.g
        nb = (self.mask + g.bottom[col]) & g.colmask[col]
        return has_four(self.cur | nb, g)

    def opp_wins_with(self, col: int) -> bool:
        g = self.g
        nb = (self.mask + g.bottom[col]) & g.colmask[col]
        return has_four((self.cur ^ self.mask) | nb, g)

    # ---- keys / symmetry
    def to_key(self) -> int:
        return (self.mask << (self.g.cols * self.g.H + 1)) | self.cur

    def mirror(self) -> "State":
        g = self.g
        H, C = g.H, g.cols
        blk = (1 << H) - 1
        cur = mask = 0
        for c in range(C):
            nc = C - 1 - c
            cur |= ((self.cur >> (c * H)) & blk) << (nc * H)
            mask |= ((self.mask >> (c * H)) & blk) << (nc * H)
        s = self.copy()
        s.cur, s.mask = cur, mask
        return s

    # ---- encodings
    def to_planes(self, n_planes: int = 2, out: np.ndarray | None = None) -> np.ndarray:
        g = self.g
        if out is None:
            out = np.zeros((n_planes, g.rows, g.cols), dtype=np.float32)
        sh = g.cell_shift
        out[0] = (self.cur >> sh) & 1
        out[1] = ((self.cur ^ self.mask) >> sh) & 1
        if n_planes >= 3:
            out[2] = 1.0 if self.current_player == 1 else 0.0
        return out

    def to_board(self) -> list:
        """Kaggle-style flat board (top row first, 0/1/2)."""
        g = self.g
        me, other = self.current_player, 3 - self.current_player
        mine, theirs = self.cur, self.cur ^ self.mask
        out = []
        for r in range(g.rows):
            for c in range(g.cols):
                b = 1 << (c * g.H + (g.rows - 1 - r))
                out.append(me if mine & b else (other if theirs & b else 0))
        return out

    @classmethod
    def from_board(cls, board, mark: int, rows: int = 6, cols: int = 7, inarow: int = 4) -> "State":
        board = [int(x) for x in board]
        if len(board) != rows * cols:
            raise ValueError(f"board has {len(board)} cells, expected {rows}x{cols}={rows*cols}")
        s = cls(rows, cols, inarow)
        g = s.g
        p1 = p2 = 0
        for r in range(rows):
            for c in range(cols):
                v = board[r * cols + c]
                if v:
                    b = 1 << (c * g.H + (rows - 1 - r))
                    if v == 1:
                        p1 |= b
                    elif v == 2:
                        p2 |= b
                    else:
                        raise ValueError(f"bad cell value {v}")
        s.mask = p1 | p2
        s.moves = (p1.bit_count() + p2.bit_count())
        # gravity check: every column must be a contiguous stack from the bottom
        for c in range(cols):
            col = (s.mask >> (c * g.H)) & ((1 << rows) - 1)
            if col & (col + 1):
                raise ValueError(f"floating checkers in column {c}")
        if p1.bit_count() - p2.bit_count() not in (0, 1):
            raise ValueError("checker counts inconsistent with alternating play")
        if s.current_player != mark:
            raise ValueError(f"mark={mark} does not match side to move ({s.current_player}) for this board")
        s.cur = p1 if mark == 1 else p2
        s.won = has_four(s.cur ^ s.mask, g)
        return s

    @classmethod
    def from_kaggle_observation(cls, observation, configuration=None) -> "State":
        def get(o, k, d=None):
            if o is None:
                return d
            if isinstance(o, dict):
                return o.get(k, d)
            return getattr(o, k, d)
        rows = int(get(configuration, "rows", 6) or 6)
        cols = int(get(configuration, "columns", 7) or 7)
        inarow = int(get(configuration, "inarow", 4) or 4)
        return cls.from_board(get(observation, "board"), int(get(observation, "mark")), rows, cols, inarow)

    def __repr__(self):
        b = self.to_board()
        sym = {0: ".", 1: "X", 2: "O"}
        return "\n".join("".join(sym[b[r * self.cols + c]] for c in range(self.cols)) for r in range(self.rows))


def random_opening(rng, plies, rows=6, cols=7, inarow=4):
    """`plies` random legal moves that leave a non-terminal position (list of columns)."""
    while True:
        s, moves = State(rows, cols, inarow), []
        for _ in range(plies):
            lm = s.legal_moves()
            m = int(lm[int(rng.integers(len(lm)))])
            s.play(m)
            moves.append(m)
        if not s.is_terminal():
            return moves


def random_state(rng, rows=6, cols=7, inarow=4, max_plies=None, stop_at_terminal=False) -> State:
    """Random reachable, non-terminal position by playing random legal moves."""
    while True:
        s = State(rows, cols, inarow)
        n = int(rng.integers(0, (max_plies or rows * cols) + 1))
        ok = True
        for _ in range(n):
            lm = s.legal_moves()
            if not lm:
                break
            s.play(int(lm[int(rng.integers(len(lm)))]))
            if s.is_terminal():
                ok = False
                break
        if ok and not s.is_terminal():
            return s


# --------------------------------------------------------------------------------------------------
# Slow, obviously-correct reference implementation (for cross-checking only)
# --------------------------------------------------------------------------------------------------
class ReferenceGame:
    """2-D list board, row 0 = top. Straightforward scanning rules, no bit tricks."""

    def __init__(self, rows=6, cols=7, inarow=4):
        self.rows, self.cols, self.inarow = rows, cols, inarow
        self.grid = [[0] * cols for _ in range(rows)]
        self.player = 1
        self.winner = 0

    def legal_moves(self):
        return [] if self.winner else [c for c in range(self.cols) if self.grid[0][c] == 0]

    def play(self, col):
        assert self.grid[0][col] == 0 and not self.winner
        r = max(r for r in range(self.rows) if self.grid[r][col] == 0)
        self.grid[r][col] = self.player
        if self.has_line(self.player):
            self.winner = self.player
        self.player = 3 - self.player

    def has_line(self, p):
        R, C, K = self.rows, self.cols, self.inarow
        for r in range(R):
            for c in range(C):
                for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                    if all(0 <= r + dr * i < R and 0 <= c + dc * i < C and self.grid[r + dr * i][c + dc * i] == p
                           for i in range(K)):
                        return True
        return False

    def is_draw(self):
        return not self.winner and all(self.grid[0][c] != 0 for c in range(self.cols))

    def flat_board(self):
        return [x for row in self.grid for x in row]
