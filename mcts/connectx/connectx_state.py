"""Fast ConnectX game-state class, with an independent reference
implementation used to cross-check it.

Board orientation (confirmed in connectx/ENV_NOTES.md from the local
kaggle_environments source): a flat list of length rows*columns, row-major
from the TOP row (index = row*columns + col). 0 = empty, 1 = player 1's
checker, 2 = player 2's checker. Player 1 moves first.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

EMPTY = 0


@dataclass
class ConnectXState:
    """Mutable-but-copyable ConnectX game state.

    `board` is a flat tuple of length rows*columns (row-major, top row
    first). `mark` is the player to move next (1 or 2).
    """

    rows: int = 6
    columns: int = 7
    inarow: int = 4
    board: Tuple[int, ...] = field(default_factory=tuple)
    mark: int = 1  # player to move
    last_action: Optional[int] = None
    last_mark: Optional[int] = None

    def __post_init__(self):
        if not self.board:
            self.board = tuple([EMPTY] * (self.rows * self.columns))

    # -- interface expected by mcts_core -----------------------------------
    def legal_actions(self) -> List[int]:
        return [c for c in range(self.columns) if self.board[c] == EMPTY]

    def is_terminal(self) -> bool:
        if self.last_action is not None and self._is_win(self.last_action, self.last_mark):
            return True
        return all(v != EMPTY for v in self.board)

    def winner(self) -> Optional[int]:
        """Returns the winning mark, or None if no winner (yet, or a draw)."""
        if self.last_action is not None and self._is_win(self.last_action, self.last_mark):
            return self.last_mark
        return None

    def current_player(self) -> int:
        return self.mark

    def reward(self, perspective_mark: int) -> float:
        """Terminal reward from `perspective_mark`'s point of view: +1 win,
        -1 loss, 0 draw/non-terminal."""
        w = self.winner()
        if w is None:
            return 0.0
        return 1.0 if w == perspective_mark else -1.0

    def apply(self, action: int) -> "ConnectXState":
        """Returns a NEW state with `action` applied by the player to move."""
        if self.board[action] != EMPTY:
            raise ValueError(f"Illegal move: column {action} is full")
        row = self._landing_row(action)
        board = list(self.board)
        board[action + row * self.columns] = self.mark
        next_mark = 2 if self.mark == 1 else 1
        return ConnectXState(
            rows=self.rows,
            columns=self.columns,
            inarow=self.inarow,
            board=tuple(board),
            mark=next_mark,
            last_action=action,
            last_mark=self.mark,
        )

    def copy(self) -> "ConnectXState":
        return ConnectXState(
            rows=self.rows, columns=self.columns, inarow=self.inarow,
            board=self.board, mark=self.mark,
            last_action=self.last_action, last_mark=self.last_mark,
        )

    # -- helpers -------------------------------------------------------------
    def _landing_row(self, column: int) -> int:
        """Row index (0=top) where a piece dropped in `column` lands."""
        candidates = [r for r in range(self.rows) if self.board[column + r * self.columns] == EMPTY]
        if not candidates:
            raise ValueError(f"Column {column} is full")
        return max(candidates)  # lowest empty cell (closest to the bottom)

    def _is_win(self, column: int, mark: int) -> bool:
        """Fast win-check anchored at the last-played cell (any inarow)."""
        row = min(r for r in range(self.rows) if self.board[column + r * self.columns] == mark)
        n = self.inarow - 1

        def count(dr: int, dc: int) -> int:
            cnt = 0
            for i in range(1, n + 1):
                r, c = row + dr * i, column + dc * i
                if r < 0 or r >= self.rows or c < 0 or c >= self.columns:
                    break
                if self.board[c + r * self.columns] != mark:
                    break
                cnt += 1
            return cnt

        return (
            count(1, 0) >= n
            or (count(0, 1) + count(0, -1)) >= n
            or (count(-1, -1) + count(1, 1)) >= n
            or (count(-1, 1) + count(1, -1)) >= n
        )

    def render(self) -> str:
        rows_str = []
        sep = "+" + "+".join(["---"] * self.columns) + "+"
        rows_str.append(sep)
        for r in range(self.rows):
            cells = self.board[r * self.columns:(r + 1) * self.columns]
            rows_str.append("| " + " | ".join(str(v) for v in cells) + " |")
            rows_str.append(sep)
        return "\n".join(rows_str)

    # -- Kaggle interop -------------------------------------------------------
    @classmethod
    def from_observation(cls, observation, configuration) -> "ConnectXState":
        """Builds a state from a Kaggle-style observation/configuration pair.

        `observation` must have `.board` (flat list) and `.mark` (int); may
        be a dict or an attribute-style object (both are supported).
        `configuration` must have `.rows`, `.columns`, `.inarow`.
        """
        def g(obj, key, default=None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        rows = g(configuration, "rows", 6)
        columns = g(configuration, "columns", 7)
        inarow = g(configuration, "inarow", 4)
        board = tuple(g(observation, "board"))
        mark = g(observation, "mark")

        n1 = sum(1 for v in board if v == 1)
        n2 = sum(1 for v in board if v == 2)
        # Player 1 moves first, so counts are equal (mark 1 to move) or
        # mark-1 has one more piece than mark-2 (mark 2 to move).
        inferred_to_move = 1 if n1 == n2 else 2
        if mark is not None and mark != inferred_to_move:
            # Trust the piece-count inference; `mark` identifies the agent,
            # not necessarily who moves next in this constructed board.
            pass

        return cls(rows=rows, columns=columns, inarow=inarow, board=board, mark=inferred_to_move)


def reference_is_win(board: Tuple[int, ...], rows: int, columns: int, inarow: int, mark: int) -> bool:
    """Slow, independent reference win-checker: scans the ENTIRE board for
    `inarow` consecutive `mark` values in any of the 4 directions. Used only
    in tests to cross-check `ConnectXState._is_win`."""
    def get(r, c):
        return board[c + r * columns]

    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for r in range(rows):
        for c in range(columns):
            if get(r, c) != mark:
                continue
            for dr, dc in directions:
                ok = True
                for i in range(inarow):
                    rr, cc = r + dr * i, c + dc * i
                    if rr < 0 or rr >= rows or cc < 0 or cc >= columns or get(rr, cc) != mark:
                        ok = False
                        break
                if ok:
                    return True
    return False
