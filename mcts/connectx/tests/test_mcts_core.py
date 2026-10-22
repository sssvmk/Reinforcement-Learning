import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from connectx_state import ConnectXState
from mcts_core import MCTSConfig, run_mcts
from agents import make_mcts_agent, random_policy


def empty_state(mark=1):
    return ConnectXState(rows=6, columns=7, inarow=4, mark=mark)


def test_mcts_returns_legal_move():
    s = empty_state()
    cfg = MCTSConfig(n_simulations=100, rng=random.Random(0))
    result = run_mcts(s, cfg)
    assert result["action"] in s.legal_actions()


def test_mcts_single_legal_column():
    rows, columns, inarow = 6, 7, 4
    board = [1 if c != 3 else 0 for c in range(columns)] * rows
    # ensure columns other than 3 are full (top row occupied)
    board = [0] * (rows * columns)
    for c in range(columns):
        if c == 3:
            continue
        for r in range(rows):
            board[c + r * columns] = 1
    s = ConnectXState(rows=rows, columns=columns, inarow=inarow, board=tuple(board), mark=2)
    cfg = MCTSConfig(n_simulations=50, rng=random.Random(1))
    result = run_mcts(s, cfg)
    assert result["action"] == 3


def _fill_except(board, rows, columns, open_columns):
    """Fully fills every column not in `open_columns` with alternating
    filler marks (irrelevant to any win), so only `open_columns` remain
    legal moves -- isolates the decision to a small, controlled action set."""
    for c in range(columns):
        if c in open_columns:
            continue
        for r in range(rows):
            board[c + r * columns] = 1 if (r + c) % 2 == 0 else 2


def test_mcts_takes_immediate_win():
    rows, columns, inarow = 6, 7, 4
    board = [0] * (rows * columns)
    for c in range(3):
        board[c + 5 * columns] = 1
    _fill_except(board, rows, columns, open_columns={3, 4})
    s = ConnectXState(rows=rows, columns=columns, inarow=inarow, board=tuple(board), mark=1)
    cfg = MCTSConfig(n_simulations=300, c=1.41, rollout="heuristic", rng=random.Random(2))
    result = run_mcts(s, cfg)
    assert result["action"] == 3


def test_mcts_blocks_immediate_loss():
    rows, columns, inarow = 6, 7, 4
    board = [0] * (rows * columns)
    for c in range(3):
        board[c + 5 * columns] = 2  # opponent 3 in a row, about to win at col 3
    _fill_except(board, rows, columns, open_columns={3, 4})
    s = ConnectXState(rows=rows, columns=columns, inarow=inarow, board=tuple(board), mark=1)
    cfg = MCTSConfig(n_simulations=300, c=1.41, rollout="heuristic", rng=random.Random(3))
    result = run_mcts(s, cfg)
    assert result["action"] == 3


def test_value_sign_flip_forced_win_scores_high():
    rows, columns, inarow = 6, 7, 4
    board = [0] * (rows * columns)
    for c in range(3):
        board[c + 5 * columns] = 1
    s = ConnectXState(rows=rows, columns=columns, inarow=inarow, board=tuple(board), mark=1)
    cfg = MCTSConfig(n_simulations=300, rollout="heuristic", rng=random.Random(4))
    result = run_mcts(s, cfg)
    assert result["root_value"] > 0.5


def test_same_seed_same_move():
    s = empty_state()
    cfg1 = MCTSConfig(n_simulations=150, rng=random.Random(123))
    cfg2 = MCTSConfig(n_simulations=150, rng=random.Random(123))
    r1 = run_mcts(s, cfg1)
    r2 = run_mcts(s, cfg2)
    assert r1["action"] == r2["action"]


def test_mcts_beats_random_smoke(capsys=None):
    rng = random.Random(7)
    mcts_policy = make_mcts_agent(n_sims=200, seed=7)
    wins = 0
    n_games = 30
    for g in range(n_games):
        s = ConnectXState(rows=6, columns=7, inarow=4, mark=1)
        mcts_is_p1 = g % 2 == 0
        while not s.is_terminal():
            if (s.current_player() == 1) == mcts_is_p1:
                a = mcts_policy(s, random.Random(1000 + g))
            else:
                a = random_policy(s, rng)
            s = s.apply(a)
        mcts_mark = 1 if mcts_is_p1 else 2
        r = s.reward(mcts_mark)
        if r > 0:
            wins += 1
        elif r == 0 and s.winner() is None:
            wins += 0.5
    win_rate = wins / n_games
    assert win_rate >= 0.8  # smoke test at reduced n_games; full suite uses 95%/200 games
