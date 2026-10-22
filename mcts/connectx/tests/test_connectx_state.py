import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from connectx_state import ConnectXState, reference_is_win


def make_state(rows=6, columns=7, inarow=4, board=None, mark=1):
    if board is None:
        board = tuple([0] * (rows * columns))
    return ConnectXState(rows=rows, columns=columns, inarow=inarow, board=tuple(board), mark=mark)


def test_legal_actions_full_board_empty():
    s = make_state()
    assert s.legal_actions() == list(range(7))


def test_apply_lands_at_bottom():
    s = make_state()
    s2 = s.apply(3)
    # bottom row is row rows-1 = 5; index = col + row*columns = 3 + 5*7 = 38
    assert s2.board[3 + 5 * 7] == 1
    assert s2.mark == 2


def test_apply_stacks_pieces():
    s = make_state()
    s = s.apply(0)  # player1 lands row 5
    s = s.apply(0)  # player2 lands row 4
    assert s.board[0 + 5 * 7] == 1
    assert s.board[0 + 4 * 7] == 2


def test_win_detection_matches_reference_random_positions():
    # Simulate real games with random legal moves (so any win, if present,
    # necessarily involves the just-played piece -- matching how `_is_win`
    # is actually used: checked immediately after a move in that column).
    rng = random.Random(42)
    rows, columns, inarow = 6, 7, 4
    for game in range(200):
        s = make_state(rows, columns, inarow)
        for _ply in range(rows * columns):
            if s.is_terminal():
                break
            actions = s.legal_actions()
            a = rng.choice(actions)
            mover = s.mark
            s = s.apply(a)
            fast = s._is_win(a, mover)
            ref = reference_is_win(s.board, rows, columns, inarow, mover)
            assert fast == ref, (s.board, a, mover)
            if fast:
                break


def test_horizontal_win():
    rows, columns, inarow = 6, 7, 4
    board = [0] * (rows * columns)
    for c in range(4):
        board[c + 5 * columns] = 1
    s = make_state(rows, columns, inarow, board)
    assert s._is_win(3, 1)
    assert reference_is_win(tuple(board), rows, columns, inarow, 1)


def test_vertical_win():
    rows, columns, inarow = 6, 7, 4
    board = [0] * (rows * columns)
    for r in range(2, 6):
        board[0 + r * columns] = 2
    s = make_state(rows, columns, inarow, board)
    assert s._is_win(0, 2)


def test_diagonal_wins_both_directions():
    rows, columns, inarow = 6, 7, 4
    # "/" diagonal (bottom-left to top-right)
    board = [0] * (rows * columns)
    for i, (r, c) in enumerate([(5, 0), (4, 1), (3, 2), (2, 3)]):
        board[c + r * columns] = 1
    s = make_state(rows, columns, inarow, board)
    assert s._is_win(3, 1)

    # "\" diagonal (top-left to bottom-right)
    board2 = [0] * (rows * columns)
    for (r, c) in [(2, 0), (3, 1), (4, 2), (5, 3)]:
        board2[c + r * columns] = 2
    s2 = make_state(rows, columns, inarow, board2)
    assert s2._is_win(3, 2)


def test_edge_of_board_no_false_positive():
    rows, columns, inarow = 6, 7, 4
    board = [0] * (rows * columns)
    board[0 + 5 * columns] = 1
    board[1 + 5 * columns] = 1
    board[2 + 5 * columns] = 1
    # no 4th piece -> should not report a win
    s = make_state(rows, columns, inarow, board)
    assert not reference_is_win(tuple(board), rows, columns, inarow, 1)


def test_from_observation_roundtrip():
    rows, columns, inarow = 6, 7, 4
    board = [0] * (rows * columns)
    board[3 + 5 * columns] = 1
    obs = {"board": board, "mark": 2}
    config = {"rows": rows, "columns": columns, "inarow": inarow}
    s = ConnectXState.from_observation(obs, config)
    assert s.board == tuple(board)
    assert s.mark == 2  # one piece down for mark 1 -> mark 2 to move


def test_from_observation_equal_pieces_player1_to_move():
    rows, columns, inarow = 6, 7, 4
    board = [0] * (rows * columns)
    board[3 + 5 * columns] = 1
    board[4 + 5 * columns] = 2
    obs = {"board": board, "mark": 1}
    config = {"rows": rows, "columns": columns, "inarow": inarow}
    s = ConnectXState.from_observation(obs, config)
    assert s.mark == 1
