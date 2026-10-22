import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from connectx_state import ConnectXState
from agents import (
    random_agent, greedy_agent, negamax_agent, mcts_agent,
    greedy_policy, make_negamax,
)


def test_random_agent_returns_legal_move():
    obs = {"board": [0] * 42, "mark": 1}
    config = {"rows": 6, "columns": 7, "inarow": 4}
    a = random_agent(obs, config)
    assert 0 <= a < 7


def test_greedy_takes_win():
    rows, columns, inarow = 6, 7, 4
    board = [0] * (rows * columns)
    for c in range(3):
        board[c + 5 * columns] = 1
    s = ConnectXState(rows=rows, columns=columns, inarow=inarow, board=tuple(board), mark=1)
    assert greedy_policy(s, random.Random(0)) == 3


def test_greedy_blocks():
    rows, columns, inarow = 6, 7, 4
    board = [0] * (rows * columns)
    for c in range(3):
        board[c + 5 * columns] = 2
    s = ConnectXState(rows=rows, columns=columns, inarow=inarow, board=tuple(board), mark=1)
    assert greedy_policy(s, random.Random(0)) == 3


def test_negamax_depth2_returns_legal():
    s = ConnectXState(rows=6, columns=7, inarow=4, mark=1)
    policy = make_negamax(2)
    a = policy(s, random.Random(0))
    assert a in s.legal_actions()


def test_mcts_kaggle_wrapper_legal():
    agent = mcts_agent(n_sims=30, seed=0)
    obs = {"board": [0] * 42, "mark": 1}
    config = {"rows": 6, "columns": 7, "inarow": 4}
    a = agent(obs, config)
    assert 0 <= a < 7


def test_kaggle_runtime_cross_check():
    ke = __import__("pytest").importorskip("kaggle_environments")
    # Only reached if kaggle_environments is actually importable; otherwise
    # skipped. See connectx/ENV_NOTES.md: the runtime is installed and this
    # now runs for real.
    env = ke.make("connectx", debug=True)
    env.reset()
    assert env.configuration.rows == 6
    assert env.configuration.columns == 7
    assert env.configuration.inarow == 4
    assert env.configuration.timeout == 2
    assert env.configuration.episodeSteps == 1000


def test_from_observation_matches_real_runtime_random_games():
    """Spec item 8: our state class matches the real kaggle_environments
    board after every move, across several random games.

    Quirk of this package version (confirmed live): only state[0] (mark=1)
    ever carries a populated `board`/`step` in its observation dict; state[1]
    (mark=2) observation never includes `board`, even while ACTIVE. We pull
    the board from state[0] (always present) and the acting `mark` from
    whichever index has status == 'ACTIVE'."""
    ke = __import__("pytest").importorskip("kaggle_environments")
    config = {"rows": 6, "columns": 7, "inarow": 4}
    for seed in range(5):
        rng = random.Random(seed)
        env = ke.make("connectx", configuration=config, debug=True)
        env.reset()
        while not env.done:
            board = env.state[0]["observation"]["board"]
            active_idx = next(i for i in range(2) if env.state[i]["status"] == "ACTIVE")
            active_mark = active_idx + 1
            obs = {"board": board, "mark": active_mark}
            our_state = ConnectXState.from_observation(obs, env.configuration)
            assert list(our_state.board) == board
            assert our_state.mark == active_mark
            legal = our_state.legal_actions()
            assert len(legal) > 0
            action = rng.choice(legal)
            actions = [None, None]
            actions[active_idx] = action
            env.step(actions)
        assert env.done
