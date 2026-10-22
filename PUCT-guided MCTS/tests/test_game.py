import numpy as np
import pytest

from baselines import GreedyAgent, MCTSAgent, NegamaxAgent, RandomAgent
from game import ReferenceGame, State, random_opening, random_state


def build(moves, rows=6, cols=7, inarow=4):
    s = State(rows, cols, inarow)
    for m in moves:
        s.play(m)
    return s


# ---------------------------------------------------------------- 1. win detection
@pytest.mark.parametrize("dims", [(6, 7, 4), (4, 5, 3), (5, 5, 4), (7, 6, 4)])
def test_random_games_match_reference(dims):
    rng = np.random.default_rng(0)
    R, C, K = dims
    for _ in range(300):
        s, ref = State(R, C, K), ReferenceGame(R, C, K)
        while True:
            lm = s.legal_moves()
            assert lm == ref.legal_moves()
            if not lm:
                break
            m = int(rng.choice(lm))
            s.play(m)
            ref.play(m)
            assert s.is_win() == (ref.winner != 0)
            assert s.to_board() == ref.flat_board()
            if s.is_win() or ref.is_draw():
                break
        assert s.is_draw() == ref.is_draw()


def test_random_positions_roundtrip_through_kaggle_observation():
    rng = np.random.default_rng(1)
    for _ in range(200):
        s = random_state(rng)
        t = State.from_kaggle_observation({"board": s.to_board(), "mark": s.current_player}, {"rows": 6, "columns": 7, "inarow": 4})
        assert (t.cur, t.mask, t.moves) == (s.cur, s.mask, s.moves)


def test_horizontal_vertical_diagonals_edges():
    assert build([0, 0, 1, 1, 2, 2, 3]).is_win()                    # horizontal at the left edge
    assert build([3, 0, 4, 1, 5, 2, 6]).is_win()                    # horizontal at the right edge
    assert build([0, 1, 0, 1, 0, 1, 0]).is_win()                    # vertical
    assert build([6, 5, 6, 5, 6, 5, 6]).is_win()                    # vertical, right edge
    # diagonal rising to the right: X at (0,0),(1,1),(2,2),(3,3)
    assert build([0, 1, 1, 2, 2, 3, 2, 3, 3, 6, 3]).is_win()
    # diagonal falling to the right (mirror image of the one above)
    assert build([6, 5, 5, 4, 4, 3, 4, 3, 3, 0, 3]).is_win()
    assert not build([0, 1, 0, 1, 0]).is_win()                      # only three in a column


def test_full_board_draw():
    # column pattern that fills 6x7 without four in a row (checked against the reference)
    rng = np.random.default_rng(3)
    for _ in range(2000):
        s, ref = State(), ReferenceGame()
        while s.legal_moves():
            m = int(rng.choice(s.legal_moves()))
            s.play(m)
            ref.play(m)
        if s.is_draw():
            assert ref.is_draw() and s.moves == 42 and not s.is_win()
            assert s.legal_moves() == []
            return
    pytest.fail("no draw found")


def test_illegal_moves_raise():
    s = build([0] * 6)
    with pytest.raises(ValueError):
        s.play(0)
    with pytest.raises(ValueError):
        s.play(7)


# ---------------------------------------------------------------- 2. perspective
def test_to_planes_perspective_and_orientation():
    rng = np.random.default_rng(4)
    for _ in range(100):
        s = random_state(rng)
        p = s.to_planes()
        assert p.dtype == np.float32 and p.shape == (2, 6, 7)
        board = np.array(s.to_board()).reshape(6, 7)            # top row first, like Kaggle
        me = s.current_player
        assert np.array_equal(p[0], (board == me).astype(np.float32))
        assert np.array_equal(p[1], (board == 3 - me).astype(np.float32))
        # same position viewed by the other player: planes swap
        t = s.copy()
        t.cur ^= t.mask
        assert np.array_equal(t.to_planes()[0], p[1]) and np.array_equal(t.to_planes()[1], p[0])


def test_third_plane_flag():
    s = build([3])
    assert s.to_planes(3)[2].min() == 0 and build([]).to_planes(3)[2].min() == 1


# ---------------------------------------------------------------- 3. mirror
def test_mirror_involution_and_legal_moves():
    rng = np.random.default_rng(5)
    for _ in range(200):
        s = random_state(rng)
        m = s.mirror()
        assert m.mirror().to_key() == s.to_key()
        assert sorted(m.legal_moves()) == sorted(6 - c for c in s.legal_moves())
        assert np.array_equal(m.to_planes(), s.to_planes()[:, :, ::-1])
        assert m.is_win() == s.is_win()


def test_mirror_same_game_theoretic_value_small_board():
    rng = np.random.default_rng(6)
    for _ in range(15):
        s = random_state(rng, 4, 5, 3, max_plies=8)
        a, b = NegamaxAgent(30), NegamaxAgent(30)
        a.select(s)
        b.select(s.mirror())
        assert a.last_score == b.last_score


# ---------------------------------------------------------------- baselines sanity
def test_baselines_take_win_and_block():
    s = build([0, 1, 0, 1, 0])        # O to move, X threatens col 0
    for agent in (GreedyAgent(0), NegamaxAgent(2), MCTSAgent(300, rollout="heuristic", seed=1)):
        assert agent.select(s) == 0, agent.name
    s.play(2)                          # X to move, wins at col 0
    for agent in (GreedyAgent(0), NegamaxAgent(2), MCTSAgent(300, seed=1)):
        assert agent.select(s) == 0, agent.name


def test_negamax_finds_forced_win_in_three():
    # X has an open three on the bottom row (cols 1,2,3 with 0 and 4 free): any O move loses
    s = build([1, 6, 2, 6, 3, 5])
    a = NegamaxAgent(4)
    assert a.select(s) in (0, 4)
    assert a.last_score > 50_000


def test_random_agent_is_legal_and_seeded():
    s = build([0] * 6)
    assert all(RandomAgent(i).select(s) in s.legal_moves() for i in range(20))
    assert RandomAgent(5).select(State()) == RandomAgent(5).select(State())


def test_random_opening_nonterminal():
    rng = np.random.default_rng(0)
    for _ in range(50):
        assert not build(random_opening(rng, 4)).is_terminal()
