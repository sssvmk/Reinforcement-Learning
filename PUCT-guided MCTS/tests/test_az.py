import numpy as np
import pytest
import torch

from az_mcts import AZPlayer, Node, Search, UniformEvaluator, PlayerSpec, run_games_batched
from game import State, random_state
from network import TorchEvaluator, masked_softmax


def build(moves):
    s = State()
    for m in moves:
        s.play(m)
    return s


# ---------------------------------------------------------------- 4. masking
def test_policy_masking_torch_and_numpy(random_net):
    from np_net import masked_softmax as np_softmax
    rng = np.random.default_rng(0)
    for _ in range(50):
        s = random_state(rng)
        legal = np.zeros((1, 7), dtype=bool)
        legal[0, s.legal_moves()] = True
        planes = s.to_planes()[None]
        ev = TorchEvaluator(random_net)
        p, v = ev.evaluate(planes, legal)
        assert np.all(p[0][~legal[0]] == 0.0) and abs(p.sum() - 1) < 1e-6
        assert -1 <= v[0] <= 1
        logits, _ = random_net(torch.from_numpy(planes))
        q = np_softmax(logits.detach().numpy(), legal)
        assert np.all(q[0][~legal[0]] == 0.0) and abs(q.sum() - 1) < 1e-6
        t = masked_softmax(logits, torch.from_numpy(legal))
        assert torch.all(t[0][~torch.from_numpy(legal[0])] == 0)


# ---------------------------------------------------------------- 5. PUCT behaviour
@pytest.mark.parametrize("sims", [1, 7, 50, 200])
def test_root_visits_sum_to_budget(random_net, sims):
    p = AZPlayer(TorchEvaluator(random_net), sims=sims)
    s = p.search(build([3, 3, 2]))
    assert s.visits().sum() == sims and s.sims == sims
    assert s.visits()[~s.root.legal].sum() == 0


def test_always_legal_including_single_legal_column(random_net):
    p = AZPlayer(TorchEvaluator(random_net), sims=30)
    rng = np.random.default_rng(0)
    s = None
    while s is None:                         # random playout that ends with exactly one open column
        t = State()
        while not t.is_terminal() and len(t.legal_moves()) > 1:
            t.play(int(rng.choice(t.legal_moves())))
        if not t.is_terminal():
            s = t
    assert len(s.legal_moves()) == 1
    assert p.select(s) == s.legal_moves()[0]
    rng = np.random.default_rng(1)
    for _ in range(20):
        st = random_state(rng)
        assert p.select(st) in st.legal_moves()
    assert AZPlayer(TorchEvaluator(random_net), sims=0).select(build([0] * 6)) in range(1, 7)   # policy only


@pytest.mark.parametrize("sims", [100, 400])
def test_takes_immediate_win_and_blocks_with_untrained_net(random_net, sims):
    ev = TorchEvaluator(random_net)
    block = build([0, 1, 0, 1, 0])           # O to move; X threatens a vertical four in column 0
    assert AZPlayer(ev, sims=sims).select(block) == 0
    win = block.copy().play(2)               # X to move and wins in column 0
    assert AZPlayer(ev, sims=sims).select(win) == 0
    # horizontal versions
    h = build([1, 1, 2, 2, 3, 6])            # X has 1,2,3 on the bottom row; X to move wins at 0 or 4
    assert AZPlayer(ev, sims=sims).select(h) in (0, 4)


def test_policy_only_and_uniform_priors_work():
    ev = UniformEvaluator()
    assert AZPlayer(ev, sims=0).select(State()) in range(7)
    assert AZPlayer(ev, sims=100).select(build([0, 1, 0, 1, 0])) == 0


# ---------------------------------------------------------------- 6. backup sign
def test_terminal_value_and_backup_sign():
    ev = UniformEvaluator()
    s = build([0, 1, 0, 1, 0, 2])            # X to move; col 0 wins immediately
    search = Search(s, 1.5)
    # root expansion
    req = search.begin_sim()
    p, v = ev.evaluate(req.to_planes()[None], np.ones((1, 7), bool))
    search.finish_sim(p[0], v[0])
    for _ in range(40):
        r = search.begin_sim()
        if r is not None:
            p, v = ev.evaluate(r.to_planes()[None], np.array([[c in r.legal_moves() for c in range(7)]]))
            search.finish_sim(p[0], v[0])
    root = search.root
    assert root.N[0] > 0
    assert root.W[0] == root.N[0]                 # the winning move is worth +1 to the player to move
    assert search.best_move() == 0
    # terminal child state: previous mover won -> value -1 for the player to move there
    t = s.child(0)
    assert t.is_win() and t.legal_moves() == []


def test_forced_win_preferred_with_deeper_win():
    ev = UniformEvaluator()
    s = build([1, 6, 2, 6, 3, 5])             # X plays 0 or 4 -> open three, wins next move
    pl = AZPlayer(ev, sims=300)
    assert pl.select(s) in (0, 4)


# ---------------------------------------------------------------- 7. Dirichlet noise only in self-play
def test_noise_only_when_requested():
    ev = UniformEvaluator()
    s = build([3])
    def root_P(noise):
        pl = AZPlayer(ev, sims=5, noise=noise, seed=0)
        return pl.search(s).root.P
    clean = root_P(False)
    noisy = root_P(True)
    assert np.allclose(clean, np.where(clean > 0, 1 / 7, 0))
    assert not np.allclose(noisy, clean) and abs(noisy.sum() - 1) < 1e-9
    assert AZPlayer(ev).noise is False and AZPlayer(ev).temp_moves == 0


def test_training_selfplay_uses_noise_and_arena_does_not(monkeypatch, random_net, cfg):
    import train_az
    seen = []
    real = train_az.run_games_batched
    def spy(pairs, *a, **k):
        seen.append(pairs[0])
        return real(pairs, *a, **k)
    monkeypatch.setattr(train_az, "run_games_batched", spy)
    cfg = dict(cfg)
    cfg["train"] = dict(cfg["train"], sims_per_move=3, parallel_games=2, arena_sims=3)
    train_az.selfplay(random_net, cfg, 0, 2)
    assert seen[-1][0].noise is True and seen[-1][0].temp_moves == cfg["mcts"]["temp_moves"]
    ev = TorchEvaluator(random_net)
    train_az.arena(ev, ev, cfg, 2, 3, 0)
    assert seen[-1][0].noise is False and seen[-1][0].temp_moves == 0


def test_reuse_subtree_keeps_visits_and_legality(random_net):
    p = AZPlayer(TorchEvaluator(random_net), sims=60, reuse=True)
    s = State()
    for _ in range(6):
        m = p.select(s)
        assert m in s.legal_moves()
        s.play(m)
        s.play(s.legal_moves()[0])
    assert p.last_search is not None
