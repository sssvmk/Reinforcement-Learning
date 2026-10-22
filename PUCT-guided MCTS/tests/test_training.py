import copy
import os
import time

import numpy as np
import pytest
import torch

from az_mcts import PlayerSpec, run_games_batched
from game import State
from network import TorchEvaluator, build_net


def make_net(cfg, seed=0):
    torch.manual_seed(seed)
    return build_net(cfg)


# ---------------------------------------------------------------- 8. self-play data
def test_selfplay_data_shapes_and_targets(cfg, random_net):
    ev = TorchEvaluator(random_net)
    spec = PlayerSpec(ev, 8, noise=True, temp_moves=10)
    res = run_games_batched([(spec, spec)] * 6, list(range(6)))
    for r in res:
        n = len(r.planes)
        assert n == len(r.pis) == len(r.movers) > 0
        z = r.values()
        assert len(z) == n and set(z) <= {-1.0, 0.0, 1.0}
        for x, pi, mover, zi in zip(r.planes, r.pis, r.movers, z):
            assert x.shape == (2, 6, 7) and x.dtype == np.float32
            assert pi.shape == (7,) and abs(pi.sum() - 1) < 1e-5 and (pi >= 0).all()
            # the recorded position is the true position after k plies of the recorded game, from the mover's view
            k = int(x.sum())
            s = State()
            for m in r.moves[:k]:
                s.play(m)
            assert s.current_player == mover and np.array_equal(s.to_planes(), x)
            assert pi[~np.isin(np.arange(7), s.legal_moves())].sum() == 0
            expected = 0.0 if r.winner == 0 else (1.0 if r.winner == mover else -1.0)
            assert zi == expected
        replay = State()
        for m in r.moves:
            replay.play(m)
        assert r.winner == (3 - replay.current_player if replay.is_win() else 0)


def test_mirror_augmentation_preserves_properties(cfg, random_net):
    from train_az import augment_mirror, selfplay
    c = dict(cfg)
    c["train"] = dict(cfg["train"], sims_per_move=6, parallel_games=3)
    x, pi, z, _ = selfplay(random_net, c, 5, 3)
    mx, mpi, mz = augment_mirror(x, pi, z)
    assert mx.shape == x.shape and np.allclose(mpi.sum(1), 1) and np.array_equal(mz, z)
    assert np.array_equal(mx[..., ::-1], x) and np.array_equal(mpi[:, ::-1], pi)
    for i in range(min(len(x), 20)):
        s = State()
        # mirrored planes are the planes of the mirrored position: rebuild a State from planes
        assert np.array_equal(mx[i][0] + mx[i][1], (x[i][0] + x[i][1])[:, ::-1])
        # illegal columns (full) stay illegal after mirroring
        assert mpi[i][(mx[i][0] + mx[i][1])[0] > 0].sum() == 0


# ---------------------------------------------------------------- 9. training smoke test
def test_loss_decreases_on_tiny_buffer(cfg, random_net):
    from train_az import ReplayBuffer, selfplay, train_steps
    c = dict(cfg)
    c["train"] = dict(cfg["train"], sims_per_move=6, parallel_games=4)
    x, pi, z, _ = selfplay(random_net, c, 9, 4)
    buf = ReplayBuffer(1000, 2, 6, 7)
    buf.add(x, pi, z)
    net = make_net(cfg, 1)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
    rng = np.random.default_rng(0)
    first = sum(train_steps(net, opt, buf, 10, 64, rng))
    train_steps(net, opt, buf, 300, 64, rng)
    last = sum(train_steps(net, opt, buf, 10, 64, rng))
    assert last < first - 0.1, (first, last)


# ---------------------------------------------------------------- 11. determinism
def test_same_seed_same_selfplay_game(cfg):
    def play(seed):
        net = make_net(cfg, 3).eval()
        spec = PlayerSpec(TorchEvaluator(net), 12, noise=True, temp_moves=10)
        return run_games_batched([(spec, spec)] * 3, [seed, seed + 1, seed + 2])
    a, b = play(11), play(11)
    for ra, rb in zip(a, b):
        assert ra.moves == rb.moves and ra.winner == rb.winner
        assert all(np.array_equal(p, q) for p, q in zip(ra.pis, rb.pis))
    assert play(11)[0].moves != play(99)[0].moves or play(11)[1].moves != play(99)[1].moves


def test_same_seed_same_training_step(cfg, random_net):
    from train_az import ReplayBuffer, selfplay, train_steps, set_seeds
    c = dict(cfg)
    c["train"] = dict(cfg["train"], sims_per_move=4, parallel_games=2)
    x, pi, z, _ = selfplay(random_net, c, 2, 2)
    out = []
    for _ in range(2):
        set_seeds(42)
        net = build_net(cfg)
        buf = ReplayBuffer(500, 2, 6, 7)
        buf.add(x, pi, z)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        losses = train_steps(net, opt, buf, 5, 32, np.random.default_rng(7))
        out.append((losses, [p.detach().clone() for p in net.parameters()]))
    assert out[0][0] == out[1][0]
    assert all(torch.equal(p, q) for p, q in zip(out[0][1], out[1][1]))


def test_train_loop_runs_and_resumes(tmp_path, cfg):
    import train_az
    c = copy.deepcopy(cfg)
    c["train"].update(iterations=2, games_per_iter=2, sims_per_move=4, steps_per_iter=3, arena_games=2, arena_sims=3,
                      progress_every=0, parallel_games=2, batch_size=16)
    train_az.train(c, tmp_path)
    assert (tmp_path / "best_weights.npz").exists() and (tmp_path / "checkpoints" / "latest.pt").exists()
    c["train"]["iterations"] = 3
    train_az.train(c, tmp_path, resume=True)
    lines = (tmp_path / "logs" / "train_log.jsonl").read_text().splitlines()
    assert len(lines) == 3
