import ast
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from game import State, random_state
from network import export_weights


# ---------------------------------------------------------------- 10. parity
@pytest.fixture(scope="module")
def built(tmp_path_factory, cfg, random_net):
    from build_submission import build
    d = tmp_path_factory.mktemp("sub")
    w = export_weights(random_net, d / "w.npz")
    return random_net, w, build(cfg, w, d / "submission.py")


def test_numpy_torch_parity_1000_positions(built):
    from build_submission import parity_check
    net, w, _ = built
    dl, dv = parity_check(net, w, n=1000, seed=3)
    assert dl < 1e-4 and dv < 1e-4


def test_parity_fails_loudly_on_mismatch(built, tmp_path):
    from build_submission import parity_check
    net, w, _ = built
    z = dict(np.load(w))
    z["pol_b"] = z["pol_b"] + 0.01
    np.savez(tmp_path / "bad.npz", **z)
    with pytest.raises(AssertionError):
        parity_check(net, tmp_path / "bad.npz", n=50)


def test_weights_are_float32(built):
    _, w, _ = built
    with np.load(w) as z:
        assert all(z[k].dtype == np.float32 for k in z.files)


def test_np_net_does_not_import_torch():
    src = (Path(__file__).resolve().parent.parent / "submission" / "np_net.py").read_text()
    mods = {n.names[0].name.split(".")[0] for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Import)}
    mods |= {n.module.split(".")[0] for n in ast.walk(ast.parse(src)) if isinstance(n, ast.ImportFrom) and n.module}
    assert "torch" not in mods


def test_submission_is_self_contained_numpy_only(built):
    _, _, sub = built
    src = sub.read_text()
    mods = set()
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Import):
            mods |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            mods.add(n.module.split(".")[0])
    assert mods <= {"numpy", "__future__", "time", "base64", "io", "math"}, mods


def _load(path):
    import importlib.util
    sp = importlib.util.spec_from_file_location("sub_under_test", str(path))
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


def test_submission_legal_on_1000_random_positions_within_time_guard(built):
    _, _, sub = built
    m = _load(sub)
    rng = np.random.default_rng(11)
    limit = 0.45                                       # budget = min(0.5*limit, limit-0.4) -> floor of 0.05 s
    worst = 0.0
    for i in range(1000):
        s = random_state(rng)
        obs = {"board": s.to_board(), "mark": s.current_player, "step": s.moves}
        t = time.perf_counter()
        a = m.agent(obs, {"rows": 6, "columns": 7, "inarow": 4, "actTimeout": limit})
        worst = max(worst, time.perf_counter() - t)
        assert isinstance(a, int) and a in s.legal_moves()
    assert worst < limit, worst


def test_submission_accepts_attribute_style_objects_and_defaults(built):
    from types import SimpleNamespace as NS
    _, _, sub = built
    m = _load(sub)
    obs = NS(board=[0] * 42, mark=1)
    cfg = NS(rows=6, columns=7, inarow=4)                 # no timeout key -> default limit, uses ~1 s
    t = time.perf_counter()
    a = m.agent(obs, cfg)
    assert 0 <= a < 7 and time.perf_counter() - t < 2.0


def test_submission_fallback_on_garbage_never_raises(built):
    _, _, sub = built
    m = _load(sub)
    board = [0] * 42
    board[3] = 1                                          # inconsistent board for mark=1 -> exception inside -> fallback
    a = m.agent({"board": board, "mark": 1}, {"columns": 7, "rows": 6})
    assert a in range(7) and board[a] == 0
    assert m.agent(None, None) == 0                         # even total garbage returns an int
    assert m.agent({"board": [1, 1, 1] + [0] * 39, "mark": 2}, {"columns": 7, "rows": 6}) in range(7)


def test_submission_blocks_and_wins_through_kaggle_harness(built):
    from evaluate import play_kaggle_style
    _, _, sub = built
    m = _load(sub)
    def rnd(obs, cfg):
        return random.choice([c for c in range(7) if obs["board"][c] == 0])
    random.seed(0)
    wins = 0
    for first in (True, False):
        for _ in range(2):
            a1, a2 = (m.agent, rnd) if first else (rnd, m.agent)
            winner, info = play_kaggle_style(a1, a2, {"rows": 6, "columns": 7, "inarow": 4, "actTimeout": 2})
            assert "forfeit" not in info, info
            wins += winner == (1 if first else 2)
    assert wins >= 3        # untrained network + search should still beat a random player
