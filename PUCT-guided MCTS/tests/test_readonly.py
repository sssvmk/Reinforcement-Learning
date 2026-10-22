import os

import pytest

from config import ROOT, load_config, output_dir, kaggle_dir, verify_kaggle_defaults


def snapshot(d):
    return {str(p.relative_to(d)): (p.stat().st_size, p.stat().st_mtime_ns) for p in d.rglob("*") if p.is_file()}


# ---------------------------------------------------------------- 12. read-only data
def test_kaggle_dir_is_never_modified(cfg, tmp_path):
    kd = kaggle_dir(cfg)
    if not kd.is_dir():
        pytest.skip(f"KAGGLE_DIR not present: {kd}")
    before = snapshot(kd)
    verify_kaggle_defaults(cfg)
    import copy
    import train_az
    c = copy.deepcopy(cfg)
    c["train"].update(iterations=1, games_per_iter=2, sims_per_move=3, steps_per_iter=2, arena_games=2, arena_sims=2,
                      progress_every=0, parallel_games=2, batch_size=8)
    train_az.train(c, tmp_path)
    assert snapshot(kd) == before


def test_output_dir_inside_kaggle_dir_is_refused(cfg, monkeypatch):
    c = dict(cfg)
    c["kaggle_dir"] = str(ROOT)           # pretend KAGGLE_DIR contains the outputs
    monkeypatch.delenv("KAGGLE_DIR", raising=False)
    with pytest.raises(RuntimeError, match="inside KAGGLE_DIR"):
        output_dir(c, create=False)


def test_env_var_overrides_kaggle_dir(cfg, monkeypatch, tmp_path):
    monkeypatch.setenv("KAGGLE_DIR", str(tmp_path))
    assert kaggle_dir(cfg) == tmp_path
    with pytest.raises(FileNotFoundError, match=str(tmp_path).replace("\\", "\\\\")):
        verify_kaggle_defaults(cfg)


def test_connectx_defaults_match_config(cfg):
    kd = kaggle_dir(cfg)
    if not kd.is_dir():
        pytest.skip("no KAGGLE_DIR")
    d = verify_kaggle_defaults(cfg)
    g = cfg["game"]
    assert (d["rows"], d["columns"], d["inarow"]) == (g["rows"], g["columns"], g["inarow"])
