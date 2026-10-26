"""Tests for the extensions: open-to-open returns, Q_max-free config, neural policies under the derivative-free comparison methods."""
import numpy as np
import data_trading as dt
import common
import policies as P
from reinforce import comparison_methods as cm
from synth import make_inv


def test_open_returns_fallback_on_corporate_action():
    o = np.array([100., 101., 102., 51., 52., 53.])          # 2:1 split in open between t=2 and t=3
    c = np.array([100., 101., 102., 102.5, 103., 104.])      # adjusted close has no jump
    r, share = dt.open_returns(o, c)
    assert np.isclose(r[0], 102 / 101 - 1)                   # ordinary day: open-to-open
    assert np.isclose(r[1], c[3] / c[2] - 1)                 # split day: falls back to close-to-close
    assert r[-2:].tolist() == [0.0, 0.0] and share > 0


def test_qmax_free_config(monkeypatch):
    monkeypatch.setenv("QMAX_FREE", "1")
    assert common.load_config()["inventory"]["q_max_mult"] is None
    monkeypatch.setenv("TRADING_RETURN", "open")
    assert common.load_config()["trading"]["return_source"] == "open"


def test_neural_policy_box_matches_mean_dim():
    for pol in (P.make_stoch("inventory", "I4"), P.make_stoch("trading", "T3")):
        assert len(pol.lo) == len(pol.hi) == pol.mean_dim


def test_spsa_runs_on_neural_policy():
    pr = make_inv()
    pol = P.make_stoch("inventory", "I4")
    th0 = pol.init(np.random.default_rng(0))
    r = cm.spsa(pr, pol, th0, np.random.default_rng(1), 200, B=8)
    assert np.isfinite(r["best_val"]) and len(r["theta_best"]) == pol.n_params
