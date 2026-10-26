"""Part 3 tests 2-5, 13, 14 for the trading domain (real data, read-only) and the ranking metrics."""
import hashlib, os
from pathlib import Path
import numpy as np, pandas as pd, pytest
from common import load_config, make_folds
from data_trading import load_all, trailing_features, REQUIRED
from domains import TradingDomain
from log_data import simulate_trading_log, behavior_prob, sample_starts
from trading_env import TradingEnv
from true_value import dp_values, policy_probs_table
from metrics import ranking_metrics, tie_matrix, block_bootstrap_mean


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def data(cfg):
    return load_all(cfg)[0][0]


@pytest.fixture(scope="module")
def folds(data, cfg):
    return make_folds(data.N, cfg)


def test2_reward_by_hand_and_no_lookahead(cfg, data):
    env = TradingEnv(data, cost_bps=5.0)
    t = 100
    env.set_state(t, 0)
    r = env.step(1)
    assert r == pytest.approx(data.close[t + 1] / data.close[t] - 1 - 5e-4)
    assert env.get_state() == (t + 1, 1)
    env.set_state(t, 1)
    assert env.step(1) == pytest.approx(data.close[t + 1] / data.close[t] - 1)
    env.set_state(t, 1)
    assert env.step(0) == pytest.approx(-5e-4)
    # features at row t unchanged when every later row is deleted
    p = Path(cfg["paths"]["trading_files"][0])
    df = pd.read_csv(p, compression="gzip", usecols=lambda c: c in REQUIRED)
    df["tdate"] = pd.to_datetime(df["tdate"])
    df = df.sort_values("tdate").reset_index(drop=True)
    full = trailing_features(df)
    cut = trailing_features(df.iloc[:1500].copy())
    a, b = full.iloc[:1500].to_numpy(), cut.to_numpy()
    assert np.allclose(a, b, equal_nan=True)


def test3_chronological_disjoint_splits(data, folds, cfg):
    for f in folds.values():
        assert f.train[1] <= f.val[0] and f.val[1] <= f.test[0] and f.train[0] == min(f.train[0], f.val[0])
        assert f.ope[1] <= f.tune[0] and f.ope[0] == f.val[0] and f.tune[1] == f.val[1]
    assert len([k for k in folds if k.startswith("wf")]) >= 3
    f = folds["primary"]
    rng = np.random.default_rng(0)
    H = cfg["mdp"]["H"]
    a = simulate_trading_log(data, f.ope[0], f.ope[1], 50, 0.3, H, 5.0, rng)
    b = simulate_trading_log(data, f.tune[0], f.tune[1], 50, 0.3, H, 5.0, rng)
    assert len(set(a.win_id) & set(b.win_id)) == 0
    assert a.start.min() >= f.ope[0] and (a.start + H).max() <= f.ope[1] - 1
    assert b.start.min() >= f.tune[0] and (b.start + H).max() <= f.tune[1] - 1


def test4_behavior_probabilities(data, folds, cfg):
    f = folds["primary"]
    eps = 0.3
    log = simulate_trading_log(data, f.ope[0], f.ope[1], 400, eps, cfg["mdp"]["H"], 5.0, np.random.default_rng(1))
    assert log.bprob.min() > 0 and log.bprob.max() <= 1
    assert set(np.unique(np.round(log.bprob, 6))) <= {round(1 - eps + eps / 2, 6), round(eps / 2, 6)}
    rule = data.sar_up[log.raw[:, :-1, 0].astype(int)]
    follow = (log.act == rule).mean()
    assert follow == pytest.approx(1 - eps + eps / 2, abs=0.01)
    # every target action has b > 0 (coverage)
    assert behavior_prob(np.array([0, 1]), np.array([1, 1]), 0.3, 2).min() > 0


def test5_data_read_only(cfg):
    p = Path(cfg["paths"]["trading_files"][0])
    before = (p.stat().st_size, p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest())
    load_all(cfg)
    after = (p.stat().st_size, p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest())
    assert before == after
    assert not any(q.suffix == ".gzip" for q in Path("outputs").rglob("*"))


def test13_determinism_and_setstate_equals_dp(data, folds, cfg):
    f = folds["primary"]
    H = cfg["mdp"]["H"]
    a = simulate_trading_log(data, f.ope[0], f.ope[1], 30, 0.3, H, 5.0, np.random.default_rng(5))
    b = simulate_trading_log(data, f.ope[0], f.ope[1], 30, 0.3, H, 5.0, np.random.default_rng(5))
    assert np.array_equal(a.act, b.act) and np.array_equal(a.r, b.r)
    # a deterministic always-long policy: set_state rollout reward sum equals the DP value
    gamma = cfg["mdp"]["gamma"]
    lo, hi = f.test
    G = np.ones((data.N, 2), dtype=np.int64)
    P = policy_probs_table(G, 0.0)
    V = dp_values(data.fwd_ret, lo, hi, P, 5e-4, H, gamma)
    env = TradingEnv(data, 5.0)
    t0 = lo + 10
    env.set_state(t0, 0)
    ret = sum(gamma ** k * env.step(1) for k in range(H))
    assert ret == pytest.approx(float(V[t0 - lo, 0]) if V.shape[0] == hi - lo else float(V[t0, 0]), rel=1e-6, abs=1e-9)


def test14_ranking_metrics():
    true = np.array([1.0, 2.0, 3.0, 4.0])
    m = ranking_metrics(true * 10, true)
    assert m["spearman"] == pytest.approx(1) and m["regret1"] == 0 and m["pair_acc"] == 1 and m["nregret1"] == 0
    m = ranking_metrics(-true, true)
    assert m["spearman"] == pytest.approx(-1) and m["regret1"] == pytest.approx(3) and m["nregret1"] == pytest.approx(1)
    assert m["pair_acc"] == 0
    m = ranking_metrics(np.array([5.0, 5.0, 1.0, 0.0]), true)  # tie for the top: tie-aware regret = mean of tied
    assert m["regret1_tie"] == pytest.approx(4 - 1.5)
    assert np.isnan(ranking_metrics(np.ones(4), true)["spearman"])
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, 200))
    x[0] += 5
    T = tie_matrix(x, 20, 200, rng)
    assert T[0, 1] == 1 and T[1, 0] == 0
    assert block_bootstrap_mean(x[0], 20, 50, rng).shape == (50,)
