import hashlib
import numpy as np
import pandas as pd
import pytest
import policies as P
import trading_env
from data_trading import (SchemaError, TradingData, build_features, make_splits, read_trading)
from fdpg.optimizer import HParams, run_fd
from problems import build_trading_problem


def synth_df(n=600, seed=0):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    o = c * (1 + rng.normal(0, 0.003, n))
    h, l = np.maximum(o, c) * 1.01, np.minimum(o, c) * 0.99
    return pd.DataFrame({"tdate": pd.date_range("2010-01-01", periods=n, freq="B").astype(str), "open": o, "high": h,
                         "low": l, "close": c, "volume": rng.integers(1e5, 1e6, n).astype(float),
                         "sar": c * (1 + rng.normal(0, 0.03, n)), "RSI": rng.uniform(20, 80, n)})


def test_no_lookahead_features():
    df = synth_df()
    t0 = 300
    a = build_features(df, warmup_rows=70, min_rows=100)
    df2 = df.copy()
    rng = np.random.default_rng(5)
    for col in ("open", "high", "low", "close", "volume", "sar", "RSI"):
        df2.loc[t0 + 1:, col] = df2.loc[t0 + 1:, col] * rng.uniform(0.5, 2.0, len(df2) - t0 - 1)
    b = build_features(df2, warmup_rows=70, min_rows=100)
    k = t0 - 70 + 1
    np.testing.assert_array_equal(a.X[:k], b.X[:k])
    np.testing.assert_array_equal(a.close[:k], b.close[:k])


class SeqPolicy:
    def __init__(self, seq):
        self.seq, self.i = seq, 0

    def act(self, th, X, prev):
        p = np.full(len(X), self.seq[self.i]); self.i += 1
        return p


def test_hand_calculated_reward_and_bounds():
    R = np.array([0.1, -0.05, 0.02, 0.0, 0.0])
    X = np.zeros((5, 5))
    net = trading_env.simulate(SeqPolicy([1.0, 1.0, 0.0]), None, X, R, np.array([0]), 3, 0.001, detail=True)
    np.testing.assert_allclose(net["net"][0], [0.099, -0.05, -0.001])
    clipped = trading_env.simulate(SeqPolicy([1.7, -0.3, 0.5]), None, X, R, np.array([0]), 3, 0.0, detail=True)
    assert clipped["pos"].min() >= 0 and clipped["pos"].max() <= 1


def test_gzip_schema_and_file_untouched(tmp_path):
    df = synth_df(50)
    p = tmp_path / "X_extend.csv.gzip"
    df.to_csv(p, index=False, compression="gzip")
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    mt = p.stat().st_mtime
    assert len(read_trading(str(p))) == 50
    q = tmp_path / "Y_extend.csv.gzip"
    df.drop(columns=["sar"]).to_csv(q, index=False, compression="gzip")
    with pytest.raises(SchemaError):
        read_trading(str(q))
    assert hashlib.sha256(p.read_bytes()).hexdigest() == h and p.stat().st_mtime == mt
    assert sorted(x.name for x in tmp_path.iterdir()) == ["X_extend.csv.gzip", "Y_extend.csv.gzip"]


def test_fd_on_t1_beats_flat_on_synthetic_rule():
    rng = np.random.default_rng(0)
    n = 3000
    X = np.column_stack([rng.normal(50, 10, n), rng.choice([-1.0, 1.0], n), rng.normal(0, 0.02, n),
                         rng.normal(0, 0.03, n), rng.normal(0, 1, n)])
    R = 0.006 * X[:, 1] + rng.normal(0, 0.01, n)
    data = TradingData(np.arange(n), 100 * np.cumprod(1 + R), X, R, {})
    prob = build_trading_problem(data, "T1", {"horizon": 100, "cost_bps": 5.0, "n_paths": 16})
    Jt, Jv, Jd = prob.objectives()
    hp = HParams(est="central", eps=0.2, a=2.0, A=10, budget=12000)
    out = run_fd(Jt, Jv, np.zeros(7), prob.policy.ilo, prob.policy.ihi, hp, 0)
    th = out["u_best"] * prob.policy.scale
    test = prob.window_returns(prob.policy, th, "test").mean()
    assert test > 0.0                                   # beats always-flat (0)
    assert test > prob.window_returns(P.Fixed(1.0, "bh"), np.zeros(0), "test").mean()
