import os
import zipfile
import numpy as np
import pandas as pd
import pytest
import data_m5
import metrics
from data_trading import TradingData, make_splits
from fdpg.crn import PathSampler
from fdpg.optimizer import HParams, run_fd
from problems import build_trading_problem


def test_splits_chronological_disjoint_and_windows_inside():
    s = make_splits(1000)
    assert s["train"][1] == s["val"][0] and s["val"][1] == s["test"][0] and s["test"][1] == 1000
    assert s["train"] == (0, 600) and s["val"] == (600, 800)
    rng = np.random.default_rng(0)
    for k, (lo, hi) in s.items():
        st = PathSampler(lo, hi, 100).sample(rng, 5000)
        assert st.min() >= lo and st.max() + 100 <= hi


def test_no_test_data_touched_during_training_and_stats_from_train():
    rng = np.random.default_rng(1)
    n = 2000
    X = rng.normal(size=(n, 5)); R = rng.normal(0, 0.01, n)
    data = TradingData(np.arange(n), np.ones(n), X, R, {})
    prob = build_trading_problem(data, "T1", {"horizon": 100, "cost_bps": 5.0, "n_paths": 8})
    tr = prob.splits["train"]
    np.testing.assert_allclose(prob.policy.mu, X[tr[0]:tr[1]].mean(axis=0))
    Jt, Jv, Jd = prob.objectives()
    run_fd(Jt, Jv, np.zeros(7), prob.policy.ilo, prob.policy.ihi, HParams(budget=3000), 0, Jdiag=Jd)
    assert Jt.max_touch <= prob.splits["train"][1] and Jd.max_touch <= prob.splits["train"][1]
    assert Jv.max_touch <= prob.splits["val"][1] < prob.splits["test"][1]


def _snapshot(d):
    return sorted((os.path.join(r, f), os.path.getmtime(os.path.join(r, f)), os.path.getsize(os.path.join(r, f)))
                  for r, _, fs in os.walk(d) for f in fs)


def _write_m5(d, zipped=False):
    days = 400
    cal = pd.DataFrame({"d": [f"d_{i+1}" for i in range(days)], "date": pd.date_range("2011-01-29", periods=days).astype(str),
                        "wm_yr_wk": 11101 + np.arange(days) // 7, "wday": np.arange(days) % 7 + 1,
                        "event_name_1": [None] * days, "snap_CA": 0, "snap_TX": 0, "snap_WI": 0})
    sales = pd.DataFrame({"id": ["FOODS_1_001_CA_1_validation"], "item_id": ["FOODS_1_001"], "dept_id": ["FOODS_1"],
                          "cat_id": ["FOODS"], "store_id": ["CA_1"], "state_id": ["CA"]})
    y = np.r_[np.zeros(5), np.random.default_rng(0).poisson(3, days - 5)]
    sales = pd.concat([sales, pd.DataFrame([y], columns=[f"d_{i+1}" for i in range(days)])], axis=1)
    wk = np.unique(cal.wm_yr_wk)
    prices = pd.DataFrame({"store_id": "CA_1", "item_id": "FOODS_1_001", "wm_yr_wk": wk[:-10], "sell_price": 2.5})
    files = {"calendar.csv": cal, "sales_train_validation.csv": sales, "sell_prices.csv": prices}
    for k, v in files.items():
        v.to_csv(os.path.join(d, k), index=False)
    if zipped:
        with zipfile.ZipFile(os.path.join(d, "m5.zip"), "w") as z:
            for k in files:
                z.write(os.path.join(d, k), k); os.remove(os.path.join(d, k))


@pytest.mark.parametrize("zipped", [False, True])
def test_m5_loader_read_only_and_series(tmp_path, zipped):
    _write_m5(str(tmp_path), zipped)
    before = _snapshot(tmp_path)
    ok, missing, found = data_m5.status(str(tmp_path))
    assert ok and not missing
    df = data_m5.load_series(found, "FOODS_1_001_CA_1_validation", cache_dir=None)
    assert df["sales"].iloc[0] > 0 and len(df) == 395          # leading zeros trimmed
    assert df["price_ok"].iloc[0] and not df["price_ok"].iloc[-1]   # price gap > 2 weeks stays missing
    assert _snapshot(tmp_path) == before


def test_m5_missing_file_reported(tmp_path):
    _write_m5(str(tmp_path))
    os.remove(tmp_path / "sell_prices.csv")
    ok, missing, _ = data_m5.status(str(tmp_path))
    assert not ok and "sell_prices.csv" in missing[0] and str(tmp_path) in missing[0]


def test_metrics_hand_calculated():
    D = {k: np.array([[2.0, 4.0]]) for k in ("sales",)}
    D.update({"demand": np.array([[2.0, 8.0]]), "lost": np.array([[0.0, 4.0]]), "on_hand": np.array([[1.0, 3.0]]),
              "order": np.array([[0.0, 5.0]]), "revenue": np.array([[20.0, 40.0]]), "purchase": np.array([[0.0, 3.0]]),
              "holding": np.zeros((1, 2)), "stockout": np.zeros((1, 2)), "fixed": np.zeros((1, 2))})
    m = metrics.inventory_metrics(D, 10.0)
    assert m["fill_rate"] == pytest.approx(0.6) and m["stockout_rate"] == 0.5 and m["lost_units"] == 4.0
    net = np.array([0.1, -0.2, 0.05])
    assert metrics.max_drawdown(net) == pytest.approx(0.88 / 1.1 - 1)
    t = metrics.trading_metrics(np.array([0.01, -0.01]), np.array([1.0, 1.0]))
    assert t["sharpe"] == pytest.approx(0.0, abs=1e-12) and t["hit_rate"] == 0.5
    s = np.array([0.01, 0.03, 0.02])
    assert metrics.trading_metrics(s, np.ones(3))["sharpe"] == pytest.approx(0.02 / 0.01 * np.sqrt(252))
