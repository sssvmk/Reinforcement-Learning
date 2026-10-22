"""Isolated unit tests for demand_model.py (bootstrap / negbin /
deterministic_mean), using synthetic data -- no M5 files needed."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from demand_model import (  # noqa: E402
    DayFeatures, BootstrapDemandModel, NegBinDemandModel,
    DeterministicMeanDemandModel, make_demand_model,
)


def _synthetic_df(n=700, seed=0):
    """7 weekdays x 100 weeks, weekday-dependent mean demand plus SNAP
    lift on weekday 1, so models have genuine structure to recover."""
    rng = np.random.default_rng(seed)
    wdays = np.tile(np.arange(1, 8), n // 7 + 1)[:n]
    snap = (rng.random(n) < 0.3).astype(int)
    base_mean = {1: 10, 2: 4, 3: 4, 4: 4, 5: 4, 6: 6, 7: 8}
    means = np.array([base_mean[w] for w in wdays], dtype=float)
    means = means + snap * (wdays == 1) * 5.0
    sales = rng.poisson(means)
    return pd.DataFrame({"wday": wdays, "snap": snap, "sales": sales})


def test_deterministic_mean_matches_training_mean_per_weekday():
    df = _synthetic_df()
    model = DeterministicMeanDemandModel().fit(df)
    for wday in range(1, 8):
        expected = df.loc[df["wday"] == wday, "sales"].mean()
        got = model.mean(DayFeatures(wday=wday))
        assert abs(got - expected) < 1e-9
        # sample() is deterministic (always returns the mean)
        rng = np.random.default_rng(0)
        assert model.sample(DayFeatures(wday=wday), rng) == got


def test_bootstrap_sample_only_draws_from_observed_training_values():
    df = _synthetic_df()
    model = BootstrapDemandModel(condition_on_snap=False).fit(df)
    pool = set(df.loc[df["wday"] == 3, "sales"].to_numpy().tolist())
    rng = np.random.default_rng(1)
    draws = [model.sample(DayFeatures(wday=3), rng) for _ in range(200)]
    assert all(d in pool for d in draws)


def test_bootstrap_conditions_on_snap_when_enough_samples():
    df = _synthetic_df(n=2000)
    model = BootstrapDemandModel(condition_on_snap=True, min_cell_size=15).fit(df)
    rng = np.random.default_rng(2)
    # weekday 1 has a strong SNAP lift built into the synthetic generator
    snap_mean = np.mean([model.sample(DayFeatures(wday=1, snap=1), rng) for _ in range(500)])
    no_snap_mean = np.mean([model.sample(DayFeatures(wday=1, snap=0), rng) for _ in range(500)])
    assert snap_mean > no_snap_mean + 1.0


def test_bootstrap_falls_back_to_weekday_pool_when_cell_too_small():
    df = _synthetic_df(n=50)  # too few rows per (wday, snap) cell
    model = BootstrapDemandModel(condition_on_snap=True, min_cell_size=1000).fit(df)
    assert model._by_wday_snap == {}
    rng = np.random.default_rng(0)
    # should not raise, should fall back to the weekday-only pool
    s = model.sample(DayFeatures(wday=2, snap=1), rng)
    assert s >= 0.0


def test_negbin_mean_close_to_empirical_mean():
    df = _synthetic_df(n=2000)
    model = NegBinDemandModel().fit(df)
    for wday in range(1, 8):
        empirical_mean = df.loc[df["wday"] == wday, "sales"].mean()
        got_mean = model.mean(DayFeatures(wday=wday))
        assert abs(got_mean - empirical_mean) < 1.0


def test_negbin_sample_is_nonnegative_integer_valued():
    df = _synthetic_df()
    model = NegBinDemandModel().fit(df)
    rng = np.random.default_rng(3)
    draws = [model.sample(DayFeatures(wday=w), rng) for w in range(1, 8) for _ in range(20)]
    assert all(d >= 0 for d in draws)
    assert all(float(d).is_integer() for d in draws)


def test_unknown_weekday_returns_zero_gracefully():
    df = _synthetic_df()
    model_det = DeterministicMeanDemandModel().fit(df)
    model_boot = BootstrapDemandModel().fit(df)
    model_nb = NegBinDemandModel().fit(df)
    rng = np.random.default_rng(0)
    bogus = DayFeatures(wday=99)
    assert model_det.sample(bogus, rng) == 0.0
    assert model_boot.sample(bogus, rng) == 0.0
    assert model_nb.sample(bogus, rng) == 0.0


def test_make_demand_model_factory_dispatches_correctly():
    df = _synthetic_df()
    assert isinstance(make_demand_model("bootstrap", df), BootstrapDemandModel)
    assert isinstance(make_demand_model("negbin", df), NegBinDemandModel)
    assert isinstance(make_demand_model("deterministic_mean", df), DeterministicMeanDemandModel)
    with pytest.raises(ValueError):
        make_demand_model("not_a_real_kind", df)


def test_no_leakage_fit_only_uses_given_rows():
    """Fitting on a slice must not be influenced by rows outside the slice:
    use a sentinel value outside the slice and confirm it never appears."""
    df = _synthetic_df(n=200)
    train_df = df.iloc[:100].copy()
    held_out = df.iloc[100:].copy()
    held_out["sales"] = 9999  # sentinel, distinguishable from any real draw
    model = BootstrapDemandModel().fit(train_df)  # fit on train_df only
    rng = np.random.default_rng(0)
    draws = [model.sample(DayFeatures(wday=w), rng) for w in range(1, 8) for _ in range(50)]
    assert 9999 not in draws
