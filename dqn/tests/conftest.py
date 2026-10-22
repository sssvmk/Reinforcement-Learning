"""Shared pytest fixtures: synthetic SeriesBundle construction, independent of
the real M5 CSVs, so most tests run anywhere."""
import numpy as np
import pytest

from data import SeriesBundle


def make_synthetic_bundle(
    n_days=300,
    demand_fn=None,
    mean_price=10.0,
    train_frac=0.7,
    val_frac=0.15,
):
    rng = np.random.default_rng(0)
    if demand_fn is None:
        demand = rng.poisson(5, size=n_days).astype(float)
    else:
        demand = demand_fn(n_days)

    sell_price = np.full(n_days, mean_price, dtype=float)
    price_available = np.ones(n_days, dtype=bool)
    wday = (np.arange(n_days) % 7) + 1
    event_flag = np.zeros(n_days, dtype=int)
    snap_flag = np.zeros(n_days, dtype=int)
    dates = np.array(["2011-01-01"] * n_days, dtype="datetime64[D]")

    n_train = int(round(n_days * train_frac))
    n_val = int(round(n_days * val_frac))
    train_range = (0, n_train)
    val_range = (n_train, n_train + n_val)
    test_range = (n_train + n_val, n_days)

    mean_demand = float(np.mean(demand[:n_train]))
    std_demand = float(np.std(demand[:n_train]))

    return SeriesBundle(
        series_id="SYNTH_TEST",
        item_id="SYNTH",
        store_id="X_1",
        state_id="CA",
        dates=dates,
        demand=demand,
        sell_price=sell_price,
        price_available=price_available,
        wday=wday,
        event_flag=event_flag,
        snap_flag=snap_flag,
        mean_demand=mean_demand,
        std_demand=std_demand,
        mean_price=mean_price,
        train_range=train_range,
        val_range=val_range,
        test_range=test_range,
    )


@pytest.fixture
def synthetic_bundle():
    return make_synthetic_bundle()


@pytest.fixture
def constant_demand_bundle():
    """Toy deterministic environment: constant demand of 5 units/day."""
    return make_synthetic_bundle(n_days=400, demand_fn=lambda n: np.full(n, 5.0), mean_price=10.0)


@pytest.fixture
def conftest_module():
    """Expose this module itself so tests can reuse make_synthetic_bundle
    without relying on a `tests.conftest` import path (which can collide
    with an unrelated `tests` package on sys.path)."""
    import sys

    return sys.modules[__name__]
