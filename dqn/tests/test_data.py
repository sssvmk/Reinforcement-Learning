"""Sanity check #6: train/val/test ranges are disjoint and chronological.
Also a smoke test against the real M5 files when available (skipped otherwise,
so the suite is portable)."""
import os

import numpy as np
import pytest
import yaml

from data import build_series_bundle, load_all


def test_splits_disjoint_and_chronological(synthetic_bundle):
    b = synthetic_bundle
    assert b.train_range[0] == 0
    assert b.train_range[1] == b.val_range[0]
    assert b.val_range[1] == b.test_range[0]
    assert b.test_range[1] == b.n_days()
    # strictly increasing boundaries (non-degenerate splits)
    assert b.train_range[0] < b.train_range[1] < b.val_range[1] < b.test_range[1]


def test_series_stats_from_train_only(synthetic_bundle):
    b = synthetic_bundle
    train_slice = b.demand[b.train_range[0]:b.train_range[1]]
    assert np.isclose(b.mean_demand, np.mean(train_slice))


_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")


def _data_dir_available():
    if not os.path.exists(_CONFIG_PATH):
        return False
    with open(_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    return os.path.isdir(cfg["data"]["data_dir"])


@pytest.mark.skipif(not _data_dir_available(), reason="M5 data directory not available in this environment")
def test_real_data_loads_and_splits():
    with open(_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    bundles = load_all(cfg)
    assert len(bundles) == len(cfg["series_ids"])
    for sid, bundle in bundles.items():
        assert bundle.series_id == sid
        assert bundle.train_range[1] == bundle.val_range[0]
        assert bundle.val_range[1] == bundle.test_range[0]
        assert bundle.test_range[1] == bundle.n_days()
        assert bundle.n_days() > 0
        assert np.all(bundle.demand >= 0)
