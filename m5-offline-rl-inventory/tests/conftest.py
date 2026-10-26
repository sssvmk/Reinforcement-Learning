import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import Series, load_config  # noqa: E402


def make_series(n=1000, mean=10.0, seed=0, constant=None, price=2.0):
    rng = np.random.default_rng(seed)
    wday = np.arange(n) % 7
    if constant is not None:
        sales = np.full(n, constant, dtype=int)
    else:
        sales = rng.poisson(mean * (1 + 0.3 * np.sin(2 * np.pi * wday / 7)))
    return Series("synthetic", sales.astype(int), np.full(n, price), wday, np.zeros(n, int),
                  (rng.random(n) < 0.3).astype(int), np.ones(n, bool), np.array([str(i) for i in range(n)]))


@pytest.fixture
def cfg():
    c = load_config()
    c["out_dir"] = str(ROOT / "outputs" / "_pytest")
    return c


@pytest.fixture
def series():
    return make_series()
