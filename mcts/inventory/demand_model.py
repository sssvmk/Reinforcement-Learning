"""Demand models for the MCTS planner. The planner must NOT see true future
demand -- these models are fit on the TRAINING period only and sample
(or return a deterministic mean) conditioned on day features (weekday,
event flag, SNAP flag).

Three options (switchable):
  - "bootstrap": empirical bootstrap by day-of-week (optionally also
    conditioning on event/SNAP), sampling historical training-period values.
  - "negbin": Negative Binomial fitted per weekday by method of moments
    (mean/variance), sampled via numpy.
  - "deterministic_mean": returns the historical mean for that weekday (an
    ablation showing why stochastic planning matters -- ignores variance
    entirely).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class DayFeatures:
    wday: int           # 1..7
    event_flag: int = 0
    snap: int = 0


class DemandModel:
    kind = "base"

    def fit(self, train_df: pd.DataFrame) -> "DemandModel":
        raise NotImplementedError

    def sample(self, day_features: DayFeatures, rng: np.random.Generator) -> float:
        raise NotImplementedError

    def mean(self, day_features: DayFeatures) -> float:
        raise NotImplementedError


class BootstrapDemandModel(DemandModel):
    """Empirical bootstrap by weekday, optionally conditioned further on
    SNAP (since SNAP measurably lifts FOODS demand) when enough samples
    exist per (weekday, snap) cell; falls back to the weekday-only pool."""
    kind = "bootstrap"

    def __init__(self, condition_on_snap: bool = True, min_cell_size: int = 15):
        self.condition_on_snap = condition_on_snap
        self.min_cell_size = min_cell_size
        self._by_wday: Dict[int, np.ndarray] = {}
        self._by_wday_snap: Dict[tuple, np.ndarray] = {}

    def fit(self, train_df: pd.DataFrame) -> "BootstrapDemandModel":
        for wday, g in train_df.groupby("wday"):
            self._by_wday[int(wday)] = g["sales"].to_numpy(dtype=float)
        if self.condition_on_snap:
            for (wday, snap), g in train_df.groupby(["wday", "snap"]):
                if len(g) >= self.min_cell_size:
                    self._by_wday_snap[(int(wday), int(snap))] = g["sales"].to_numpy(dtype=float)
        return self

    def sample(self, day_features: DayFeatures, rng: np.random.Generator) -> float:
        key = (day_features.wday, day_features.snap)
        pool = self._by_wday_snap.get(key)
        if pool is None:
            pool = self._by_wday.get(day_features.wday)
        if pool is None or len(pool) == 0:
            return 0.0
        return float(rng.choice(pool))

    def mean(self, day_features: DayFeatures) -> float:
        key = (day_features.wday, day_features.snap)
        pool = self._by_wday_snap.get(key, self._by_wday.get(day_features.wday))
        return float(np.mean(pool)) if pool is not None and len(pool) else 0.0


class NegBinDemandModel(DemandModel):
    """Negative Binomial fitted per weekday by method of moments:
    for mean m and variance v > m, r = m^2/(v-m), p = r/(r+m). If v <= m
    (under-dispersed), falls back to a Poisson(m) sample as a degenerate
    case (NegBin is undefined for v <= m)."""
    kind = "negbin"

    def __init__(self):
        self._params: Dict[int, tuple] = {}  # wday -> (r, p) or ("poisson", m)

    def fit(self, train_df: pd.DataFrame) -> "NegBinDemandModel":
        for wday, g in train_df.groupby("wday"):
            x = g["sales"].to_numpy(dtype=float)
            m, v = x.mean(), x.var(ddof=1) if len(x) > 1 else 0.0
            if v > m > 0:
                r = m ** 2 / (v - m)
                p = r / (r + m)
                self._params[int(wday)] = ("nbinom", r, p)
            else:
                self._params[int(wday)] = ("poisson", max(m, 1e-6))
        return self

    def sample(self, day_features: DayFeatures, rng: np.random.Generator) -> float:
        params = self._params.get(day_features.wday)
        if params is None:
            return 0.0
        if params[0] == "nbinom":
            _, r, p = params
            return float(rng.negative_binomial(r, p))
        else:
            _, m = params
            return float(rng.poisson(m))

    def mean(self, day_features: DayFeatures) -> float:
        params = self._params.get(day_features.wday)
        if params is None:
            return 0.0
        if params[0] == "nbinom":
            _, r, p = params
            return r * (1 - p) / p
        else:
            return params[1]


class DeterministicMeanDemandModel(DemandModel):
    """Ablation: ignores variance, always 'samples' the historical mean for
    that weekday. Shows why stochastic planning matters when compared to
    the stochastic models above."""
    kind = "deterministic_mean"

    def __init__(self):
        self._mean_by_wday: Dict[int, float] = {}

    def fit(self, train_df: pd.DataFrame) -> "DeterministicMeanDemandModel":
        self._mean_by_wday = train_df.groupby("wday")["sales"].mean().to_dict()
        self._mean_by_wday = {int(k): float(v) for k, v in self._mean_by_wday.items()}
        return self

    def sample(self, day_features: DayFeatures, rng: np.random.Generator) -> float:
        return self._mean_by_wday.get(day_features.wday, 0.0)

    def mean(self, day_features: DayFeatures) -> float:
        return self._mean_by_wday.get(day_features.wday, 0.0)


def make_demand_model(kind: str, train_df: pd.DataFrame) -> DemandModel:
    if kind == "bootstrap":
        return BootstrapDemandModel().fit(train_df)
    elif kind == "negbin":
        return NegBinDemandModel().fit(train_df)
    elif kind == "deterministic_mean":
        return DeterministicMeanDemandModel().fit(train_df)
    else:
        raise ValueError(f"Unknown demand model kind: {kind!r}")
