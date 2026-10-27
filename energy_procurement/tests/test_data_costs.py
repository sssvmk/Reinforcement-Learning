import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data, costs, risk, features

CFG = data.load_config()
HAVE = os.path.exists(CFG["energy_file"]) and os.path.exists(CFG["weather_file"])
real = pytest.mark.skipif(not HAVE, reason="data files not present")


# ---------------- test 1: data ----------------
@real
def test_hourly_index_continuous_utc_and_files_untouched():
    mt = [os.path.getmtime(CFG["energy_file"]), os.path.getmtime(CFG["weather_file"])]
    df = data.load_hourly(CFG)
    assert str(df.index.tz) == "UTC"
    assert df.index.is_unique and df.index.is_monotonic_increasing
    assert (np.diff(df.index.values).astype("timedelta64[h]").astype(int) == 1).all()
    assert mt == [os.path.getmtime(CFG["energy_file"]), os.path.getmtime(CFG["weather_file"])]


def test_weather_dedupe_deterministic_and_kelvin():
    t = pd.to_datetime(["2015-01-01 00:00", "2015-01-01 00:00", "2015-01-01 01:00"], utc=True)
    w = pd.DataFrame({"dt_iso": t, "city_name": ["Madrid"] * 3, "temp": [300.0, 310.0, 273.15],
                      "temp_min": [300.0, 310.0, 273.15], "temp_max": [300.0, 310.0, 273.15]})
    a = data.clean_weather(w)
    b = data.clean_weather(w)
    assert len(a) == 2
    assert a["temp"].tolist() == [300.0 - 273.15, 0.0]
    assert a.equals(b)


@real
def test_day_arrays_shape_and_splits():
    A = data.day_arrays(data.load_hourly(CFG), CFG)
    assert A["Lhat"].shape == (len(A["dates"]), 24)
    m = data.split_masks(A["dates"], CFG)
    assert not (m["train"] & m["val"]).any() and not (m["val"] & m["test"]).any()
    d = A["dates"]
    assert d[m["train"]].max() < d[m["val"]].min() < d[m["val"]].max() < d[m["test"]].min()
    assert set(d[m["train"]].year) == {2015, 2016} and set(d[m["test"]].year) == {2018}


# ---------------- test 2: no leakage ----------------
def _synthetic_days(D=60, seed=0):
    r = np.random.default_rng(seed)
    d = pd.date_range("2016-01-01", periods=D, freq="D")
    g = lambda m, s: m + s * r.standard_normal((D, 24))
    return {"dates": d, "scale": 1.0, "Lhat": g(25000, 2000), "L": g(25000, 2000), "pDA": g(50, 5),
            "pA": g(58, 5), "solar_fc": np.abs(g(2000, 500)), "wind_fc": np.abs(g(5000, 800)),
            "temp": g(15, 5), "gen_solar": g(2000, 300), "gen_wind": g(5000, 500)}


@pytest.mark.parametrize("cutoff", [2, 3])
def test_features_unchanged_when_late_actuals_altered(cutoff):
    A = _synthetic_days()
    norm = features.build_features(A, cutoff)[2]
    X0, _, _ = features.build_features(A, cutoff, norm=norm)
    for D in (20, 35, 50):
        B = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in A.items()}
        s = D - cutoff + 1  # first day that must be invisible
        for k in ("L", "pA", "temp", "gen_solar", "gen_wind"):
            B[k][s:] += 1e3 * np.random.default_rng(1).standard_normal(B[k][s:].shape)
        X1, _, _ = features.build_features(B, cutoff, norm=norm)
        assert np.allclose(X0[D], X1[D], equal_nan=True)


# ---------------- test 3: costs ----------------
def test_hourly_cost_hand_examples():
    assert costs.hourly_cost(8, 10, 50, 60) == pytest.approx(520)
    assert costs.hourly_cost(12, 10, 50, 60, rho=0) == pytest.approx(600)
    assert costs.hourly_cost(12, 10, 50, 60, rho=0.5) == pytest.approx(540)
    assert costs.hourly_cost(8, 10, 50, 60, pi=5) == pytest.approx(530)


@pytest.mark.parametrize("rho", [0, 0.5, 1.0])
def test_cost_convex_in_q(rho):
    q = np.linspace(0, 20, 201)
    c = costs.hourly_cost(q, 10.0, 50.0, 60.0, rho=rho, pi=2.0)
    assert (np.diff(c, 2) >= -1e-9).all()


def test_breakdown_sums_to_cost():
    r = np.random.default_rng(0)
    q, L, pd_, pa = r.uniform(5, 15, (4, 24)), r.uniform(5, 15, (4, 24)), r.uniform(30, 60, (4, 24)), r.uniform(40, 70, (4, 24))
    b = costs.cost_breakdown(q, L, pd_, pa, 0.5, 3.0)
    assert np.allclose(b["purchase"] + b["shortfall"] + b["surplus_credit"], costs.daily_cost(q, L, pd_, pa, 0.5, 3.0))


# ---------------- test 4: risk ----------------
def test_cvar_ru_equals_tail_average():
    x = np.random.default_rng(0).standard_normal(1000)
    assert risk.cvar(x, 0.95) == pytest.approx(np.sort(x)[-50:].mean(), rel=1e-9)
    assert risk.cvar(x, 0.95) == pytest.approx(risk.cvar_tail_mean(x, 0.95))
    ru = [e + np.mean(np.maximum(x - e, 0)) / 0.05 for e in np.sort(x)]
    assert min(ru) == pytest.approx(risk.cvar(x, 0.95))


def test_variance_and_hedge_ratio():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert risk.var_(x) == pytest.approx(1.25)
    assert risk.mean_variance(x, 2.0) == pytest.approx(2.5 + 2 * 1.25)
    assert risk.hedge_ratio(np.array([8.0, 9.0]), np.array([10.0, 10.0])) == pytest.approx(0.85)
