"""Feature whitelist for day-D hourly decisions. Actuals are used only up to day D-cutoff_days."""
import numpy as np
import pandas as pd
import data

NAMES = ["hsin", "hcos", "dsin", "dcos", "msin", "mcos", "offday", "pDA_n", "Lhat_n", "solar_n", "wind_n",
         "lag_eL", "lag_g", "eL_mean7", "eL_std7", "g_mean7", "g_std7", "temp_lag", "temp_lag7",
         "pDA_rel_lag"]


def _lag_roll(x, cutoff, w, fn):
    """Statistic over days d-cutoff-w+1 .. d-cutoff (shifted so nothing after d-cutoff is used)."""
    s = pd.Series(x)
    return getattr(s.rolling(w, min_periods=1), fn)().shift(cutoff).to_numpy()


def build_features(A, cutoff_days=2, norm=None, train_mask=None):
    """Return X of shape (D, 24, F), names, norm. Entries without enough history are NaN."""
    D = len(A["dates"])
    eL = A["L"] / A["Lhat"] - 1.0
    g = A["pA"] - A["pDA"]
    c = cutoff_days
    if norm is None:
        m = np.ones(D, bool) if train_mask is None else train_mask
        norm = {"pDA": A["pDA"][m].mean(), "Lhat": A["Lhat"][m].mean(),
                "solar": A["solar_fc"][m].mean() + 1e-9, "wind": A["wind_fc"][m].mean() + 1e-9,
                "temp": A["temp"][m].mean(), "temp_sd": A["temp"][m].std()}
    dates = A["dates"]
    hrs = np.arange(24)[None, :].repeat(D, 0)
    dow = np.asarray(dates.dayofweek)[:, None].repeat(24, 1)
    mon = (np.asarray(dates.month)[:, None].repeat(24, 1) - 1)
    hol = data.spanish_holidays(sorted(set(dates.year)))
    off = np.array([(d.dayofweek >= 5) or (d in hol) for d in dates], float)[:, None].repeat(24, 1)

    def lag(x):  # same hour, cutoff days earlier
        out = np.full_like(x, np.nan, dtype=float)
        out[c:] = x[:-c]
        return out

    day_eL = eL.mean(1)
    day_g = g.mean(1)
    temp_d = A["temp"].mean(1)
    feats = [
        np.sin(2 * np.pi * hrs / 24), np.cos(2 * np.pi * hrs / 24),
        np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
        np.sin(2 * np.pi * mon / 12), np.cos(2 * np.pi * mon / 12), off,
        A["pDA"] / norm["pDA"], A["Lhat"] / norm["Lhat"],
        A["solar_fc"] / norm["solar"], A["wind_fc"] / norm["wind"],
        lag(eL), lag(g),
        _lag_roll(day_eL, c, 7, "mean")[:, None].repeat(24, 1),
        _lag_roll(day_eL, c, 7, "std")[:, None].repeat(24, 1),
        _lag_roll(day_g, c, 7, "mean")[:, None].repeat(24, 1),
        _lag_roll(day_g, c, 7, "std")[:, None].repeat(24, 1),
        ((lag(temp_d[:, None].repeat(24, 1)) - norm["temp"]) / norm["temp_sd"]),
        ((_lag_roll(temp_d, c, 7, "mean") - norm["temp"]) / norm["temp_sd"])[:, None].repeat(24, 1),
        A["pDA"] / (lag(A["pDA"].mean(1, keepdims=True).repeat(24, 1)) + 1e-9),
    ]
    X = np.stack(feats, -1)
    return X, list(NAMES), norm


def fill_nan(X, fill=0.0):
    return np.nan_to_num(X, nan=fill)
