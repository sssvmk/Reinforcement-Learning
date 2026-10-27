"""Data loading and cleaning. Source files are only ever read, never written."""
import os
import zipfile
import numpy as np
import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "outputs")
TZ = "Europe/Madrid"
KELVIN = 273.15


def load_config(path=None):
    path = path or os.path.join(ROOT, "config.yaml")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["energy_file"] = os.environ.get("ENERGY_FILE", cfg["energy_file"])
    cfg["weather_file"] = os.environ.get("WEATHER_FILE", cfg["weather_file"])
    return cfg


def _read_csv(path):
    if path.lower().endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            name = [n for n in z.namelist() if n.endswith(".csv")][0]
            with z.open(name) as f:
                return pd.read_csv(f)
    return pd.read_csv(path)


def read_energy_raw(path):
    e = _read_csv(path)
    e["time"] = pd.to_datetime(e["time"], utc=True)
    return e


def read_weather_raw(path):
    w = _read_csv(path)
    w["dt_iso"] = pd.to_datetime(w["dt_iso"], utc=True)
    w["city_name"] = w["city_name"].str.strip()
    return w


def clean_weather(w):
    """Remove city-hour duplicates deterministically (keep first), convert K to C."""
    w = w.sort_values(["city_name", "dt_iso"], kind="stable")
    w = w.drop_duplicates(["city_name", "dt_iso"], keep="first").copy()
    for c in ["temp", "temp_min", "temp_max"]:
        w[c] = w[c] - KELVIN
    return w


def weather_city_mean(w):
    """Hourly city-average weather (Celsius), indexed by UTC time."""
    cols = ["temp", "humidity", "wind_speed", "clouds_all", "pressure"]
    return w.groupby("dt_iso")[cols].mean()


def clean_energy(e):
    """Index by UTC time, drop empty/constant columns, interpolate short gaps."""
    e = e.sort_values("time").drop_duplicates("time", keep="first").set_index("time")
    num = e.select_dtypes("number")
    drop = [c for c in num.columns if num[c].nunique(dropna=True) <= 1]
    e = e.drop(columns=drop)
    full = pd.date_range(e.index.min(), e.index.max(), freq="h")
    e = e.reindex(full)
    e = e.interpolate(method="time", limit=6, limit_direction="both")
    return e, drop


def spanish_holidays(years):
    from dateutil.easter import easter
    out = set()
    for y in years:
        for m, d in [(1, 1), (1, 6), (5, 1), (8, 15), (10, 12), (11, 1), (12, 6), (12, 8), (12, 25)]:
            out.add(pd.Timestamp(y, m, d))
        out.add(pd.Timestamp(easter(y)) - pd.Timedelta(days=2))
    return out


SHORT = {
    "total load forecast": "Lhat", "total load actual": "L",
    "price day ahead": "pDA", "price actual": "pA",
    "forecast solar day ahead": "solar_fc", "forecast wind onshore day ahead": "wind_fc",
    "generation solar": "gen_solar", "generation wind onshore": "gen_wind",
}


def load_hourly(cfg=None):
    """Hourly frame indexed by UTC with local-time calendar columns and city-mean weather."""
    cfg = cfg or load_config()
    e, dropped = clean_energy(read_energy_raw(cfg["energy_file"]))
    w = weather_city_mean(clean_weather(read_weather_raw(cfg["weather_file"])))
    df = e.join(w, how="left").interpolate(method="time", limit=6, limit_direction="both")
    loc = df.index.tz_convert(TZ)
    df["date"] = pd.DatetimeIndex(loc.date)
    df["hour_local"] = loc.hour
    df["year"] = loc.year
    df.attrs["dropped"] = dropped
    return df


def day_arrays(df, cfg=None):
    """Stack hourly data into (D, 24) arrays per local delivery day.
    On 25-hour days the first repeated hour is kept; on 23-hour days the missing hour is interpolated."""
    cfg = cfg or load_config()
    cols = {"total load forecast": "Lhat", "total load actual": "L", "price day ahead": "pDA",
            "price actual": "pA", "forecast solar day ahead": "solar_fc",
            "forecast wind onshore day ahead": "wind_fc", "temp": "temp",
            "generation solar": "gen_solar", "generation wind onshore": "gen_wind"}
    A = {"scale": cfg["customer_avg_load_mw"] / df["total load actual"].mean()}
    for col, nm in cols.items():
        pv = df.pivot_table(index="date", columns="hour_local", values=col, aggfunc="first")
        pv = pv.reindex(columns=range(24)).interpolate(axis=1, limit_direction="both")
        A[nm] = pv.to_numpy()
        A["dates"] = pd.DatetimeIndex(pv.index)
    return A


def split_masks(dates, cfg=None):
    cfg = cfg or load_config()
    y = np.asarray(dates.year)
    return {k: np.isin(y, cfg["splits"][k]) for k in ["train", "val", "test"]}
