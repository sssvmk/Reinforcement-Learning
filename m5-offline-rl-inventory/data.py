"""Data access: M5 discovery (read-only), per-series loading, splits."""
import io
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).parent


def load_config(path=None):
    cfg = yaml.safe_load(Path(path or ROOT / "config.yaml").read_text(encoding="utf-8"))
    cfg["m5_dir"] = os.environ.get("M5_DIR", cfg["m5_dir"])
    return cfg


def find_file(m5_dir, name):
    """Case-insensitive recursive search; returns Path or (zip Path, member). Raises with searched path."""
    m5_dir = Path(m5_dir)
    if not m5_dir.exists():
        raise FileNotFoundError(f"M5 directory not found: {m5_dir}")
    low = name.lower()
    for p in m5_dir.rglob("*"):
        if p.is_file() and p.name.lower() == low:
            return p
    for z in m5_dir.rglob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            for m in zf.namelist():
                if Path(m).name.lower() == low:
                    return (z, m)
    raise FileNotFoundError(f"Required file '{name}' not found (searched recursively, incl. zips, in: {m5_dir})")


def read_csv_any(loc, **kw):
    if isinstance(loc, tuple):
        with zipfile.ZipFile(loc[0]) as zf, zf.open(loc[1]) as f:
            return pd.read_csv(f, **kw)
    return pd.read_csv(loc, **kw)


@dataclass
class Series:
    """Processed single item-store series (trimmed at first sale)."""
    sid: str
    sales: np.ndarray       # int, daily demand (post-launch)
    price: np.ndarray       # float, forward-filled sell price (nan where gap > max_gap)
    wday: np.ndarray        # 0..6
    event: np.ndarray       # 0/1
    snap: np.ndarray        # 0/1 for the series' state
    valid: np.ndarray       # bool: price available (after ffill)
    dates: np.ndarray

    def __len__(self):
        return len(self.sales)


def _sales_file(m5_dir):
    for n in ["sales_train_validation.csv", "sales_train_evaluation.csv"]:
        try:
            return find_file(m5_dir, n)
        except FileNotFoundError:
            continue
    raise FileNotFoundError(f"Neither sales_train_validation.csv nor sales_train_evaluation.csv found in {m5_dir}")


def load_series(cfg, sid, cache=True):
    out = Path(cfg["out_dir"]) if Path(cfg["out_dir"]).is_absolute() else ROOT / cfg["out_dir"]
    cpath = out / "series" / f"{sid}.npz"
    if cache and cpath.exists():
        z = np.load(cpath, allow_pickle=True)
        return Series(sid, **{k: z[k] for k in ["sales", "price", "wday", "event", "snap", "valid", "dates"]})
    m5 = cfg["m5_dir"]
    sf = _sales_file(m5)
    # read only the needed row: iterate in chunks
    row = None
    for ch in _chunks(sf, 2000):
        hit = ch[ch["id"] == sid]
        if len(hit):
            row = hit.iloc[0]
            break
    if row is None:
        raise KeyError(f"series id {sid} not in {sf}")
    dcols = [c for c in row.index if c.startswith("d_")]
    cal = read_csv_any(find_file(m5, "calendar.csv"))
    cal = cal.set_index("d").loc[dcols].reset_index()
    sales = row[dcols].to_numpy(dtype=np.int16)
    store, item, state = row["store_id"], row["item_id"], row["state_id"]
    wk = cal["wm_yr_wk"].to_numpy()
    pr = _prices_for(m5, store, item)
    s = pd.Series(pr).reindex(range(wk.min(), wk.max() + 1))
    weeks = pd.Series(range(wk.min(), wk.max() + 1))
    s = s.ffill(limit=cfg["env"]["max_gap_ffill_weeks"])
    price = s.reindex(wk).to_numpy(dtype=np.float64)
    first = int((sales > 0).argmax())
    sl = slice(first, None)
    ev = (cal["event_name_1"].notna() | cal["event_name_2"].notna()).to_numpy().astype(np.int8)
    ser = Series(sid, sales[sl].astype(np.int32), price[sl], cal["wday"].to_numpy()[sl] - 1, ev[sl],
                 cal[f"snap_{state}"].to_numpy()[sl].astype(np.int8), ~np.isnan(price[sl]),
                 cal["date"].to_numpy()[sl].astype(str))
    if cache:
        cpath.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cpath, sales=ser.sales, price=ser.price, wday=ser.wday, event=ser.event, snap=ser.snap,
                 valid=ser.valid, dates=ser.dates)
    return ser


def _chunks(loc, n):
    kw = dict(chunksize=n)
    if isinstance(loc, tuple):
        zf = zipfile.ZipFile(loc[0])
        f = zf.open(loc[1])
        yield from pd.read_csv(f, **kw)
    else:
        yield from pd.read_csv(loc, **kw)


def _prices_for(m5, store, item):
    loc = find_file(m5, "sell_prices.csv")
    parts = []
    for ch in _chunks(loc, 2_000_000):
        h = ch[(ch["store_id"] == store) & (ch["item_id"] == item)]
        if len(h):
            parts.append(h)
    if not parts:
        raise KeyError(f"no prices for {store}/{item}")
    p = pd.concat(parts)
    return p.set_index("wm_yr_wk")["sell_price"]


def split_bounds(n, cfg):
    a = int(n * cfg["splits"]["train"])
    b = int(n * (cfg["splits"]["train"] + cfg["splits"]["val"]))
    return {"train": (0, a), "val": (a, b), "test": (b, n)}
