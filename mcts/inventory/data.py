"""Loads one M5 item-store series into a clean daily DataFrame for the
inventory simulator. Reads ONLY from M5_DIR (read-only); never writes there.
Caches the processed result under outputs/series/<id>.parquet (falls back to
.csv if pyarrow/parquet is not available) so repeated runs skip re-scanning
the large CSVs.

See inventory/DATA_NOTES.md for the full Step 0 findings this module
implements: both sales files exist locally (we use the evaluation file),
missing-price weeks are forward-filled for gaps <= N weeks (else trimmed),
and leading pre-launch zeros are trimmed.
"""
from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

STATE_FROM_STORE_PREFIX = {
    "CA": "CA", "TX": "TX", "WI": "WI",
}


def _load_config() -> dict:
    cfg_path = Path(__file__).resolve().parent.parent / "config.yaml"
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def _resolve_m5_dir(m5_dir: Optional[str]) -> Path:
    if m5_dir is not None:
        return Path(m5_dir)
    env_override = os.environ.get("M5_DIR")
    if env_override:
        return Path(env_override)
    cfg = _load_config()
    return Path(cfg["paths"]["m5_dir"])


def _resolve_outputs_dir(outputs_dir: Optional[str]) -> Path:
    if outputs_dir is not None:
        return Path(outputs_dir)
    cfg = _load_config()
    root = Path(__file__).resolve().parent.parent
    return (root / cfg["paths"]["outputs_dir"]).resolve()


def _find_file(m5_dir: Path, names) -> Path:
    """Recursively searches m5_dir for any of `names` (case-insensitive).
    Also looks inside any .zip archives found, WITHOUT extracting them to
    disk (read via zipfile directly). Raises FileNotFoundError naming the
    exact path searched if nothing is found."""
    names_lower = {n.lower() for n in names}
    if not m5_dir.exists():
        raise FileNotFoundError(f"M5_DIR does not exist: {m5_dir}")
    for p in m5_dir.rglob("*"):
        if p.is_file() and p.name.lower() in names_lower:
            return p
    for p in m5_dir.rglob("*.zip"):
        with zipfile.ZipFile(p) as zf:
            for member in zf.namelist():
                if Path(member).name.lower() in names_lower:
                    return p  # caller handles zip-member extraction via _open_in_m5
    raise FileNotFoundError(
        f"None of {sorted(names)} found under {m5_dir} (searched recursively, "
        f"including .zip archives)."
    )


def _open_csv_in_m5(m5_dir: Path, names, **read_csv_kwargs) -> pd.DataFrame:
    """Opens a CSV, possibly nested inside a zip, read-only."""
    path = _find_file(m5_dir, names)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            names_lower = {n.lower() for n in names}
            member = next(m for m in zf.namelist() if Path(m).name.lower() in names_lower)
            with zf.open(member, "r") as fh:
                return pd.read_csv(fh, **read_csv_kwargs)
    return pd.read_csv(path, **read_csv_kwargs)


def _sales_file_names():
    # Prefer evaluation (superset in time); fall back to validation if the
    # evaluation file is not present (see DATA_NOTES.md correction: in this
    # dataset BOTH are present; we still support either).
    return ["sales_train_evaluation.csv"], ["sales_train_validation.csv"]


def _load_calendar(m5_dir: Path) -> pd.DataFrame:
    cal = _open_csv_in_m5(m5_dir, ["calendar.csv"])
    cal["date"] = pd.to_datetime(cal["date"])
    for c in ("snap_CA", "snap_TX", "snap_WI"):
        cal[c] = cal[c].astype("int8")
    cal["wday"] = cal["wday"].astype("int16")
    cal["month"] = cal["month"].astype("int16")
    cal["year"] = cal["year"].astype("int16")
    return cal


def _load_one_series_row(m5_dir: Path, series_id: str) -> pd.Series:
    """Reads only the row for `series_id` from the (large) sales file,
    without loading the whole file into memory as long-format data."""
    eval_names, val_names = _sales_file_names()
    try:
        path = _find_file(m5_dir, eval_names)
        sales_names = eval_names
    except FileNotFoundError:
        path = _find_file(m5_dir, val_names)
        sales_names = val_names

    # Read just the header first to know the d_* columns and dtypes.
    header = _open_csv_in_m5(m5_dir, sales_names, nrows=0)
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    d_cols = [c for c in header.columns if c.startswith("d_")]
    dtype = {c: "int16" for c in d_cols}

    # pandas can filter rows only after reading; for a single id this is a
    # full scan, but we downcast dtypes and only keep the matching row to
    # bound memory. (Caching below avoids repeating this across runs.)
    chunks = []
    for chunk in pd.read_csv(path if path.suffix.lower() != ".zip" else _zip_reader(path, sales_names),
                              dtype=dtype, usecols=id_cols + d_cols, chunksize=500_000):
        match = chunk[chunk["id"] == series_id]
        if not match.empty:
            chunks.append(match)
            break
    if not chunks:
        raise ValueError(f"series_id {series_id!r} not found in {path}")
    return chunks[0].iloc[0]


def _zip_reader(zip_path: Path, names):
    names_lower = {n.lower() for n in names}
    zf = zipfile.ZipFile(zip_path)
    member = next(m for m in zf.namelist() if Path(m).name.lower() in names_lower)
    return zf.open(member, "r")


def _load_prices_for(m5_dir: Path, store_id: str, item_id: str) -> pd.DataFrame:
    """Streams sell_prices.csv in chunks, filtering to (store_id, item_id)
    immediately, so the full 6.8M-row file is never held in memory at once."""
    path = _find_file(m5_dir, ["sell_prices.csv"])
    dtype = {"store_id": "category", "item_id": "category", "wm_yr_wk": "int32", "sell_price": "float32"}
    reader_src = path if path.suffix.lower() != ".zip" else _zip_reader(path, ["sell_prices.csv"])
    frames = []
    for chunk in pd.read_csv(reader_src, dtype=dtype, chunksize=1_000_000):
        sub = chunk[(chunk["store_id"] == store_id) & (chunk["item_id"] == item_id)]
        if not sub.empty:
            frames.append(sub)
    if not frames:
        return pd.DataFrame(columns=["store_id", "item_id", "wm_yr_wk", "sell_price"])
    return pd.concat(frames, ignore_index=True)


def _ffill_short_gaps(df: pd.DataFrame, price_col: str, max_gap_weeks: int) -> pd.DataFrame:
    """Forward-fills `price_col` across gaps of at most `max_gap_weeks`
    consecutive missing weeks; leaves longer gaps as NaN (caller trims
    those out of episode windows). Operates on a day-indexed, date-sorted
    frame with one row per day."""
    df = df.copy()
    is_na = df[price_col].isna()
    # identify run-lengths of consecutive NaNs
    group = (~is_na).cumsum()
    run_len = is_na.groupby(group).transform("sum")
    filled = df[price_col].ffill()
    # a gap is "short" if its run length in DAYS is <= max_gap_weeks*7
    short_gap = is_na & (run_len <= max_gap_weeks * 7)
    df[price_col] = np.where(short_gap, filled, df[price_col])
    return df


def load_series(series_id: str, m5_dir: Optional[str] = None,
                 outputs_dir: Optional[str] = None,
                 max_price_gap_weeks_ffill: int = 2,
                 use_cache: bool = True) -> pd.DataFrame:
    """Returns one row per day for `series_id` with columns:
    d, date, sales, sell_price, wday, weekday, month, event_flag, snap.

    - Trims leading pre-launch zero-sales days (before the first nonzero
      sale).
    - Missing weekly prices: forward-filled for gaps <= `max_price_gap_weeks_ffill`
      weeks; longer gaps are left as NaN rows which callers (inventory_env)
      must exclude from episode windows (documented in DATA_NOTES.md).
    - Reads only from `m5_dir` (read-only); writes only to `outputs_dir/series/`.
    """
    m5_path = _resolve_m5_dir(m5_dir)
    out_path = _resolve_outputs_dir(outputs_dir)
    cache_dir = out_path / "series"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{series_id}.parquet"

    if use_cache and cache_file.exists():
        try:
            return pd.read_parquet(cache_file)
        except Exception:
            cache_file_csv = cache_dir / f"{series_id}.csv"
            if cache_file_csv.exists():
                df = pd.read_csv(cache_file_csv, parse_dates=["date"])
                return df

    row = _load_one_series_row(m5_path, series_id)
    store_id = row["store_id"]
    item_id = row["item_id"]
    state_id = row["state_id"]
    d_cols = [c for c in row.index if c.startswith("d_")]
    sales = row[d_cols].to_numpy(dtype="int16")

    calendar = _load_calendar(m5_path)
    cal_sub = calendar[calendar["d"].isin(d_cols)].copy()
    cal_sub["sales"] = cal_sub["d"].map(dict(zip(d_cols, sales))).astype("int16")
    cal_sub = cal_sub.sort_values("date").reset_index(drop=True)

    prices = _load_prices_for(m5_path, store_id, item_id)
    cal_sub = cal_sub.merge(prices[["wm_yr_wk", "sell_price"]], on="wm_yr_wk", how="left")

    cal_sub = _ffill_short_gaps(cal_sub, "sell_price", max_price_gap_weeks_ffill)

    snap_col = f"snap_{state_id}"
    cal_sub["snap"] = cal_sub[snap_col].astype("int8")
    cal_sub["event_flag"] = ((~cal_sub["event_name_1"].isna()) | (~cal_sub["event_name_2"].isna())).astype("int8")

    out = cal_sub[["d", "date", "sales", "sell_price", "wday", "weekday", "month", "event_flag", "snap"]].copy()

    # Trim leading pre-launch zeros: before the first nonzero sale.
    nonzero_idx = np.flatnonzero(out["sales"].to_numpy() > 0)
    if len(nonzero_idx) > 0:
        out = out.iloc[nonzero_idx[0]:].reset_index(drop=True)

    out.attrs["series_id"] = series_id
    out.attrs["store_id"] = store_id
    out.attrs["item_id"] = item_id
    out.attrs["state_id"] = state_id

    try:
        out.to_parquet(cache_file, index=False)
    except Exception:
        out.to_csv(cache_dir / f"{series_id}.csv", index=False)

    return out


if __name__ == "__main__":
    import sys
    sid = sys.argv[1] if len(sys.argv) > 1 else "FOODS_3_477_CA_3_evaluation"
    df = load_series(sid)
    print(df.head())
    print(df.shape)
    print("price nulls:", df["sell_price"].isna().sum())
