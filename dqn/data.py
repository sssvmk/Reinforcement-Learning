"""
data.py -- load, melt, and merge the raw M5 Forecasting - Accuracy files into
per-series (item-store) daily panels, and produce chronological
train/val/test splits.

Step-0 schema notes (see README.md "Data inspection" section for the full
write-up):
  * Both sales_train_validation.csv (d_1..d_1913) and
    sales_train_evaluation.csv (d_1..d_1941) are present. We use the
    *evaluation* file because it has the longer history (more days to train
    / validate / test on). If only the validation file is present, we fall
    back to it automatically.
  * calendar.csv covers d_1..d_1969 (28 extra future days used only for the
    original forecasting competition's submission horizon); we simply never
    index past the last day present in the sales file.
  * sell_prices.csv has no NaN cells, but it is *missing rows* for
    (store_id, item_id, wm_yr_wk) combinations where the item was not yet
    (or no longer) sold in that store/week. We treat such weeks as "item
    unavailable": price is forward/back filled for continuity in the state
    features, but the usable (orderable) day range for a series is restricted
    to the contiguous span of days where a real priced row exists (first
    priced week .. last priced week). This avoids training the agent to order
    an item during a period when Walmart itself was not selling it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

STATE_SNAP_COL = {"CA": "snap_CA", "TX": "snap_TX", "WI": "snap_WI"}


def resolve_sales_file(data_dir: str, preferred: str = "sales_train_evaluation.csv") -> str:
    """Return the path to whichever sales file is present, preferring `preferred`."""
    candidates = [preferred, "sales_train_evaluation.csv", "sales_train_validation.csv"]
    for name in candidates:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"No sales_train_*.csv file found in {data_dir}. Looked for: {candidates}"
    )


def load_calendar(data_dir: str, calendar_file: str = "calendar.csv") -> pd.DataFrame:
    cal = pd.read_csv(os.path.join(data_dir, calendar_file), parse_dates=["date"])
    return cal


def load_sell_prices(data_dir: str, prices_file: str = "sell_prices.csv") -> pd.DataFrame:
    return pd.read_csv(os.path.join(data_dir, prices_file))


def load_sales(data_dir: str, sales_file: str = "sales_train_evaluation.csv") -> pd.DataFrame:
    path = resolve_sales_file(data_dir, sales_file)
    return pd.read_csv(path)


@dataclass
class SeriesBundle:
    """All information required to simulate one item-store series."""

    series_id: str
    item_id: str
    store_id: str
    state_id: str
    dates: np.ndarray            # datetime64[D], length = n_valid_days
    demand: np.ndarray           # float, units sold (will be int-valued)
    sell_price: np.ndarray       # float, forward/back-filled for continuity
    price_available: np.ndarray  # bool, True where a real (non-filled) price row existed
    wday: np.ndarray             # int 1..7
    event_flag: np.ndarray       # 0/1, 1 if any event_name_{1,2} is non-null
    snap_flag: np.ndarray        # 0/1, state-specific SNAP flag
    mean_demand: float
    std_demand: float
    mean_price: float
    train_range: tuple
    val_range: tuple
    test_range: tuple

    def n_days(self) -> int:
        return len(self.demand)


def build_series_bundle(
    series_id: str,
    sales_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> SeriesBundle:
    """Build a long daily panel for a single series id and compute temporal splits."""
    row = sales_df.loc[sales_df["id"] == series_id]
    if len(row) == 0:
        raise KeyError(f"series id {series_id!r} not found in sales file")
    row = row.iloc[0]

    item_id = row["item_id"]
    store_id = row["store_id"]
    state_id = row["state_id"]

    d_cols = [c for c in sales_df.columns if c.startswith("d_")]
    demand_long = row[d_cols].astype(float).values  # shape (n_days_total,)

    cal = calendar_df.iloc[: len(d_cols)].copy()  # calendar may extend past sales horizon
    cal = cal.set_index("d").loc[d_cols].reset_index()

    snap_col = STATE_SNAP_COL[state_id]

    panel = pd.DataFrame(
        {
            "d": d_cols,
            "date": cal["date"].values,
            "wm_yr_wk": cal["wm_yr_wk"].values,
            "wday": cal["wday"].values,
            "snap": cal[snap_col].values,
            "event1": cal["event_name_1"].values,
            "event2": cal["event_name_2"].values,
            "demand": demand_long,
        }
    )
    panel["event_flag"] = (
        panel["event1"].notna() | panel["event2"].notna()
    ).astype(int)

    item_prices = prices_df.loc[
        (prices_df["store_id"] == store_id) & (prices_df["item_id"] == item_id),
        ["wm_yr_wk", "sell_price"],
    ].drop_duplicates(subset="wm_yr_wk")

    panel = panel.merge(item_prices, on="wm_yr_wk", how="left")
    panel["price_available"] = panel["sell_price"].notna()

    # Document assumption: forward-fill then back-fill price for state continuity,
    # but restrict the *usable* day range to the contiguous priced span below.
    panel["sell_price_filled"] = panel["sell_price"].ffill().bfill()

    priced_idx = np.where(panel["price_available"].values)[0]
    if len(priced_idx) == 0:
        raise ValueError(f"series {series_id!r} has no priced weeks at all; cannot simulate")
    first_priced, last_priced = priced_idx[0], priced_idx[-1]
    usable = panel.iloc[first_priced : last_priced + 1].reset_index(drop=True)

    n = len(usable)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    train_range = (0, n_train)
    val_range = (n_train, n_train + n_val)
    test_range = (n_train + n_val, n)

    demand = usable["demand"].values.astype(float)
    sell_price = usable["sell_price_filled"].values.astype(float)
    mean_price = float(np.nanmean(sell_price[: n_train])) if n_train > 0 else float(np.nanmean(sell_price))
    mean_demand = float(np.mean(demand[:n_train])) if n_train > 0 else float(np.mean(demand))
    std_demand = float(np.std(demand[:n_train])) if n_train > 0 else float(np.std(demand))

    return SeriesBundle(
        series_id=series_id,
        item_id=item_id,
        store_id=store_id,
        state_id=state_id,
        dates=usable["date"].values,
        demand=demand,
        sell_price=sell_price,
        price_available=usable["price_available"].values,
        wday=usable["wday"].values.astype(int),
        event_flag=usable["event_flag"].values.astype(int),
        snap_flag=usable["snap"].values.astype(int),
        mean_demand=mean_demand,
        std_demand=std_demand,
        mean_price=mean_price,
        train_range=train_range,
        val_range=val_range,
        test_range=test_range,
    )


def load_all(config: dict) -> dict:
    """Convenience loader: returns {series_id: SeriesBundle} for config['series_ids']."""
    dcfg = config["data"]
    sales = load_sales(dcfg["data_dir"], dcfg.get("sales_file", "sales_train_evaluation.csv"))
    calendar = load_calendar(dcfg["data_dir"], dcfg.get("calendar_file", "calendar.csv"))
    prices = load_sell_prices(dcfg["data_dir"], dcfg.get("prices_file", "sell_prices.csv"))

    bundles = {}
    for sid in config["series_ids"]:
        bundles[sid] = build_series_bundle(
            sid,
            sales,
            calendar,
            prices,
            train_frac=dcfg.get("train_frac", 0.70),
            val_frac=dcfg.get("val_frac", 0.15),
            test_frac=dcfg.get("test_frac", 0.15),
        )
    return bundles
