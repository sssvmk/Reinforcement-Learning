"""M5 loaders (read-only, in place; zips read with zipfile). Demand, prices and calendar only."""
import io
import os
import zipfile
import numpy as np
import pandas as pd

NEEDED = ["calendar.csv", "sell_prices.csv"]
SALES = ["sales_train_validation.csv", "sales_train_evaluation.csv"]


def find_files(m5_dir):
    """Recursive search; returns {filename: ('file', path) | ('zip', zippath, member)}."""
    found = {}
    if not m5_dir or not os.path.isdir(m5_dir):
        return found
    for root, _, files in os.walk(m5_dir):
        for f in files:
            p = os.path.join(root, f)
            if f.lower().endswith(".csv"):
                found.setdefault(f, ("file", p))
            elif f.lower().endswith(".zip"):
                try:
                    with zipfile.ZipFile(p) as z:
                        for m in z.namelist():
                            if m.lower().endswith(".csv"):
                                found.setdefault(os.path.basename(m), ("zip", p, m))
                except zipfile.BadZipFile:
                    pass
    return found


def status(m5_dir):
    """(ok, missing, found). Missing entries say exactly which file and folder."""
    found = find_files(m5_dir)
    missing = [f"{f} (not found under {m5_dir})" for f in NEEDED if f not in found]
    if not any(s in found for s in SALES):
        missing.append(f"sales_train_validation.csv or sales_train_evaluation.csv (not found under {m5_dir})")
    return not missing, missing, found


def _read(loc, **kw):
    if loc[0] == "file":
        return pd.read_csv(loc[1], **kw)
    with zipfile.ZipFile(loc[1]) as z, z.open(loc[2]) as fh:
        return pd.read_csv(io.BytesIO(fh.read()), **kw)


def read_calendar(found):
    return _read(found["calendar.csv"])


def read_sales(found, ids=None):
    """Wide sales; sales cast to int16. If ids is given, only those rows are kept."""
    name = next(s for s in SALES if s in found)
    head = _read(found[name], nrows=1)
    dt = {c: "int16" for c in head.columns if c.startswith("d_")}
    if ids is None:
        return _read(found[name], dtype=dt)
    return pd.concat([ch[ch["id"].isin(ids)] for ch in _read(found[name], dtype=dt, chunksize=2000)])


def profile_candidates(found):
    """Per series: mean, zero fraction (after trimming leading zeros), CV, weekday seasonality, leading zeros."""
    sales = read_sales(found)
    cal = read_calendar(found)
    dcols = [c for c in sales.columns if c.startswith("d_")]
    wday = cal.set_index("d").loc[dcols, "wday"].to_numpy()
    Y = sales[dcols].to_numpy(dtype=np.float32)
    first = (Y > 0).argmax(axis=1)
    rows = []
    for i in range(len(Y)):
        y = Y[i, first[i]:]
        m = y.mean()
        if len(y) < 500 or m <= 0:
            continue
        wm = np.array([y[wday[first[i]:] == k].mean() for k in range(1, 8)])
        rows.append((sales["id"].iat[i], sales["cat_id"].iat[i], sales["state_id"].iat[i], m, float((y == 0).mean()),
                     float(y.std() / m), float(wm.max() / max(wm.min(), 1e-9)), int(first[i]), len(y)))
    return pd.DataFrame(rows, columns=["id", "cat", "state", "mean", "zero_frac", "cv", "wday_ratio", "lead_zeros", "days"])


def choose_series(prof):
    """1 primary (mid-volume FOODS, low zero fraction), 1 intermittent, 1 strongly weekday-seasonal."""
    foods = prof[(prof.cat == "FOODS") & (prof.days >= 1500) & (prof.zero_frac < 0.2)]
    mid = foods[(foods["mean"] >= foods["mean"].quantile(0.5)) & (foods["mean"] <= foods["mean"].quantile(0.8))]
    primary = mid.iloc[(mid["cv"] - mid["cv"].median()).abs().argsort()].iloc[0]["id"]
    rest = prof[prof.id != primary]
    inter = rest[(rest.zero_frac > 0.5) & (rest["mean"] > 0.5) & (rest.days >= 1500)].sort_values("mean").iloc[-1]["id"]
    seas = rest[(rest.id != inter) & (rest.zero_frac < 0.3) & (rest["mean"] > 1) & (rest.days >= 1500)].sort_values("wday_ratio").iloc[-1]["id"]
    return {"primary": primary, "intermittent": inter, "seasonal": seas}


def load_series(found, sid, cache_dir=None):
    """One daily frame per series: sales, price (weekly price forward-filled <= 2 weeks), calendar flags; leading zeros trimmed.

    `price_ok` is False where the price is still missing (gaps > 2 weeks / not sold); windows must avoid those days.
    """
    cache = os.path.join(cache_dir, f"{sid}.pkl") if cache_dir else None
    if cache and os.path.exists(cache):
        return pd.read_pickle(cache)
    row = read_sales(found, ids={sid})
    row = row[row["id"] == sid].iloc[0]
    cal = read_calendar(found)
    dcols = [c for c in row.index if c.startswith("d_")]
    df = pd.DataFrame({"d": dcols, "sales": row[dcols].astype(np.int32).to_numpy()}).merge(cal, on="d", how="left")
    item, store, state = row["item_id"], row["store_id"], row["state_id"]
    parts = []
    dtypes = {"store_id": "category", "item_id": "category", "wm_yr_wk": "int32", "sell_price": "float32"}
    for ch in _read(found["sell_prices.csv"], dtype=dtypes, chunksize=1_000_000):
        parts.append(ch[(ch["store_id"].astype(str) == store) & (ch["item_id"].astype(str) == item)])
    pr = pd.concat(parts)[["wm_yr_wk", "sell_price"]]
    wk = pd.DataFrame({"wm_yr_wk": np.sort(cal["wm_yr_wk"].unique())}).merge(pr, on="wm_yr_wk", how="left")
    wk["sell_price"] = wk["sell_price"].ffill(limit=2)
    df = df.merge(wk, on="wm_yr_wk", how="left")
    first = int(np.argmax(df["sales"].to_numpy() > 0))
    df = df.iloc[first:].reset_index(drop=True)
    df["snap"] = df[f"snap_{state}"].astype(float)
    df["event"] = df["event_name_1"].notna().astype(float)
    df["price_ok"] = df["sell_price"].notna()
    df["price"] = df["sell_price"].ffill().bfill()
    df["wday0"] = (df["wday"] - 1).astype(int)
    out = df[["date", "sales", "price", "price_ok", "wday0", "event", "snap"]].copy()
    if cache:
        os.makedirs(cache_dir, exist_ok=True)
        out.to_pickle(cache)
    return out
