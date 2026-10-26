"""M5 loader (read-only). Reads calendar.csv, sell_prices.csv and sales_train_{evaluation,validation}.csv in place,
profiles the series, picks 1 primary + 2 replication series, and caches the processed series under outputs/series/."""
from __future__ import annotations
import zipfile
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


def _open(folder: Path, name: str):
    """Return a path or an open zip member for `name` found recursively in folder (never extracts)."""
    for p in folder.rglob(name):
        if p.is_file():
            return p
    for z in folder.rglob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            for n in zf.namelist():
                if Path(n).name == name:
                    return (z, n)
    return None


def _read_csv(folder, name, **kw):
    src = _open(Path(folder), name)
    if src is None:
        raise FileNotFoundError(name)
    if isinstance(src, tuple):
        with zipfile.ZipFile(src[0]) as zf, zf.open(src[1]) as f:
            return pd.read_csv(f, **kw)
    return pd.read_csv(src, **kw)


def sales_name(folder):
    for n in ("sales_train_evaluation.csv", "sales_train_validation.csv"):
        if _open(Path(folder), n) is not None:
            return n
    raise FileNotFoundError("sales_train_*.csv")


@dataclass
class Series:
    sid: str
    store: str
    item: str
    cat: str
    dates: np.ndarray
    demand: np.ndarray      # (n,) float
    price: np.ndarray       # (n,) forward-filled (<= 2 weeks); NaN where unavailable
    valid: np.ndarray       # (n,) bool: price available
    weekday: np.ndarray     # (n,) 0..6
    event: np.ndarray       # (n,) 0/1
    snap: np.ndarray        # (n,) 0/1 for the series' state

    @property
    def N(self):
        return len(self.demand)


def load_sales_matrix(folder):
    name = sales_name(folder)
    head = _read_csv(folder, name, nrows=1)
    dcols = [c for c in head.columns if c.startswith("d_")]
    dtypes = {c: "int16" for c in dcols}
    df = _read_csv(folder, name, dtype=dtypes)
    return name, df[ID_COLS].copy(), df[dcols].to_numpy(dtype=np.int16), dcols


def profile(ids, S):
    """Per-series profile on the trimmed (from first sale) history."""
    N = S.shape[0]
    first = (S > 0).argmax(axis=1)
    rows = []
    for i in range(N):
        x = S[i, first[i]:].astype(float)
        if len(x) < 365:
            rows.append((np.nan,) * 5)
            continue
        m = x.mean()
        wd = np.array([x[k::7].mean() for k in range(7)])
        rows.append((len(x), m, float((x == 0).mean()), x.std() / m if m > 0 else np.nan,
                     wd.std() / m if m > 0 else np.nan))
    p = pd.DataFrame(rows, columns=["n", "mean", "zero_frac", "cv", "wd_amp"])
    return pd.concat([ids.reset_index(drop=True), p], axis=1)


def choose_series(prof):
    """1 primary (mid-volume FOODS, low zero fraction), 1 intermittent, 1 seasonal. Deterministic."""
    p = prof.dropna().copy()
    p = p[p.n >= 1500]
    a = p[(p.cat_id == "FOODS") & (p.zero_frac < 0.15) & (p["mean"] >= 3)]
    med = a["mean"].median()
    primary = a.iloc[(a["mean"] - med).abs().argsort(kind="stable")].iloc[0]
    b = p[(p.zero_frac >= 0.5) & (p.zero_frac <= 0.8) & (p["mean"] >= 0.3) & (p.id != primary.id)]
    interm = b.iloc[(b.zero_frac - 0.65).abs().argsort(kind="stable")].iloc[0]
    c = p[(p["mean"] >= 3) & (p["mean"] <= 15) & (p.zero_frac < 0.3) & ~p.id.isin([primary.id, interm.id])]
    seasonal = c.sort_values(["wd_amp", "id"], ascending=[False, True]).iloc[0]
    return {"primary": primary, "intermittent": interm, "seasonal": seasonal}


def build_series(folder, row, S, dcols, cal, prices_cache=None):
    """Join demand, calendar and weekly prices for one series row (a pandas Series from the profile table)."""
    sid, store, item, state = row["id"], row["store_id"], row["item_id"], row["state_id"]
    i = int(row["_idx"])
    dem = S[i].astype(float)
    first = int((dem > 0).argmax())
    cal = cal.set_index("d").loc[dcols]
    wk = cal["wm_yr_wk"].to_numpy()
    pr = prices_cache if prices_cache is not None else load_prices(folder, [(store, item)])
    pp = pr[(store, item)]                      # Series indexed by wm_yr_wk
    weeks = pd.Index(sorted(set(wk)))
    wp = pp.reindex(weeks).ffill(limit=2)       # forward-fill gaps of up to 2 weeks
    price = wp.reindex(wk).to_numpy(float)
    sl = slice(first, None)
    snap_col = f"snap_{state}"
    return Series(sid, store, item, row["cat_id"], pd.to_datetime(cal["date"]).to_numpy()[sl], dem[sl], price[sl],
                  np.isfinite(price[sl]), cal["wday"].to_numpy()[sl].astype(int) % 7,
                  cal["event_name_1"].notna().to_numpy()[sl].astype(int), cal[snap_col].to_numpy()[sl].astype(int))


def load_prices(folder, pairs):
    want = set(pairs)
    keep = []
    src = _open(Path(folder), "sell_prices.csv")
    if src is None:
        raise FileNotFoundError("sell_prices.csv")
    if isinstance(src, tuple):
        zf = zipfile.ZipFile(src[0])
        f = zf.open(src[1])
    else:
        f = open(src, "rb")
    try:
        for ch in pd.read_csv(f, chunksize=2_000_000):
            m = [(s, it) in want for s, it in zip(ch.store_id, ch.item_id)]
            keep.append(ch[np.array(m, bool)])
    finally:
        f.close()
    df = pd.concat(keep)
    return {k: g.set_index("wm_yr_wk")["sell_price"].sort_index() for k, g in df.groupby(["store_id", "item_id"])}


def load_inventory_series(cfg, out_dir):
    """Return dict name -> Series (primary, intermittent, seasonal). Caches the processed series and profile."""
    folder = cfg["paths"]["m5_dir"]
    out_dir = Path(out_dir) / "series"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / "m5_series.npz"
    meta = out_dir / "m5_series_meta.csv"
    if cache.exists() and meta.exists():
        z = np.load(cache, allow_pickle=True)
        m = pd.read_csv(meta)
        out = {}
        for r in m.itertuples():
            k = r.role
            out[k] = Series(r.id, r.store, r.item, r.cat, z[f"{k}_dates"], z[f"{k}_demand"], z[f"{k}_price"],
                            z[f"{k}_valid"], z[f"{k}_weekday"], z[f"{k}_event"], z[f"{k}_snap"])
        return out
    name, ids, S, dcols = load_sales_matrix(folder)
    prof = profile(ids, S)
    prof["_idx"] = np.arange(len(prof))
    prof.to_csv(out_dir / "m5_profile.csv", index=False)
    ch = choose_series(prof)
    cal = _read_csv(folder, "calendar.csv")
    pairs = [(r.store_id, r.item_id) for r in ch.values()]
    pr = load_prices(folder, pairs)
    out, rows, arrs = {}, [], {}
    for role, row in ch.items():
        s = build_series(folder, row, S, dcols, cal, pr)
        out[role] = s
        rows.append(dict(role=role, id=s.sid, store=s.store, item=s.item, cat=s.cat, sales_file=name))
        for fld in ("dates", "demand", "price", "valid", "weekday", "event", "snap"):
            arrs[f"{role}_{fld}"] = getattr(s, fld)
    np.savez(cache, **arrs)
    pd.DataFrame(rows).to_csv(meta, index=False)
    return out
