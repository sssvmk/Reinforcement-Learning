"""Trading data loader (read-only). One ticker per file in config paths.trading_files."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED = ["tdate", "open", "high", "low", "close", "volume", "sar", "RSI"]


def rsi_wilder(close, n=14):
    """Wilder RSI (trailing). Used only for tests; stored RSI matched this in the Step 0 audit."""
    c = np.asarray(close, float)
    d = np.diff(c, prepend=np.nan)
    up = np.where(d > 0, d, 0.0)
    dn = np.where(d < 0, -d, 0.0)
    out = np.full(len(c), np.nan)
    if len(c) <= n:
        return out
    au = up[1:n + 1].mean()
    ad = dn[1:n + 1].mean()
    out[n] = 100 - 100 / (1 + au / ad) if ad > 0 else 100.0
    for i in range(n + 1, len(c)):
        au = (au * (n - 1) + up[i]) / n
        ad = (ad * (n - 1) + dn[i]) / n
        out[i] = 100 - 100 / (1 + au / ad) if ad > 0 else 100.0
    return out


def trailing_features(df):
    """All features use rows <= t only. df must be sorted ascending by date."""
    c = df["close"].astype(float)
    lc = np.log(c)
    f = pd.DataFrame(index=df.index)
    f["ret1"] = lc.diff(1)
    f["ret5"] = lc.diff(5)
    f["ret20"] = lc.diff(20)
    f["rsi"] = df["RSI"].astype(float)
    # SAR trend flag on the unadjusted scale (sar vs low/high only): 0 uptrend, 1 downtrend, 2 reversal
    sar, lo, hi = df["sar"].astype(float), df["low"].astype(float), df["high"].astype(float)
    flag = np.where(sar < lo, 0, np.where(sar > hi, 1, 2)).astype(float)
    flag[sar.isna().values] = np.nan
    f["flag"] = flag
    f["dist"] = ((df["open"] - sar) / df["open"]).clip(-0.2, 0.2)
    lv = np.log1p(df["volume"].astype(float))
    m = lv.rolling(60, min_periods=60).mean()
    s = lv.rolling(60, min_periods=60).std().clip(lower=1e-3)
    f["volz"] = ((lv - m) / s).clip(-4, 4)
    return f


class TradingData:
    def __init__(self, path, cfg, df=None):
        self.path = Path(path)
        self.ticker = self.path.name.split("_extend")[0]
        self.excluded_reason = None
        if df is None:
            df = pd.read_csv(self.path, compression="gzip", usecols=lambda c: c in REQUIRED)
        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            self.excluded_reason = f"missing columns {missing}"
            self.N = 0
            return
        df = df[REQUIRED].copy()
        df["tdate"] = pd.to_datetime(df["tdate"])
        df = df.sort_values("tdate").reset_index(drop=True)
        self.n_raw = len(df)
        feats = trailing_features(df)
        warm = cfg["trading"]["warmup_rows"]
        keep = (np.arange(len(df)) >= warm) & feats.notna().all(axis=1).values & df["close"].notna().values
        # contiguous clean block from the first fully valid row on (drops any later NaN rows too)
        idx = np.where(keep)[0]
        if len(idx) < cfg["trading"]["min_clean_rows"]:
            self.excluded_reason = f"only {len(idx)} clean rows"
            self.N = 0
            return
        self.row_idx = idx
        d = df.iloc[idx].reset_index(drop=True)
        f = feats.iloc[idx].reset_index(drop=True)
        self.dates = d["tdate"].values
        self.close = d["close"].values.astype(float)
        self.open = d["open"].values.astype(float)
        self.high = d["high"].values.astype(float)
        self.low = d["low"].values.astype(float)
        self.sar = d["sar"].values.astype(float)
        self.volume = d["volume"].values.astype(float)
        self.feat = {k: f[k].values.astype(float) for k in f.columns}
        self.N = len(d)
        # fraction of rows where close lies inside [low, high]: documented scale audit
        self.close_in_range = float(((d.close >= d.low) & (d.close <= d.high)).mean())
        # SAR-trend rule used as behavior policy base: long when sar < low
        self.sar_up = (self.feat["flag"] == 0).astype(np.int8)
        # next-day simple return from close (return t -> t+1); last row has none
        self.fwd_ret = np.append(self.close[1:] / self.close[:-1] - 1.0, 0.0)


def load_all(cfg):
    out, log = [], []
    for p in cfg["paths"]["trading_files"]:
        p = Path(p)
        if not p.is_file():
            log.append(f"{p}: file not found, skipped")
            continue
        td = TradingData(p, cfg)
        if td.excluded_reason:
            log.append(f"{p.name}: excluded ({td.excluded_reason})")
        else:
            out.append(td)
            log.append(f"{td.ticker}: {td.N} clean rows {str(td.dates[0])[:10]}..{str(td.dates[-1])[:10]}")
    return out, log
