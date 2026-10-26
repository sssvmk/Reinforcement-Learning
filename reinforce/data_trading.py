"""Trading data: read-only gzip loader, schema check, audits, whitelist features (no look-ahead)."""
from dataclasses import dataclass
import numpy as np
import pandas as pd

REQUIRED = ["tdate", "open", "high", "low", "close", "volume", "sar", "RSI"]
FEATURES = ["rsi", "trend", "osar", "ret5", "volz"]   # columns of TradingData.X (raw, unstandardized)


class SchemaError(Exception):
    pass


def read_trading(path, usecols=None):
    """Read in place, read-only. Pandas does not auto-detect '.gzip', so compression is explicit."""
    df = pd.read_csv(path, compression="gzip", usecols=usecols)
    check_schema(df)
    return df


def check_schema(df):
    miss = [c for c in REQUIRED if c not in df.columns]
    if miss:
        raise SchemaError(f"missing required columns: {miss}")


def read_trading_safe(path, log=print):
    """Returns a DataFrame or None (file excluded and logged)."""
    try:
        return read_trading(path, usecols=lambda c: c in REQUIRED + ["nifty_close"])
    except (SchemaError, FileNotFoundError, ValueError, OSError) as e:
        log(f"EXCLUDED {path}: {e}")
        return None


def wilder_rsi(close, period):
    c = np.asarray(close, float)
    d = np.diff(c, prepend=np.nan)
    up, dn = np.where(d > 0, d, 0.0), np.where(d < 0, -d, 0.0)
    au, ad = np.full(len(c), np.nan), np.full(len(c), np.nan)
    if len(c) <= period:
        return au
    au[period] = up[1:period + 1].mean(); ad[period] = dn[1:period + 1].mean()
    for i in range(period + 1, len(c)):
        au[i] = (au[i - 1] * (period - 1) + up[i]) / period
        ad[i] = (ad[i - 1] * (period - 1) + dn[i]) / period
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = au / ad
        return np.where(ad == 0, 100.0, 100 - 100 / (1 + rs))


def parabolic_sar(high, low, step=0.02, maxaf=0.2):
    h, l = np.asarray(high, float), np.asarray(low, float)
    n = len(h)
    sar = np.full(n, np.nan)
    up, af, ep, s = True, step, h[0], l[0]
    sar[0] = s
    for i in range(1, n):
        s = s + af * (ep - s)
        if up:
            s = min(s, l[i - 1], l[i - 2] if i > 1 else l[i - 1])
            if l[i] < s:
                up, s, ep, af = False, ep, l[i], step
            elif h[i] > ep:
                ep, af = h[i], min(af + step, maxaf)
        else:
            s = max(s, h[i - 1], h[i - 2] if i > 1 else h[i - 1])
            if h[i] > s:
                up, s, ep, af = True, ep, h[i], step
            elif l[i] < ep:
                ep, af = l[i], min(af + step, maxaf)
        sar[i] = s
    return sar


def audit(df, warm=250):
    """Schema, sanity, price-scale and indicator audits. Returns a dict of findings."""
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    d = pd.to_datetime(df["tdate"])
    out = {"rows": len(df), "start": str(d.iloc[0].date()), "end": str(d.iloc[-1].date())}
    out["dates_monotonic"] = bool(d.is_monotonic_increasing)
    out["dates_duplicated"] = int(d.duplicated().sum())
    out["high_ge_open_close"] = float(np.mean(h >= np.maximum(o, c) - 1e-9))
    out["low_le_open_close"] = float(np.mean(l <= np.minimum(o, c) + 1e-9))
    rsi = df["RSI"].to_numpy(float)
    out["rsi_in_0_100"] = bool(np.nanmin(rsi) >= 0 and np.nanmax(rsi) <= 100)
    out["sar_positive"] = bool(np.nanmin(df["sar"].to_numpy(float)) > 0)
    inb = (l <= c) & (c <= h)
    yrs = d.dt.year.to_numpy()
    out["close_in_range_share"] = float(inb.mean())
    out["scale_by_year"] = {int(y): {"in_range": float(inb[yrs == y].mean()),
                                     "close_over_mid": float(np.median(c[yrs == y] / ((h[yrs == y] + l[yrs == y]) / 2)))}
                            for y in np.unique(yrs)}
    r = pd.Series(c).pct_change()
    top = r.abs().nlargest(10).index
    out["top10_abs_returns"] = [(str(d.iloc[i].date()), float(r.iloc[i])) for i in top]
    sar_s = df["sar"].to_numpy(float)
    best = None
    for p in (7, 9, 14, 21):
        mine = wilder_rsi(c, p)
        err = np.nanmean(np.abs(mine[warm:] - rsi[warm:]))
        best = (err, p) if best is None or err < best[0] else best
    out["rsi_best_period"], out["rsi_best_mae"] = best[1], float(best[0])
    mine = wilder_rsi(c, best[1])
    out["rsi_timing_mae"] = {lag: float(np.nanmean(np.abs(np.roll(mine, lag)[warm + 2:] - rsi[warm + 2:]))) for lag in (-1, 0, 1)}
    bs = None
    for st in (0.01, 0.02, 0.03):
        for mx in (0.1, 0.2, 0.3):
            mine = parabolic_sar(h, l, st, mx)
            for lag in (0, 1):
                m = np.roll(mine, lag)
                err = float(np.mean(np.abs(m[warm + 2:] / sar_s[warm + 2:] - 1) < 0.005))
                bs = (err, st, mx, lag) if bs is None or err > bs[0] else bs
    out["sar_best"] = {"match_share_0.5pct": bs[0], "step": bs[1], "max": bs[2], "lag": bs[3]}
    out["sar_equals_prev_hl_row2"] = bool(abs(sar_s[1] - h[0]) < 1e-6)
    out["sar_trend_uses_prior_data"] = float(np.mean((sar_s[1:] < l[1:]) | (sar_s[1:] > h[1:])))
    return out


@dataclass
class TradingData:
    dates: np.ndarray
    close: np.ndarray
    X: np.ndarray            # raw features, columns FEATURES
    R: np.ndarray            # R[t] = close[t+1]/close[t]-1 (return earned by the position chosen at close t)

    @property
    def n(self):
        return len(self.close)


def open_returns(o, c, tol=0.08):
    """R[t] = open[t+2]/open[t+1]-1: trade at the next open (unadjusted OHLC scale). On days where the open-to-open ratio differs from the
    close-to-close ratio of the same interval by more than `tol` in log terms (split / large corporate action) the close-to-close return is used."""
    n = len(o)
    r = np.zeros(n)
    ro = o[2:] / o[1:-1]
    rc = c[2:] / c[1:-1]
    bad = np.abs(np.log(ro) - np.log(rc)) > tol
    r[:n - 2] = np.where(bad, rc, ro) - 1.0
    return r, float(bad.mean())


def build_features(df, warmup_rows=120, allow_adjusted_close=True, min_rows=500, return_source="close"):
    """Whitelist features. Every value at row t depends only on rows <= t.

    SAR features use `sar`, `high`, `low`, `open` only (unadjusted scale); `close` is used for returns only.
    """
    check_schema(df)
    o, h, l, c, v = (df[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume"))
    s = df["sar"].to_numpy(float)
    trend = np.where(s < l, 1.0, np.where(s > h, -1.0, 0.0))
    osar = (o - s) / o
    cs = pd.Series(c)
    ret5 = np.log(cs / cs.shift(5)).to_numpy()
    vs = pd.Series(v)
    volz = ((vs - vs.rolling(60).mean()) / vs.rolling(60).std()).to_numpy()
    X = np.column_stack([df["RSI"].to_numpy(float), trend, osar, ret5, volz])
    R = np.r_[c[1:] / c[:-1] - 1.0, 0.0]
    if return_source == "open":
        R, _ = open_returns(o, c)
    elif return_source != "close":
        raise ValueError("return_source must be 'close' or 'open'")
    keep = np.arange(len(df)) >= warmup_rows
    keep &= np.all(np.isfinite(X[:, [0, 2, 3]]), axis=1)
    if not allow_adjusted_close:
        ok = (l <= c) & (c <= h)
        best, start = (0, 0), None
        for i, f in enumerate(np.r_[ok, False]):
            if f and start is None:
                start = i
            if not f and start is not None:
                best = max(best, (i - start, start)); start = None
        rng_ = np.zeros(len(df), bool); rng_[best[1]:best[1] + best[0]] = True
        keep &= rng_
    idx = np.where(keep)[0]
    if len(idx) < min_rows:
        raise SchemaError(f"only {len(idx)} usable rows (< {min_rows}): price-scale audit leaves too little data")
    sel = slice(idx[0], idx[-1] + 1)
    X = X[sel].copy(); X[:, 4] = np.nan_to_num(X[:, 4])
    Rr = R[sel].copy(); Rr[-1] = 0.0
    return TradingData(pd.to_datetime(df["tdate"]).to_numpy()[sel], c[sel], X, Rr)


def make_splits(n, a=0.6, b=0.8):
    """Chronological train/val/test index ranges [lo, hi)."""
    i, j = int(n * a), int(n * b)
    return {"train": (0, i), "val": (i, j), "test": (j, n)}
