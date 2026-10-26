"""Stage 0: read-only inspection of trading files. Writes outputs/step0_report.txt (never touches data dirs)."""
import sys, io
from pathlib import Path
import numpy as np
import pandas as pd

TRADING_FILES = ["C:/Users/username/OneDrive - XXYY/Muni/project/India/TECHM/TECHM_extend.csv.gzip"]
OUT = Path(__file__).parent / "outputs" / "step0_report.txt"
buf = io.StringIO()


def P(*a):
    print(*a, file=buf)
    print(*a)


def rsi_wilder(close, n):
    d = np.diff(close, prepend=np.nan)
    up = np.where(d > 0, d, 0.0)
    dn = np.where(d < 0, -d, 0.0)
    out = np.full(len(close), np.nan)
    if len(close) <= n:
        return out
    au = up[1:n + 1].mean(); ad = dn[1:n + 1].mean()
    out[n] = 100 - 100 / (1 + au / ad) if ad > 0 else 100.0
    for i in range(n + 1, len(close)):
        au = (au * (n - 1) + up[i]) / n
        ad = (ad * (n - 1) + dn[i]) / n
        out[i] = 100 - 100 / (1 + au / ad) if ad > 0 else 100.0
    return out


def psar(high, low, step, mx):
    """Parabolic SAR; value on row t is the SAR *for* day t (computed from rows <= t-1)."""
    n = len(high)
    out = np.full(n, np.nan)
    if n < 3:
        return out
    up = high[1] > high[0]
    sar = low[0] if up else high[0]
    ep = high[0] if up else low[0]
    af = step
    out[1] = sar
    for i in range(2, n):
        sar = sar + af * (ep - sar)
        if up:
            sar = min(sar, low[i - 1], low[i - 2])
            if low[i] < sar:
                up = False; sar = ep; ep = low[i]; af = step
            elif high[i] > ep:
                ep = high[i]; af = min(af + step, mx)
        else:
            sar = max(sar, high[i - 1], high[i - 2])
            if high[i] > sar:
                up = True; sar = ep; ep = high[i]; af = step
            elif low[i] < ep:
                ep = low[i]; af = min(af + step, mx)
        out[i] = sar
    return out


def main():
    import torch
    P("python", sys.version.split()[0], "torch", torch.__version__)
    for f in TRADING_FILES:
        p = Path(f)
        P("FILE", p, "exists", p.exists(), "size", p.stat().st_size if p.exists() else None)
        df = pd.read_csv(p, compression="gzip")
        P("shape", df.shape)
        P("first cols", list(df.columns[:12]))
        P("dtypes counts", df.dtypes.value_counts().to_dict())
        req = ["tdate", "open", "high", "low", "close", "volume", "sar", "RSI"]
        P("missing required", [c for c in req if c not in df.columns])
        df["tdate"] = pd.to_datetime(df["tdate"])
        P("date range", df.tdate.min(), df.tdate.max(), "monotonic", df.tdate.is_monotonic_increasing,
          "dups", int(df.tdate.duplicated().sum()))
        na = df.isna().sum()
        P("NaN count top 15:\n", na[na > 0].sort_values(ascending=False).head(15).to_string())
        P("NaN core:", na[req].to_dict())
        P("nifty_volume all zero:", bool((df.nifty_volume == 0).all()))
        # sanity
        P("high>=max(open,close) share", float((df.high >= np.maximum(df.open, df.close) - 1e-9).mean()))
        P("low<=min(open,close) share", float((df.low <= np.minimum(df.open, df.close) + 1e-9).mean()))
        P("high>=low share", float((df.high >= df.low).mean()))
        P("RSI range", df.RSI.min(), df.RSI.max(), "sar>0 share", float((df.sar.dropna() > 0).mean()))
        # price-scale audit
        df["year"] = df.tdate.dt.year
        inb = (df.close >= df.low - 1e-9) & (df.close <= df.high + 1e-9)
        P("share low<=close<=high", float(inb.mean()))
        mid = (df.high + df.low) / 2
        g = pd.DataFrame({"year": df.year, "inb": inb, "ratio": df.close / mid})
        P("by year (in-bounds share, median close/mid):\n", g.groupby("year").agg(inb=("inb", "mean"), ratio=("ratio", "median"),
                                                                                  n=("inb", "size")).round(4).to_string())
        ret = df.close.pct_change()
        P("10 largest |close ret|:\n", pd.DataFrame({"tdate": df.tdate, "ret": ret}).reindex(ret.abs().sort_values(ascending=False).index[:10]).to_string())
        ro = (df.open / df.close.shift(1) - 1)
        P("open/prev close vs close ratio check (max |open gap|):", float(ro.abs().max()))
        # adj factor candidates: close / (typical unadjusted close). Use open as unadjusted proxy
        P("median close/open by year:\n", (df.close / df.open).groupby(df.year).median().round(4).to_string())
        # indicator audit
        d = df.dropna(subset=["RSI"]).copy()
        close = df.close.values
        P("-- RSI audit (on adjusted close and on open-based proxy) --")
        res = []
        for n in (7, 9, 14, 21):
            for name, series in (("close", df.close.values), ("open", df.open.values)):
                r = rsi_wilder(series, n)
                m = ~np.isnan(r) & ~np.isnan(df.RSI.values)
                m[:300] = False
                if m.sum():
                    res.append((n, name, float(np.nanmedian(np.abs(r[m] - df.RSI.values[m]))), float(np.nanmax(np.abs(r[m] - df.RSI.values[m])))))
        for r in sorted(res, key=lambda z: z[2]):
            P("RSI period/series medAbsDiff maxAbsDiff", r)
        # RSI timing: compare shift
        best = min(res, key=lambda z: z[2])
        r = rsi_wilder(df[best[1]].values, best[0])
        for sh in (-1, 0, 1):
            rr = np.roll(r, sh)
            m = ~np.isnan(rr) & ~np.isnan(df.RSI.values); m[:300] = False
            P("RSI shift", sh, float(np.nanmedian(np.abs(rr[m] - df.RSI.values[m]))))
        P("-- SAR audit --")
        h, l = df.high.values, df.low.values
        resS = []
        for st in (0.01, 0.02, 0.03):
            for mx in (0.1, 0.2, 0.3):
                s = psar(h, l, st, mx)
                for sh in (0, 1):
                    ss = np.roll(s, sh)
                    m = ~np.isnan(ss) & ~np.isnan(df.sar.values); m[:300] = False
                    if m.sum():
                        rel = np.abs(ss[m] / df.sar.values[m] - 1)
                        resS.append((st, mx, sh, float(np.median(rel)), float((rel < 1e-3).mean())))
        for r in sorted(resS, key=lambda z: z[3])[:8]:
            P("SAR step/max/shift medRelDiff share<1e-3", r)
        # SAR timing: is sar_t related to data <= t-1 ?
        P("sar[t] vs high[t-1] equality share", float((np.abs(df.sar - df.high.shift(1)) < 1e-6).mean()),
          "vs low[t-1]", float((np.abs(df.sar - df.low.shift(1)) < 1e-6).mean()))
        P("share sar<low", float((df.sar < df.low).mean()), "share sar>high", float((df.sar > df.high).mean()),
          "reversal", float(((df.sar >= df.low) & (df.sar <= df.high)).mean()))
        P("share sar<low or >high among rows with sar", float(((df.sar < df.low) | (df.sar > df.high))[df.sar.notna()].mean()))
        # volume trailing zscore reproducibility trivial; return features from close trivial
        P("zero volume days", int((df.volume == 0).sum()), "turnover_pp NaN", int(df.turnover_pp.isna().sum()))
        # clean rows
        core = df[["open", "high", "low", "close", "volume", "sar", "RSI"]].notna().all(axis=1)
        P("clean core rows", int(core.sum()))
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(buf.getvalue(), encoding="utf-8")


if __name__ == "__main__":
    main()
