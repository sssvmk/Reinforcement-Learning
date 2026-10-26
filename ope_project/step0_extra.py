import numpy as np, pandas as pd
from pathlib import Path
from step0_audit import psar, rsi_wilder

f = "C:/Users/msk80743/OneDrive - GSK/Muni/project/India/TECHM/TECHM_extend.csv.gzip"
df = pd.read_csv(f, compression="gzip"); df["tdate"] = pd.to_datetime(df.tdate)
out = []
P = out.append
rc = df.close.pct_change(); mid = (df.high + df.low) / 2; rm = mid.pct_change(); ro = df.open.pct_change()
d = (np.log1p(rc) - np.log1p(ro))
P("close-ret minus open-ret: std %.4f, share |diff|>5%%: %.4f, >10%%: %.4f" % (d.std(), (d.abs() > .05).mean(), (d.abs() > .10).mean()))
big = df.loc[d.abs() > .08, ["tdate", "open", "close"]].assign(rc=rc, ro=ro)
P("days with |log diff|>8%:\n" + big.to_string())
# smoothness of adjustment factor f = close/open (proxy): daily log change, large jumps = adjustment events
fct = np.log(df.close / df.open)
P("ex-date-like factor steps (|dlog(close/open)|>3pct) count: %d" % (fct.diff().abs() > .03).sum())
nc = np.log(df.nifty_close).diff()
P("corr(close ret, nifty ret)=%.3f  corr(open ret, nifty ret)=%.3f" % (rc.corr(nc.pipe(np.expm1)), ro.corr(nc.pipe(np.expm1))))
s = psar(df.high.values, df.low.values, .02, .2)
rel = np.abs(s / df.sar.values - 1)
bad = rel > 1e-3
P("SAR mismatch share overall %.3f ; by year:\n%s" % (np.nanmean(bad[300:]), pd.Series(bad).groupby(df.tdate.dt.year).mean().round(3).to_string()))
# side agreement
side_s = (s < df.low.values); side_f = (df.sar.values < df.low.values)
P("SAR flag (sar<low) agreement between recomputed and stored: %.4f" % np.mean(side_s[5:] == side_f[5:]))
# causality of recompute
ok = True
for t in (500, 1500, 3000):
    s2 = psar(df.high.values[:t + 1], df.low.values[:t + 1], .02, .2)
    ok &= bool(np.isclose(s2[t], s[t]))
P("psar recompute causal (truncation test): %s" % ok)
# RSI warmup
r = rsi_wilder(df.close.values, 14)
dd = np.abs(r - df.RSI.values)
P("RSI abs diff first 40 rows: %s" % np.round(dd[:40], 3).tolist())
P("first rows RSI: %s" % df.RSI.head(20).round(2).tolist())
# volume
P("volume zero rows:\n" + df.loc[df.volume == 0, ["tdate", "volume", "turnover_pp"]].to_string())
Path("outputs/step0_extra.txt").write_text("\n".join(map(str, out)), encoding="utf-8")
print("\n".join(map(str, out)))
