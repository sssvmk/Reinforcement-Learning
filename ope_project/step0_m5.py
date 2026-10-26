"""Stage 0 for M5: file listing, head/shape/dtypes/nulls, calendar range, price gaps, series profile and selection."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from common import load_config, check_m5
import data_m5 as m5

cfg = load_config()
st = check_m5(cfg)
out = Path(cfg["paths"]["outputs"])
lines = [f"M5 status: {st}"]
folder = Path(cfg["paths"]["m5_dir"])
for p in sorted(folder.rglob("*")):
    if p.is_file():
        lines.append(f"  {p.name}  {p.stat().st_size:,} bytes")
cal = m5._read_csv(folder, "calendar.csv")
lines += ["calendar.shape %s" % (cal.shape,), "calendar dtypes:\n" + cal.dtypes.to_string(),
          "calendar nulls:\n" + cal.isna().sum().to_string(),
          f"calendar date range: {cal.date.min()} .. {cal.date.max()}"]
name, ids, S, dcols = m5.load_sales_matrix(folder)
lines += [f"sales file used: {name}", f"sales shape: {S.shape} (series x days), d_* columns: {len(dcols)}",
          f"id columns head:\n{ids.head(3).to_string()}", f"S dtype {S.dtype}, negatives: {(S < 0).sum()}",
          f"categories: {ids.cat_id.value_counts().to_dict()}", f"states: {ids.state_id.value_counts().to_dict()}"]
prof = m5.profile(ids, S)
prof["_idx"] = np.arange(len(prof))
lines += ["profile describe:\n" + prof[["n", "mean", "zero_frac", "cv", "wd_amp"]].describe().round(3).to_string()]
ch = m5.choose_series(prof)
for role, r in ch.items():
    lines.append(f"chosen {role}: {r['id']} cat={r['cat_id']} n={int(r['n'])} mean={r['mean']:.2f} "
                 f"zero_frac={r['zero_frac']:.3f} cv={r['cv']:.2f} weekday_amp={r['wd_amp']:.3f}")
ser = m5.load_inventory_series(cfg, out)
for role, s in ser.items():
    lines.append(f"{role}: days={s.N} valid_price_days={int(s.valid.sum())} first_date={str(s.dates[0])[:10]} "
                 f"last_date={str(s.dates[-1])[:10]} price min/mean/max={np.nanmin(s.price):.2f}/"
                 f"{np.nanmean(s.price):.2f}/{np.nanmax(s.price):.2f} invalid_days={int((~s.valid).sum())} "
                 f"events={int(s.event.sum())} snap_days={int(s.snap.sum())}")
(out / "step0_m5_report.txt").write_text("\n".join(lines), encoding="utf8")
print("\n".join(lines))
