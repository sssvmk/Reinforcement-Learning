"""Step 0: read-only inspection of M5_DIR. Writes DATA_NOTES.md and series_selection.txt."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from data import find_file, load_config, read_csv_any

ROOT = Path(__file__).parent


def describe(name, df, out):
    out.append(f"\n### {name}\nshape: {df.shape}\n")
    out.append("dtypes:\n```\n" + df.dtypes.astype(str).to_string() + "\n```")
    nn = df.isna().sum()
    out.append("null counts (non-zero only):\n```\n" + (nn[nn > 0].to_string() if (nn > 0).any() else "none") + "\n```")
    out.append("head (first 12 cols):\n```\n" + df.iloc[:3, :12].to_string() + "\n```")


def main():
    cfg = load_config()
    m5 = Path(cfg["m5_dir"])
    out = ["# DATA_NOTES (step 0, read-only inspection)\n", f"m5_dir: `{m5}`\n", "## File listing\n```"]
    for p in sorted(m5.rglob("*")):
        if p.is_file():
            line = f"{p.relative_to(m5)}  {p.stat().st_size:,} bytes"
            print(line)
            out.append(line)
    out.append("```")
    found = {}
    for key in ["sales_train_validation.csv", "sales_train_evaluation.csv", "calendar.csv", "sell_prices.csv"]:
        try:
            found[key] = find_file(m5, key)
            out.append(f"- found `{key}` -> `{found[key]}`")
        except FileNotFoundError as e:
            out.append(f"- MISSING `{key}`: {e}")
    sales_key = next((k for k in ["sales_train_validation.csv", "sales_train_evaluation.csv"] if k in found), None)
    if sales_key is None or "calendar.csv" not in found or "sell_prices.csv" not in found:
        (ROOT / "DATA_NOTES.md").write_text("\n".join(out), encoding="utf-8")
        sys.exit("Required file missing; see DATA_NOTES.md")
    out.append("- `sample_submission.csv` ignored by design.")

    cal = read_csv_any(found["calendar.csv"])
    describe("calendar.csv", cal, out)
    out.append(f"date range: {cal['date'].min()} .. {cal['date'].max()}; rows={len(cal)}")

    sales = read_csv_any(found[sales_key])
    dcols = [c for c in sales.columns if c.startswith("d_")]
    describe(sales_key + " (id columns only)", sales.iloc[:, :6], out)
    out.append(f"number of d_* columns: {len(dcols)}; unique series ids: {sales['id'].nunique()}")
    out.append(f"states={sales.state_id.nunique()} stores={sales.store_id.nunique()} cats={sales.cat_id.nunique()} "
               f"depts={sales.dept_id.nunique()} items={sales.item_id.nunique()}")
    X = sales[dcols].to_numpy(dtype=np.int16)
    out.append(f"sales min={X.min()} max={X.max()}; all non-negative: {bool((X >= 0).all())}")

    prices = read_csv_any(found["sell_prices.csv"], dtype={"store_id": "category", "item_id": "category",
                                                           "wm_yr_wk": "int16", "sell_price": "float32"})
    describe("sell_prices.csv", prices, out)
    out.append(f"Missing prices appear as ABSENT rows (not NaN): sell_price nulls = {int(prices.sell_price.isna().sum())}")
    del prices

    first = (X > 0).argmax(axis=1)
    Xf = X.astype(np.float64)
    wday = cal.set_index("d").loc[dcols, "wday"].to_numpy()
    rec = []
    for i in range(len(X)):
        x = Xf[i, first[i]:]
        m = x.mean()
        w = wday[first[i]:]
        dm = np.array([x[w == k].mean() for k in range(1, 8)])
        rec.append((m, (x == 0).mean(), x.std() / m if m > 0 else np.nan,
                    dm.max() / dm.min() if dm.min() > 0 else np.nan))
    prof = pd.DataFrame(rec, columns=["mean", "zero_frac", "cv", "wk_ratio"])
    prof.insert(0, "first_sale_idx", first)
    prof.insert(0, "state", sales.state_id.values)
    prof.insert(0, "cat", sales.cat_id.values)
    prof.insert(0, "id", sales["id"].values)

    d = prof[(prof.cat == "FOODS") & prof["mean"].between(8, 15) & (prof.zero_frac < 0.05) & (prof.first_sale_idx < 30)]
    d = d.sort_values("cv")
    prim = d.iloc[len(d) // 2]
    inter = prof[prof.zero_frac.between(0.5, 0.7) & (prof["mean"] > 0.3) & (prof.first_sale_idx < 30)]
    inter = inter.iloc[(inter.zero_frac - 0.6).abs().argsort()].iloc[0]
    seas = prof[prof["mean"].between(5, 20) & (prof.zero_frac < 0.1) & (prof.first_sale_idx < 30)]
    seas = seas[seas.id != prim.id].sort_values("wk_ratio", ascending=False).iloc[0]
    out.append("\n## Series profiling (leading zeros trimmed before stats)\n")
    out.append(f"series whose first sale is after d_100: {int((prof.first_sale_idx > 100).sum())} of {len(prof)}; "
               f"median zero fraction {prof.zero_frac.median():.3f}; FOODS candidates for primary: {len(d)}")
    rows = pd.DataFrame([prim, inter, seas], index=["primary", "intermittent", "seasonal"])
    out.append("```\n" + rows.round(3).to_string() + "\n```")
    out.append("""
Selection rationale:
- **primary**: FOODS, post-launch mean 8-15 units/day, zero fraction < 5%, first sale within 30 days, median CV
  among such items -> mid-volume dense demand, so lost-sales MDP is informative and censoring is minor.
- **intermittent** (replication 1): zero fraction closest to 0.6 with mean > 0.3 -> sparse demand.
- **seasonal** (replication 2): largest max/min weekday-mean ratio among dense mid-volume series -> strong weekday effect.
""")
    print(rows.round(3).to_string())
    ids = {"primary": prim["id"], "intermittent": inter["id"], "seasonal": seas["id"]}
    out.append("selected ids: " + str(ids))
    out.append(f"\n## Observed vs. documented\n- observed {len(sales)} series, {len(dcols)} d_* columns ({sales_key}), "
               f"{len(cal)} calendar rows. Documented: ~30,490 / 1,913 (val) or 1,941 (eval) / ~1,969.")
    try:
        import torch
        out.append(f"\n## PyTorch\nimportable, version {torch.__version__}")
        print("torch", torch.__version__)
    except ImportError:
        (ROOT / "DATA_NOTES.md").write_text("\n".join(out), encoding="utf-8")
        sys.exit("PyTorch missing: stop and tell the user (no install).")
    (ROOT / "DATA_NOTES.md").write_text("\n".join(out), encoding="utf-8")
    (ROOT / "series_selection.txt").write_text("\n".join(f"{k}={v}" for k, v in ids.items()), encoding="utf-8")


if __name__ == "__main__":
    main()
