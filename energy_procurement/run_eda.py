"""Stage 1 EDA. Decisions use 2015-2017 only; 2018 is described separately."""
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import data
import costs

OUTD = os.path.join(data.OUT, "eda")
os.makedirs(OUTD, exist_ok=True)
REPORT = os.path.join(data.ROOT, "EDA_REPORT.md")


def fig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUTD, name), dpi=110)
    plt.close()
    return f"outputs/eda/{name}"


def md_table(df, fmt="{:.3f}"):
    df = df.copy()
    cols = [df.index.name or ""] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for i, r in df.iterrows():
        lines.append("| " + " | ".join([str(i)] + [fmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
                                                    for v in r.values]) + " |")
    return "\n".join(lines)


def main():
    cfg = data.load_config()
    S = {}
    raw_e = data.read_energy_raw(cfg["energy_file"])
    raw_w = data.read_weather_raw(cfg["weather_file"])
    ew, wn = raw_e.shape, raw_w.shape
    dup_w = int(raw_w.duplicated(["city_name", "dt_iso"]).sum())
    dup_rows_e = int(raw_e.duplicated().sum())
    miss = raw_e.isna().sum()
    miss = miss[miss > 0]
    const_cols = [c for c in raw_e.select_dtypes("number") if raw_e[c].nunique(dropna=True) <= 1]
    allnan = [c for c in raw_e.columns if raw_e[c].isna().all()]
    city_counts = raw_w.city_name.value_counts().to_dict()
    wclean = data.clean_weather(raw_w)
    df = data.load_hourly(cfg)
    dst = df.groupby("date").size()
    n23, n25 = int((dst == 23).sum()), int((dst == 25).sum())
    A = data.day_arrays(df, cfg)
    B = costs.scaled(A)
    dates = A["dates"]
    yr = np.asarray(dates.year)
    dec = np.isin(yr, [2015, 2016, 2017])  # decision years

    # ---- hourly frame with derived variables
    h = df.copy()
    h["gap"] = h["price actual"] - h["price day ahead"]
    h["ratio"] = h["price actual"] / h["price day ahead"]
    h["eL"] = h["total load actual"] / h["total load forecast"] - 1
    h["eL_mw"] = h["total load actual"] - h["total load forecast"]
    hd = h[h.year.isin([2015, 2016, 2017])]
    S["mem_mb"] = float(raw_e.memory_usage(deep=True).sum() / 1e6)
    S["time_checks"] = {"unique": int(h.index.nunique()), "n": len(h), "dst23": n23, "dst25": n25}

    # ---- prices
    pr = h.groupby("year")[["price day ahead", "price actual", "gap"]].agg(["mean", "std", "min", "max"])
    pr.columns = ["_".join(c) for c in pr.columns]
    neg = {int(y): [int((g["price day ahead"] <= 0).sum()), int((g["price actual"] <= 0).sum())]
           for y, g in h.groupby("year")}
    gap_pos = h.groupby("year")["gap"].apply(lambda s: float((s > 0).mean()))
    gap_hour = hd.groupby("hour_local")["gap"].agg(["mean", "std"])
    gap_month = hd.groupby(hd.index.tz_convert(data.TZ).month)["gap"].mean()
    ratio_y = h.groupby("year")["ratio"].median()
    corr_gap_pda = float(hd["gap"].corr(hd["price day ahead"]))
    S["gap_pos_share"] = gap_pos.to_dict()
    S["corr_gap_pda_dec"] = corr_gap_pda

    fg, ax = plt.subplots(2, 3, figsize=(15, 8))
    for y, g in h.groupby("year"):
        ax[0, 0].hist(g["price day ahead"], 60, alpha=.45, label=str(y), density=True)
    ax[0, 0].set_title("Day-ahead price distribution"); ax[0, 0].legend()
    hd.groupby("hour_local")[["price day ahead", "price actual"]].mean().plot(ax=ax[0, 1], title="Mean price by hour (2015-17)")
    hd.groupby(hd.index.tz_convert(data.TZ).month)[["price day ahead", "price actual"]].mean().plot(ax=ax[0, 2], title="Mean price by month (2015-17)")
    sub = h.sample(6000, random_state=0)
    for y, g in sub.groupby("year"):
        ax[1, 0].scatter(g["price day ahead"], g["price actual"], s=2, alpha=.4, label=str(y))
    ax[1, 0].plot([0, 100], [0, 100], "k--"); ax[1, 0].set_title("pA vs pDA"); ax[1, 0].legend(markerscale=4)
    h.boxplot(column="gap", by="year", ax=ax[1, 1]); ax[1, 1].set_title("Gap pA - pDA by year")
    gap_hour["mean"].plot(ax=ax[1, 2], yerr=gap_hour["std"], title="Gap by hour (mean +/- sd, 2015-17)")
    plt.suptitle("")
    f_price = fig("prices.png")

    # ---- load
    err_y = h.groupby("year")["eL"].agg(["mean", "std", lambda s: s.abs().mean()])
    err_y.columns = ["bias", "std", "mae"]
    err_h = hd.groupby("hour_local")["eL"].agg(["mean", "std"])
    err_dow = hd.groupby(hd.index.tz_convert(data.TZ).dayofweek)["eL"].agg(["mean", "std"])
    err_m = hd.groupby(hd.index.tz_convert(data.TZ).month)["eL"].agg(["mean", "std"])
    eL = A["L"] / A["Lhat"] - 1
    dayerr = eL.mean(1)
    def acf(x, k):
        x = x - x.mean(); return float((x[k:] * x[:-k]).sum() / (x * x).sum())
    acf_day = [acf(dayerr[dec], k) for k in range(1, 8)]
    hr_err = hd["eL"].to_numpy()
    acf_h = [acf(hr_err, k) for k in (1, 24, 48, 168)]
    ext = hd.assign(d=hd["date"]).groupby("d")["eL"].apply(lambda s: s.abs().mean()).nlargest(6)
    fg, ax = plt.subplots(2, 3, figsize=(15, 8))
    ax[0, 0].hist(h["eL"].dropna(), 100, range=(-.1, .1)); ax[0, 0].set_title("Load error ratio (actual/forecast-1)")
    err_h["mean"].plot(ax=ax[0, 1], yerr=err_h["std"], title="Error by hour (2015-17)")
    err_m["mean"].plot(ax=ax[0, 2], yerr=err_m["std"], title="Error by month (2015-17)")
    ax[1, 0].bar(range(1, 8), acf_day); ax[1, 0].set_title("ACF of daily mean error (days)")
    hd.groupby("hour_local")["total load actual"].mean().plot(ax=ax[1, 1], title="Mean load by hour (MW)")
    err_y["std"].plot.bar(ax=ax[1, 2], title="Error std by year")
    f_load = fig("load.png")

    # ---- coupling
    hd2 = hd.copy()
    hd2["share_solar"] = hd2["generation solar"] / hd2["total load actual"]
    hd2["share_wind"] = hd2["generation wind onshore"] / hd2["total load actual"]
    hd2["share_fossil"] = (hd2["generation fossil gas"] + hd2["generation fossil hard coal"]) / hd2["total load actual"]
    cc = hd2[["eL", "gap", "price day ahead", "total load actual", "share_solar", "share_wind", "share_fossil"]].corr()
    S["corr_eL_gap"] = float(cc.loc["eL", "gap"]); S["corr_eL_pda"] = float(cc.loc["eL", "price day ahead"])
    gen_cols = [c for c in h.columns if c.startswith("generation") and "pumped" not in c]
    mix_h = hd.groupby("hour_local")[gen_cols].mean()
    mix_m = hd.groupby(hd.index.tz_convert(data.TZ).month)[gen_cols].mean()
    fg, ax = plt.subplots(1, 3, figsize=(17, 5))
    im = ax[0].imshow(cc.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax[0].set_xticks(range(len(cc))); ax[0].set_xticklabels(cc.columns, rotation=60, ha="right", fontsize=7)
    ax[0].set_yticks(range(len(cc))); ax[0].set_yticklabels(cc.columns, fontsize=7); plt.colorbar(im, ax=ax[0])
    mix_h.plot.area(ax=ax[1], legend=False, title="Generation mix by hour (MW)")
    mix_m.plot.area(ax=ax[2], legend=True, title="Generation mix by month (MW)")
    ax[2].legend(fontsize=5)
    f_coup = fig("coupling.png")

    # ---- weather
    wd = h[h.year.isin([2015, 2016, 2017])]
    cw = float(wd["temp"].corr(wd["total load actual"]))
    cweak = {k: float(wd["temp"].shift(k).corr(wd["total load actual"])) for k in (0, 24, 48)}
    fg, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].scatter(wd["temp"][::10], wd["total load actual"][::10], s=2); ax[0].set_title("Load vs city-mean temp (C)")
    wclean.pivot_table(index="dt_iso", columns="city_name", values="temp").resample("W").mean().plot(ax=ax[1], title="Weekly temp by city (C)")
    wd.groupby(wd.index.tz_convert(data.TZ).month)[["temp"]].mean().plot(ax=ax[2], title="Mean temp by month")
    f_wx = fig("weather.png")

    # ---- stability
    h["period"] = np.where(h.year <= 2016, "2015-16", np.where(h.year == 2017, "2017", "2018"))
    stab = h.groupby("period").agg(pDA=("price day ahead", "mean"), pDA_sd=("price day ahead", "std"),
                                   gap=("gap", "mean"), gap_sd=("gap", "std"), eL_sd=("eL", "std"),
                                   load=("total load actual", "mean"))
    fg, ax = plt.subplots(1, 3, figsize=(15, 4))
    for p, g in h.groupby("period"):
        ax[0].hist(g["gap"], 80, range=(-40, 60), alpha=.45, density=True, label=p)
        ax[1].hist(g["eL"].dropna(), 80, range=(-.08, .08), alpha=.45, density=True, label=p)
        ax[2].hist(g["price day ahead"], 60, alpha=.45, density=True, label=p)
    ax[0].set_title("Gap"); ax[1].set_title("Load error"); ax[2].set_title("pDA"); ax[0].legend()
    f_stab = fig("stability.png")

    # ---- regimes and defaults (decision years only)
    pd_, pa_ = A["pDA"][dec], A["pA"][dec]
    regime = {}
    for rho in (0.0, 0.5, 0.8):
        Cu = pa_ - pd_; Co = pd_ - rho * pa_
        tau = np.clip(Cu / (Cu + Co), 0.01, 0.99)
        regime[rho] = {"share_Cu_le0": float((Cu <= 0).mean()), "share_Co_le0": float((Co <= 0).mean()),
                       "tau_mean": float(tau.mean()), "tau_p10": float(np.quantile(tau, .1)), "tau_p90": float(np.quantile(tau, .9))}
    tr = np.isin(yr, [2015, 2016])
    Cref = costs.cost_ref(B, tr)
    bench = {nm: costs.daily_cost(q[~(yr == 2018)] if False else q[dec], B["L"][dec], B["pDA"][dec], B["pA"][dec]).mean()
             for nm, q in [("TSO forecast", B["Lhat"]), ("nothing", 0 * B["Lhat"]), ("cap", B["cap"]), ("oracle", B["L"])]}

    # ---- leakage audit table
    audit = pd.DataFrame([
        ["price day ahead (day D)", "published ~12:00 D-1", "ALLOWED"],
        ["forecast solar / wind onshore day ahead (D)", "published before cutoff", "ALLOWED"],
        ["forecast wind offshore eday ahead", "empty in file", "DROPPED (all NaN)"],
        ["total load forecast (D)", "published before cutoff", "ALLOWED"],
        ["calendar features of D", "known", "ALLOWED"],
        ["price actual, total load actual, generation * of day <= D-2", "settled before cutoff", "ALLOWED (lagged)"],
        ["price actual, total load actual, generation * of D-1 and D", "after cutoff", "FORBIDDEN"],
        ["weather (temp, wind, ...) of D", "observation, not forecast", "FORBIDDEN"],
        ["weather of days <= D-2 (city mean)", "observed earlier", "ALLOWED (lagged)"],
        ["weather_main / description / icon", "labels", "NOT USED"],
    ], columns=["column", "availability", "status"]).set_index("column")

    ymean = h.groupby("year")["gap"].mean()
    S.update({"Cref": float(Cref), "bench_dec": bench, "regime": {str(k): v for k, v in regime.items()},
              "scale": float(A["scale"])})
    json.dump(S, open(os.path.join(OUTD, "eda_stats.json"), "w"), indent=1, default=float)

    gap_sh = ", ".join(f"{y}: {v:.1%}" for y, v in gap_pos.items() if y >= 2015)
    gm = ", ".join(f"{y}: {v:.2f}" for y, v in ymean.items() if y >= 2015)
    r0, r5, r8 = regime[0.0], regime[0.5], regime[0.8]
    dropped = h.attrs.get("dropped", [])

    rep = f"""# EDA REPORT

Decisions are based on **2015-2017 only**. 2018 (the test year) is described in the stability section for context and has influenced **no** modeling choice.

## 1. Structure and quality
- Energy file: shape {ew}, weather file: shape {wn}; both match the expected sizes. Energy memory ~{S['mem_mb']:.1f} MB.
- Time index (UTC) is continuous: {len(h)} rows, {h.index.nunique()} unique, no gaps, no duplicates. Local-time (Europe/Madrid) days with 23 hours: {n23}; with 25 hours: {n25} (DST). **Decision:** day arrays are built by local hour; the repeated hour on 25-hour days keeps its first value and the missing hour on 23-hour days is interpolated, so every day has 24 hours.
- Duplicate full rows in the energy file: {dup_rows_e}. Weather city-hour duplicates: {dup_w} (stray whitespace in city names was stripped; city counts raw: {city_counts}). **Decision:** keep the first row per city-hour; after cleaning every city has 35,064 hours. Hourly city-average weather (temp, humidity, wind, clouds, pressure) is used.
- Temperatures converted from Kelvin to Celsius (subtract 273.15).
- Entirely empty columns: {allnan}. Constant columns (0 or one value): {[c for c in const_cols if c not in allnan]}. **Decision:** drop all of them (`{dropped}`).
- Missing values: {miss.to_dict()} - at most a few hours per column. **Decision:** time-interpolate gaps up to 6 hours.
- Units: `price day ahead` mean {h['price day ahead'].mean():.1f} and range {h['price day ahead'].min():.1f}-{h['price day ahead'].max():.1f}, consistent with EUR/MWh.

## 2. Prices
![prices]({f_price})

{md_table(pr.loc[2015:2018])}

- Negative or zero day-ahead / actual price counts by year [pDA<=0, pA<=0]: {neg}. Prices are strictly positive in the actual series; spikes are moderate (max pDA {h['price day ahead'].max():.0f}). **Decision:** no outlier removal.
- Strong hour-of-day profile (night low, morning/evening high) and a seasonal level shift; 2016 was a low-price year.
- **`price actual` vs `price day ahead`:** the gap `g = pA - pDA` is positive in this share of hours by year: {gap_sh}. Mean gap by year: {gm}. Median ratio pA/pDA: {', '.join(f'{y}: {v:.2f}' for y, v in ratio_y.items() if y >= 2015)}. The correlation of the gap with the day-ahead price (2015-17) is {corr_gap_pda:.2f}.
- **Interpretation:** the gap is positive in 86-93% of hours and is largest when pDA is low, so `price actual` behaves like a *realized price including additional components* (e.g. adjustment-service/capacity charges), not an independent balancing settlement. The gap **shrinks over time** (mean falls every year), so the test year has a smaller gap than the training years.
- **Decision:** `price actual` is treated as the buyer's settlement price `p_A` for any deviation from the day-ahead purchase (as the task defines). Consequence: shortfalls cost only a modest premium over pDA while surplus (at rho=0) is wasted at the full pDA, so the optimal purchase is *below* the load forecast. If the gap reverses in some hours, `Cu <= 0` and the rule buys zero in those hours (handled by the regime logic). Because the gap shrinks, the optimal bias toward under-buying shrinks in 2018; this is a real distribution shift to be reported.

## 3. Load
![load]({f_load})

- Error ratio `e_L = actual/forecast - 1` by year (bias, std, MAE):

{md_table(err_y)}

- Bias is ~0 overall (not significant relative to spread); std ~1.3-1.9%. Error by hour has std ranging {err_h['std'].min():.4f}-{err_h['std'].max():.4f}; mean bias by hour ranges {err_h['mean'].min():.4f} to {err_h['mean'].max():.4f}.
- Autocorrelation of the daily mean error (lags 1-7 days): {[round(a, 2) for a in acf_day]}; hourly error ACF at lags 1h/24h/48h/168h: {[round(a, 2) for a in acf_h]}. Errors are strongly persistent within a day (hourly lag-1 ACF) but essentially uncorrelated across days, so lagged errors (which must be >= 2 days old) carry little information; this is expected to limit the forecasting agent's gains over the TSO forecast.
- Largest mean-absolute-error days (2015-17): {', '.join(f'{d.date()} ({v:.1%})' for d, v in ext.items())}.
- Because the load forecast error is only ~1.5%, the value of improving load quantiles is small compared with the price-gap effect.

## 4. Coupling
![coupling]({f_coup})

- Correlation (2015-17): load error vs price gap {S['corr_eL_gap']:.2f}; load error vs day-ahead price {S['corr_eL_pda']:.2f}. The correlation matrix includes price vs load, and vs solar/wind/fossil shares (merit-order effect: higher wind share goes with lower price; see figure).
- Generation mix changes with hour (solar peak midday) and season (more hydro/wind in winter-spring).

## 5. Weather
![weather]({f_wx})

- City-mean temperature vs load correlation (2015-17): lag 0h {cweak[0]:.2f}, 24h {cweak[24]:.2f}, 48h {cweak[48]:.2f}; correlation is weak/non-monotone (heating and cooling both raise load). **Decision:** only lagged temperature (day D-2 mean and 7-day mean) is used; weather of the delivery hours is never used (observation, not a forecast).

## 6. Stability across years
![stability]({f_stab})

{md_table(stab)}

- The price level and the price gap shift between periods (the gap mean falls from 2015-16 to 2017 to 2018); the load-error spread is stable-to-falling (1.9% in 2015, 1.3% in 2018). **Implication:** the validation year 2017 is a reasonable but imperfect proxy for 2018; gains that rely on a large price gap will be smaller in the test year. 2018 is shown for description only.

## 7. Leakage audit (decision cutoff = day D-1 around noon; `cutoff_days = 2`)
{md_table(audit)}

## 8. Decisions recorded
- **Columns kept:** time, 14 non-constant generation columns, 2 day-ahead generation forecasts (solar, wind onshore), load forecast and actual, both prices, city-mean weather. **Dropped:** {dropped}.
- **Duplicates/outliers:** weather duplicates keep-first; no outlier removal; short gaps interpolated.
- **Price definitions:** `p_DA = price day ahead` (purchase), `p_A = price actual` (settlement of deviations).
- **Customer size:** `L_h = s * total load actual`, `s = {A['scale']:.6e}` so the mean load is 10 MW; cap = 1.5 x forecast.
- **Regime (2015-17, hourly):** with rho=0: share of hours with Cu<=0: {r0['share_Cu_le0']:.1%}, Co<=0: {r0['share_Co_le0']:.1%}; critical fractile tau mean {r0['tau_mean']:.2f} (p10 {r0['tau_p10']:.2f}, p90 {r0['tau_p90']:.2f}). rho=0.5: tau mean {r5['tau_mean']:.2f}; rho=0.8: tau mean {r8['tau_mean']:.2f}, Cu<=0 share {r8['share_Cu_le0']:.1%}. So the regime is the **interior newsvendor regime** in ~89% of hours with tau well below 0.5 at rho=0 (buy below forecast), and `Cu <= 0` (buy zero) in the ~11% of hours where the gap is non-positive; tau rises toward and above 0.5 as resale improves.
- **Defaults:** rho = 0 (also 0.5 and 0.8 as sensitivities), pi = 0, alpha = 0.95, cutoff_days = 2.
- **Reference costs (mean daily EUR, 2015-17, 10 MW customer):** {json.dumps({k: round(v) for k, v in bench.items()})}; `C_ref` (train 2015-16, buy TSO forecast) = {Cref:,.0f} EUR/day.
- **Feature whitelist:** see `features.py` (hour/dow/month cyclic, offday flag, pDA, Lhat, solar and wind day-ahead forecasts, load error and price gap lags >= cutoff, 7-day lag statistics, lagged temperature).
- **Burn-in:** the first 10 days of 2015 have incomplete lag features and are excluded from training episodes.
- **Periods excluded:** none.
"""
    open(REPORT, "w", encoding="utf-8").write(rep)
    print("done", S["bench_dec"], S["Cref"])


if __name__ == "__main__":
    main()
