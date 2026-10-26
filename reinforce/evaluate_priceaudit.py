"""Price-scale robustness for TECHM: (1) quantify how `close` (adjusted) differs from the unadjusted OHLC scale, (2) compare REINFORCE / buy-and-hold results
when returns come from adjusted close (default) vs next-open-to-open on the unadjusted scale ('open'; corporate-action days fall back to close).
python evaluate_priceaudit.py     (reads outputs/full/ and outputs/ropen/; writes outputs/ropen/eval/)"""
import json
import os
import sys
import numpy as np
import pandas as pd

from common import load_config, outdir, md
import data_trading as dt
import policies as P
import problems as pb
import trading_env as te
import setup_problems as sp
import stats as st


def replay(pr, pol, th, mode):
    lo, hi = pr.splits["test"]
    o = te.simulate(pol, th, pr.env, np.array([lo]), hi - 1 - lo, np.random.default_rng(0), mode, detail=True)
    return float(np.prod(1 + o["net"][0]) - 1)


def main():
    cfg = load_config()
    out = outdir("ropen", "eval")
    print("[ropen] estimated runtime ~1 min")
    df = dt.read_trading_safe(cfg["trading_files"][0])
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    d = pd.to_datetime(df["tdate"])
    rc = c[2:] / c[1:-1] - 1
    ro = o[2:] / o[1:-1] - 1
    lg = np.log(1 + ro) - np.log(1 + rc)
    bad = np.abs(lg) > 0.08
    ratio = c / ((h + l) / 2)
    yr = d.dt.year.to_numpy()
    A = pd.DataFrame({"year": np.unique(yr), "close_over_mid_median": [np.median(ratio[yr == y]) for y in np.unique(yr)]})
    A["ratio_change_vs_prev_year"] = A.close_over_mid_median.pct_change()
    summ = pd.DataFrame([{"corr_open_vs_close_daily_returns": float(np.corrcoef(ro[~bad], rc[~bad])[0, 1]), "share_days_flagged_corporate_action": float(bad.mean()),
                          "n_flagged": int(bad.sum()), "median_abs_log_gap_unflagged": float(np.median(np.abs(lg[~bad]))),
                          "ann_vol_close": float(rc.std() * np.sqrt(252)), "ann_vol_open": float(ro[~bad].std() * np.sqrt(252))}])
    flagged = pd.DataFrame({"date": d.iloc[1:-1][bad].dt.date.astype(str).to_numpy(), "ret_close": rc[bad], "ret_open": ro[bad]})
    A.to_csv(os.path.join(out, "close_over_mid_by_year.csv"), index=False)
    flagged.to_csv(os.path.join(out, "flagged_days.csv"), index=False)

    data = {s: dt.build_features(df, cfg["trading"]["warmup_rows"], cfg["trading"]["allow_adjusted_close"], return_source=s) for s in ("close", "open")}
    prs = {s: pb.build_trading(data[s], "TECHM", cfg, cfg["splits"]) for s in data}
    rows, raw = [], {}
    for s in ("close", "open"):
        bh, _ = prs[s], None
        lo, hi = prs[s].splits["test"]
        rows.append({"env_returns": s, "policy_trained_on": "-", "strategy": "buy_and_hold", "n_seeds": 1, "test_cum_return": replay(prs[s], P.FixedPos(1.0, "bh"), np.zeros(0), "mean"), "sd": 0.0})
        pol, th = pb.trading_benchmarks(prs[s])["RSI_rule"]
        rows.append({"env_returns": s, "policy_trained_on": "train of env", "strategy": "RSI_rule(train-tuned)", "n_seeds": 1, "test_cum_return": replay(prs[s], pol, th, "mean"), "sd": 0.0})
    srcs = {"close": outdir("full", "models"), "open": outdir("ropen", "models")}
    for trained, md_ in srcs.items():
        for kind in ("T1", "T2"):
            for meth in ("reinforce-loo_time", "reinforce-none", "cm-central", "cm-spsa"):
                f = os.path.join(md_, "TECHM__%s__%s.json" % (kind, meth))
                if not os.path.exists(f) or (kind == "T2" and meth.startswith("cm")):
                    continue
                m = json.load(open(f))
                pol = P.make_stoch("trading", kind)
                for env in ("close", "open"):
                    t = [replay(prs[env], pol, np.asarray(v["theta_best"]), "mean" if meth.startswith("cm") else "det") for v in m.values()]
                    raw[(env, trained, kind, meth)] = t
                    rows.append({"env_returns": env, "policy_trained_on": trained, "strategy": "R-%s(%s)" % (kind, meth), "n_seeds": len(t), "test_cum_return": float(np.mean(t)), "sd": float(np.std(t, ddof=1))})
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(out, "return_source_comparison.csv"), index=False)
    tests = []
    for (env, trained, kind, meth), t in raw.items():
        if trained == "close" and env == "close":
            o_ = raw.get(("open", "open", kind, meth))
            if o_:
                tests.append({"policy": kind, "method": meth, "close_trained_on_close_env": np.mean(t), "open_trained_on_open_env": np.mean(o_), "mannwhitney_p": st.mannwhitney(t, o_)})
    T = pd.DataFrame(tests)
    L = ["# TECHM price-scale robustness\n",
         "## Interpretation\n",
         "The close/mid ratio drifts smoothly from %.3f (%d) to %.3f (%d) with year-to-year changes of %.1f%%-%.1f%% and no single-year jump: this pattern is what a cumulative DIVIDEND adjustment "
         "(total-return series) looks like, not a split (a split would give a one-off step of 2x or more). So `close` is a dividend-adjusted series and its returns are the appropriate total-return "
         "series, while the %.0f%% cumulative gap to the unadjusted OHLC scale explains why the level audit fails. Returns from adjusted close are therefore kept as the primary choice; "
         "the 'open' variant below is a sensitivity check (it excludes dividends, so it understates total return, and open prints are noisy: %d days flagged).\n" % (
             A.close_over_mid_median.iloc[0], A.year.iloc[0], A.close_over_mid_median.iloc[-1], A.year.iloc[-1], 100 * A.ratio_change_vs_prev_year.min(), 100 * A.ratio_change_vs_prev_year.max(),
             100 * (A.close_over_mid_median.iloc[-1] / A.close_over_mid_median.iloc[0] - 1), int(bad.sum())),
         "Caveat: this is an inference from the data pattern; the data provider's adjustment method is not documented, so please confirm with the data owner.\n",
         "`close` is split/dividend-adjusted while open/high/low/sar are not (DATA_NOTES.md). Adjusted close gives the economically correct return series (corporate actions removed); "
         "the 'open' variant takes returns from next-open to following-open on the UNADJUSTED scale (execution at the next open), replacing days whose open-to-open and close-to-close ratios differ "
         "by more than 8% in log terms (splits / large corporate actions) by the close-to-close return. It is a robustness check, not a replacement of the primary results.\n",
         "## close / mid(high, low) by year (median)\n", md(A), "\n## Agreement of the two return series\n", md(summ), "\n## Flagged days\n", md(flagged) if len(flagged) else "none",
         "\n## Test-period cumulative return (contiguous replay, 5 bps cost, mean and sd over seeds)\n", md(R), "\n## Same policy family, trained and evaluated within its own return source (Mann-Whitney)\n", md(T) if len(T) else "n/a"]
    open(os.path.join(out, "PRICE_AUDIT.md"), "w", encoding="utf8").write("\n".join(L) + "\n")
    print(summ.round(4).to_string())
    print(R.round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
