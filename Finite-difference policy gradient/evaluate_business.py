"""Script 2: business metrics on the untouched test period (+ walk-forward for trading) -> ./outputs/business_eval/

  python evaluate_business.py --quick | --full
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import glob
import pickle
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import metrics as M
import run_fdpg as R
from audit_data import HERE, load_config
from fdpg.optimizer import HParams
from problems import inventory_benchmarks, trading_benchmarks

OUT = os.path.join(HERE, "outputs", "business_eval")
BLOCK = 20          # moving-block bootstrap block length (trading days)
KEYS = ["cum_return", "ann_return", "ann_vol", "sharpe", "sortino", "max_drawdown", "calmar", "hit_rate", "profit_factor",
        "avg_win", "avg_loss", "turnover", "cost_paid", "cost_drag", "avg_position", "exposure",
        "excess_cum_vs_benchmark", "vol_ratio_vs_benchmark", "mdd_diff_vs_benchmark"]


def fmt(m, s):
    return f"{m:.3f} ± {s:.3f}" if np.isfinite(s) and s > 0 else f"{m:.3f}"


def _md(df):
    cols = list(df.columns)
    rows = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        rows.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(rows)


def block_idx(T, n, rng, block=BLOCK):
    nb = int(np.ceil(T / block))
    st = rng.integers(0, T - block + 1, size=(n, nb))
    return (st[..., None] + np.arange(block)).reshape(n, -1)[:, :T]


def cum(x):
    return np.prod(1 + x) - 1


def sharpe(x):
    s = x.std(ddof=1)
    return x.mean() / s * np.sqrt(252) if s > 0 else np.nan


def boot_ci(series, ref, rng, n=1000):
    """95% moving-block bootstrap CIs (shared resampled days) for cum return, Sharpe, and the cum-return difference to `ref`."""
    idx = block_idx(len(series), n, rng)
    S, Rf = series[idx], ref[idx]
    a = np.array([cum(x) for x in S]); b = np.array([cum(x) for x in Rf])
    sh = np.array([sharpe(x) for x in S])
    return {"cum": np.percentile(a, [2.5, 97.5]), "sharpe": np.nanpercentile(sh, [2.5, 97.5]), "cum_minus_bh": np.percentile(a - b, [2.5, 97.5])}


def eval_strategy(prob, pol, thetas, lo, hi, bh_net):
    rows, nets = [], []
    for th in thetas:
        d = prob.detail(pol, th, lo, hi)
        rows.append(M.trading_metrics(d["net"], d["pos"], d["gross"], prob.extra["cost"], benchmark=bh_net))
        nets.append(d["net"])
    return pd.DataFrame(rows)[KEYS], np.mean(nets, axis=0), d["dates"]


def trading_split_report(cfg, series, policy, d, mode, rng, od, md):
    recs = d["records"]
    best = max(R.ESTIMATORS, key=lambda m: np.mean([r["val_final"] for r in recs if r["method"] == m]))
    thetas = [r["theta"] for r in sorted(recs, key=lambda r: r["seed"]) if r["method"] == best]
    prob = R.get_problem(cfg, "trading", series, policy, R.MODES[mode]["n_scale"])
    lo, hi = prob.splits["test"]
    bench = trading_benchmarks(prob)
    bh = prob.detail(*bench["buy_and_hold"], lo, hi)
    strategies = {f"FD-{policy}({best})": (prob.policy, thetas), **{k: (v[0], [v[1]]) for k, v in bench.items()}}
    rows, nets, rank = [], {}, []
    for name, (pol, ths) in strategies.items():
        df, net_mean, dates = eval_strategy(prob, pol, ths, lo, hi, bh["net"])
        nets[name] = net_mean
        row = {"strategy": name, "n_seeds": len(ths)}
        for k in KEYS:
            row[k + "_mean"], row[k + "_sd"] = df[k].mean(), df[k].std(ddof=1) if len(df) > 1 else 0.0
        ci = boot_ci(net_mean, bh["net"], rng)
        row.update({"cum_ci_lo": ci["cum"][0], "cum_ci_hi": ci["cum"][1], "sharpe_ci_lo": ci["sharpe"][0], "sharpe_ci_hi": ci["sharpe"][1],
                    "cum_minus_bh_ci_lo": ci["cum_minus_bh"][0], "cum_minus_bh_ci_hi": ci["cum_minus_bh"][1]})
        rows.append(row)
        rank.append({"strategy": name, "val_mean_window_return": float(np.mean([prob.window_returns(pol, th, "val").mean() for th in ths])),
                     "test_cum_return": row["cum_return_mean"]})
    pd.DataFrame(rows).to_csv(os.path.join(od, f"trading_test_{series}_{policy}.csv"), index=False)
    rk = pd.DataFrame(rank)
    rk["val_rank"] = rk.val_mean_window_return.rank(ascending=False).astype(int)
    rk["test_rank"] = rk.test_cum_return.rank(ascending=False).astype(int)
    rk.to_csv(os.path.join(od, f"ranking_sanity_{series}_{policy}.csv"), index=False)
    fig, a = plt.subplots(figsize=(9, 4.5))
    for name, net in nets.items():
        a.plot(dates, np.cumprod(1 + net), label=name)
    a.set_title(f"{series} test-period equity (block bootstrap, block={BLOCK} days)"); a.legend(fontsize=7); fig.tight_layout()
    fig.savefig(os.path.join(od, f"equity_test_{series}_{policy}.png"), dpi=110); plt.close(fig)
    show = ["cum_return", "ann_return", "ann_vol", "sharpe", "sortino", "max_drawdown", "calmar", "hit_rate", "profit_factor", "turnover", "cost_drag", "exposure"]
    t = pd.DataFrame([{"strategy": r["strategy"], **{k: fmt(r[k + "_mean"], r[k + "_sd"]) for k in show},
                       "cum CI95": f"[{r['cum_ci_lo']:.3f}, {r['cum_ci_hi']:.3f}]", "cum-B&H CI95": f"[{r['cum_minus_bh_ci_lo']:.3f}, {r['cum_minus_bh_ci_hi']:.3f}]"} for r in rows])
    top_val = rk.loc[rk.val_rank == rk.val_rank.min(), "strategy"].iloc[0]; top_test = rk.loc[rk.test_rank == rk.test_rank.min(), "strategy"].iloc[0]
    beats = [r["strategy"] for r in rows[1:] if r["cum_return_mean"] >= rows[0]["cum_return_mean"]]
    md += [f"### {series} / {policy}: test {str(dates[0])[:10]} .. {str(dates[-1])[:10]} (best FD estimator by validation: {best}; mean ± sd over {len(thetas)} seeds; cum CI = moving-block bootstrap, block {BLOCK}, on the seed-averaged daily series)",
           "", _md(t), "", "Ranking sanity check (validation window-return rank vs test rank):", _md(rk), "",
           f"Top on validation: **{top_val}**; top on test: **{top_test}** ({'agree' if top_val == top_test else 'DISAGREE'}). "
           f"Benchmarks that matched or beat the FD policy's mean test cumulative return: {beats if beats else 'none'}.", ""]
    return best


def walk_forward(cfg, series, policy, best, d, mode, rng, od, md, folds=3):
    n_scale = R.MODES[mode]["n_scale"]
    base = R.get_problem(cfg, "trading", series, policy, n_scale)
    n = base.extra["data"].n
    t0 = base.splits["test"][0]
    edges = np.linspace(t0, n, folds + 1).astype(int)
    hp = HParams(**d["hp"])
    seeds = list(range(3 if mode == "quick" else 5))
    budget = R.MODES[mode]["budget"]
    fdn = f"FD-{policy}({best})"
    names = [fdn, "buy_and_hold", "flat", "RSI_rule", "SAR_rule"]
    pooled = {k: [[] for _ in seeds] if k == fdn else [[]] for k in names}
    rows = []
    for f in range(folds):
        s0, s1 = edges[f], edges[f + 1]
        splits = {"train": (0, int(0.75 * s0)), "val": (int(0.75 * s0), s0), "test": (s0, s1)}
        prob = R.get_problem(cfg, "trading", series, policy, n_scale, splits=splits)
        thetas = [R.train_one(prob, best, 2000 + s, hp, budget)["theta"] for s in seeds]
        bench = trading_benchmarks(prob)
        bh = prob.detail(*bench["buy_and_hold"], s0, s1)["net"]
        strat = {fdn: (prob.policy, thetas), **{k: (v[0], [v[1]]) for k, v in bench.items()}}
        for name, (pol, ths) in strat.items():
            for i, th in enumerate(ths):
                dd = prob.detail(pol, th, s0, s1)
                m = M.trading_metrics(dd["net"], dd["pos"], dd["gross"], prob.extra["cost"], benchmark=bh)
                rows.append({"fold": f + 1, "start": str(dd["dates"][0])[:10], "end": str(dd["dates"][-1])[:10], "strategy": name, "seed": i, **{k: m[k] for k in KEYS}})
                pooled[name][i].append((dd["net"], dd["pos"]))
    wf = pd.DataFrame(rows)
    bhn = np.concatenate([c[0] for c in pooled["buy_and_hold"][0]])
    prow = []
    for name in names:
        ms = [M.trading_metrics(np.concatenate([c[0] for c in ch]), np.concatenate([c[1] for c in ch]), None, base.extra["cost"], benchmark=bhn) for ch in pooled[name]]
        df = pd.DataFrame(ms)
        prow.append({"fold": "pooled", "strategy": name, **{k: df[k].mean() for k in KEYS}, **{k + "_sd": df[k].std(ddof=1) if len(df) > 1 else 0.0 for k in KEYS}})
    wf.to_csv(os.path.join(od, f"trading_walkforward_{series}_{policy}.csv"), index=False)
    pd.DataFrame(prow).to_csv(os.path.join(od, f"trading_walkforward_pooled_{series}_{policy}.csv"), index=False)
    g = wf.groupby(["fold", "strategy"])[["cum_return", "sharpe", "max_drawdown"]].mean().reset_index()
    show = ["cum_return", "ann_return", "sharpe", "sortino", "max_drawdown", "calmar", "exposure"]
    pt = pd.DataFrame([{"strategy": r["strategy"], **{k: fmt(r[k], r[k + "_sd"]) for k in show}} for r in prow])
    md += [f"### {series} / {policy}: walk-forward ({folds} expanding folds over the test region, {len(seeds)} seeds; each fold retrains on all prior data, last 25% of it for validation)",
           "", "Per fold (mean over seeds):", _md(g), "", "Pooled out-of-sample:", _md(pt), ""]


def inventory_report(cfg, mode, rng, od, md):
    md.append("## Inventory")
    series = R.list_series(cfg, "inventory")
    if not series:
        md += ["PENDING: M5 `sell_prices.csv` is missing (see DATA_NOTES.md); no inventory results were produced.", ""]
        return
    for sid in series:
        for policy in R.POLICIES["inventory"]:
            f = os.path.join(HERE, "outputs", "runs", f"inventory__{sid}__{policy}__{mode}.pkl")
            if not os.path.exists(f):
                continue
            recs = pickle.load(open(f, "rb"))["records"]
            best = max(R.ESTIMATORS, key=lambda m: np.mean([r["val_final"] for r in recs if r["method"] == m]))
            thetas = [r["theta"] for r in sorted(recs, key=lambda r: r["seed"]) if r["method"] == best]
            prob = R.get_problem(cfg, "inventory", sid, policy, R.MODES[mode]["n_scale"])
            mp = prob.extra["price"][prob.splits["train"][0]:prob.splits["train"][1]].mean()
            strat = {f"FD-{policy}({best})": (prob.policy, thetas), **{k: (v[0], [v[1]]) for k, v in inventory_benchmarks(prob).items()}}
            rows, profits = [], {}
            for name, (pol, ths) in strat.items():
                for integer in (False, True):
                    df = pd.DataFrame([M.inventory_metrics(prob.detail(pol, th, "test", integer), mp) for th in ths])
                    rows.append({"strategy": name, "integer_orders": integer, **{k: df[k].mean() for k in df}})
                D = prob.detail(pol, ths[0], "test")
                profits[name] = (D["revenue"] - D["purchase"] - D["holding"] - D["stockout"] - D["fixed"]).sum(axis=1)
            tab = pd.DataFrame(rows)
            ref = list(profits)[0]
            for name in list(profits)[1:]:
                diff = profits[ref] - profits[name]
                bs = rng.choice(diff, size=(2000, len(diff))).mean(axis=1)
                m = (tab.strategy == name) & (~tab.integer_orders)
                tab.loc[m, "fd_improvement_abs"] = diff.mean()
                tab.loc[m, "fd_improvement_ci_lo"], tab.loc[m, "fd_improvement_ci_hi"] = np.percentile(bs, [2.5, 97.5])
                tab.loc[m, "fd_improvement_pct"] = 100 * diff.mean() / abs(profits[name].mean())
            tab.to_csv(os.path.join(od, f"inventory_test_{sid}_{policy}.csv"), index=False)
            md += [f"### Inventory {sid} / {policy}", _md(tab), ""]


def main(argv=None):
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--quick", action="store_true"); g.add_argument("--full", action="store_true")
    mode = "quick" if ap.parse_args(argv).quick else "full"
    cfg = load_config()
    od = os.path.join(OUT, mode); os.makedirs(od, exist_ok=True)
    rng = np.random.default_rng(0)
    md = [f"# Business-metrics summary ({mode})", "",
          "All costs are assumptions; the test period is never used for fitting or tuning. Trading returns use ADJUSTED `close` (price-scale audit failed; see DATA_NOTES.md).", "",
          "## Trading"]
    for f in sorted(glob.glob(os.path.join(HERE, "outputs", "runs", f"trading__*__{mode}.pkl"))):
        _, series, policy, _ = os.path.basename(f)[:-4].split("__")
        d = pickle.load(open(f, "rb"))
        print("business eval", series, policy, flush=True)
        best = trading_split_report(cfg, series, policy, d, mode, rng, od, md)
        walk_forward(cfg, series, policy, best, d, mode, rng, od, md)
    inventory_report(cfg, mode, rng, od, md)
    open(os.path.join(od, "SUMMARY.md"), "w").write("\n".join(md))
    print("wrote", os.path.join(od, "SUMMARY.md"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
