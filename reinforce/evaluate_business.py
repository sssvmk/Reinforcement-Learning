"""Business evaluation on the TEST period (never used for selection): inventory profit/service metrics and trading return/risk metrics,
benchmarks tuned on train (and on validation, labelled), integer vs fractional orders, bootstrap CIs, trading walk-forward.

python evaluate_business.py --quick|--full
"""
import argparse
import copy
import json
import os
import sys
import numpy as np
import pandas as pd

from common import load_config, outdir, md
import policies as P
import setup_problems as sp
import problems as pb
import trading_env as te
import stats as st
from reinforce.algorithm import reinforce


def load_models(mdir, key, kind, meth):
    f = os.path.join(mdir, "%s__%s__%s.json" % (key, kind, meth.replace(":", "-")))
    return json.load(open(f)) if os.path.exists(f) else None


def pick_best_baseline(mdir, key, kind):
    best, bv = None, -np.inf
    for b in sp.BASELINES:
        m = load_models(mdir, key, kind, "reinforce:" + b)
        if m:
            v = np.mean([x["best_val"] for x in m.values()])
            if v > bv:
                best, bv = "reinforce:" + b, v
    return best


def inventory_block(cfg, mode, key, out, mdir):
    pr = sp.get_problem(cfg, "inventory", key)
    starts = pr.grid("test", 10)
    strategies = []                      # (label, policy, [theta...], mode, tag)
    for kind in sp.POLICIES["inventory"]:
        for meth in ("reinforce:loo_time", pick_best_baseline(mdir, key, kind)):
            m = load_models(mdir, key, kind, meth) if meth else None
            if m:
                lab = "R-%s(%s)" % (kind, meth.split(":")[1])
                if not any(s[0] == lab for s in strategies):
                    strategies.append((lab, P.make_stoch("inventory", kind), [np.asarray(v["theta_best"]) for v in m.values()], "det", "reinforce"))
    for kind in sp.CM_POLICIES["inventory"]:
        for meth in ("cm:random", "cm:neldermead", "cm:spsa", "cm:central"):
            m = load_models(mdir, key, kind, meth)
            if m:
                strategies.append(("R-%s(%s)" % (kind, meth.split(":")[1]), P.make_stoch("inventory", kind), [np.asarray(v["theta_best"]) for v in m.values()], "mean", "mean-policy method"))
    for tune in ("train", "val"):
        for nm, (pol, th) in pb.inventory_benchmarks(pr, tune).items():
            if tune == "val" and nm == "cover":
                continue
            strategies.append(("%s%s" % (nm if tune == "train" else nm + "_VALTUNED", ""), pol, [th], "mean", "benchmark" if tune == "train" else "benchmark tuned on VALIDATION (not a fair baseline)"))
    ref = None
    rows = []
    perwin = {}
    for lab, pol, ths, md_, tag in strategies:
        for integer in (False, True):
            D = [pr.run(pol, th, starts, np.random.default_rng(0), md_, detail=True, integer=integer) for th in ths]
            met = [pb.inventory_metrics(d, pr.meta["mean_price"]) for d in D]
            prof = np.mean([m["profit"] for m in met], axis=0)       # per-window, averaged over seeds
            r = {"strategy": lab, "kind": tag, "integer_orders": integer, "n_params_sets": len(ths), "profit": prof.mean(), "profit_sd_over_seeds": np.std([m["profit"].mean() for m in met], ddof=1) if len(ths) > 1 else 0.0}
            lo, hi = st.bootstrap_ci(prof)
            r.update(profit_ci_lo=lo, profit_ci_hi=hi)
            for k in ("fill_rate", "stockout_days", "avg_inventory", "orders_per_window", "order_cv", "bullwhip", "holding", "stockout_cost", "fixed_cost", "purchase", "revenue"):
                r[k] = float(np.mean([np.mean(m[k]) for m in met]))
            rows.append(r)
            if not integer:
                perwin[lab] = prof
    df = pd.DataFrame(rows)
    base = perwin.get("sS")
    if base is not None:
        for i, r in df.iterrows():
            if r.strategy in perwin and not r.integer_orders:
                m, lo, hi, p = st.paired_bootstrap(perwin[r.strategy] - base)
                df.loc[i, ["vs_tuned_sS_diff", "vs_tuned_sS_lo", "vs_tuned_sS_hi", "vs_tuned_sS_p"]] = [m, lo, hi, p]
    df.to_csv(os.path.join(out, "inventory_test_%s.csv" % key), index=False)
    return df


def trading_replay(pr, pol, th, md_):
    lo, hi = pr.splits["test"]
    o = te.simulate(pol, th, pr.env, np.array([lo]), hi - 1 - lo, np.random.default_rng(0), md_, detail=True)
    return o["net"][0], o["pos"][0], o["gross"][0]


def trading_block(cfg, mode, out, mdir):
    key = "TECHM"
    pr = sp.get_problem(cfg, "trading", key)
    strategies = []
    for kind in sp.POLICIES["trading"]:
        for meth in ("reinforce:loo_time", pick_best_baseline(mdir, key, kind)):
            m = load_models(mdir, key, kind, meth) if meth else None
            if m:
                lab = "R-%s(%s)" % (kind, meth.split(":")[1])
                if not any(s[0] == lab for s in strategies):
                    strategies.append((lab, P.make_stoch("trading", kind), [np.asarray(v["theta_best"]) for v in m.values()], "det"))
    for meth in ("cm:random", "cm:neldermead", "cm:spsa", "cm:central"):
        m = load_models(mdir, key, "T1", meth)
        if m:
            strategies.append(("R-T1(%s)[position p]" % meth.split(":")[1], P.make_stoch("trading", "T1"), [np.asarray(v["theta_best"]) for v in m.values()], "mean"))
    for nm, (pol, th) in pb.trading_benchmarks(pr).items():
        strategies.append((nm, pol, [th], "mean"))
    rows, nets = [], {}
    for lab, pol, ths, md_ in strategies:
        rs = [trading_replay(pr, pol, th, md_) for th in ths]
        met = [pb.trading_metrics(*r) for r in rs]
        r = {"strategy": lab, "n_seeds": len(ths)}
        for k in met[0]:
            vals = [m[k] for m in met]
            r[k] = float(np.nanmean(vals))
            r[k + "_sd"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
        nets[lab] = np.mean([x[0] for x in rs], axis=0)
        rows.append(r)
    df = pd.DataFrame(rows)
    bh = nets["buy_and_hold"]
    for i, r in df.iterrows():
        d, lo, hi = st.block_bootstrap_diff(nets[r.strategy], bh)
        df.loc[i, ["sum_excess_vs_bh", "sum_excess_lo", "sum_excess_hi"]] = [d, lo, hi]
        lo2, hi2 = st.bootstrap_ci(nets[r.strategy] , n=1000)
    # validation vs test ranking sanity
    df.to_csv(os.path.join(out, "trading_test_TECHM.csv"), index=False)
    return df, pr


def walk_forward(cfg, mode, out, mdir, pr_full):
    """Expanding-window walk-forward for R-T1 (loo_time): retrain on [0, k) (75% train / 25% validation), test on the next chunk."""
    data = sp.trading_data(cfg)
    n = len(data.close)
    hp = json.load(open(os.path.join(mdir, "hparams_rf.json")))["trading|TECHM|T1"]
    folds = [(int(n * a), int(n * b)) for a, b in ((0.5, 0.625), (0.625, 0.75), (0.75, 0.875), (0.875, 1.0))]
    n_seeds, eps = (3, 6000) if mode == "quick" else (5, 20000)
    pol = P.make_stoch("trading", "T1")
    rows = []
    for fi, (a, b) in enumerate(folds):
        s = int(a * 0.75)
        spl = {"train": (0, s), "val": (s, a), "test": (a, b)}
        pr = pb.build_trading(data, "TECHM", cfg, None, splits=spl)
        res = []
        for sd in range(n_seeds):
            rng = sp.seed_rng(sd, "wf", fi)
            r = reinforce(pr, pol, pol.init(rng), rng, eps, lr=hp["lr"], B=hp["B"], ent_coef=hp["ent"], baseline="loo_time", gamma=cfg["trading"]["gamma"], keep_theta=False)
            res.append(r["theta_best"])
        nets = []
        for th in res:
            net, pos, gross = trading_replay(pr, pol, th, "det")
            nets.append(net)
        bh = pr.env.R[a:b - 1]
        rows.append({"fold": fi, "test_start_idx": a, "test_end_idx": b, "days": b - 1 - a, "R_T1_net": np.mean(nets, axis=0), "bh": bh})
    net = np.concatenate([r["R_T1_net"] for r in rows])
    bh = np.concatenate([r["bh"] - 0.0 for r in rows]).copy()
    bh[[0]] -= 5e-4                      # entry cost, as in the pooled replay
    df = pd.DataFrame([{"fold": r["fold"], "days": r["days"], "R_T1_cum": float(np.prod(1 + r["R_T1_net"]) - 1), "buy_hold_cum": float(np.prod(1 + r["bh"]) - 1)} for r in rows])
    df.loc[len(df)] = {"fold": "pooled", "days": len(net), "R_T1_cum": float(np.prod(1 + net) - 1), "buy_hold_cum": float(np.prod(1 + bh) - 1)}
    df.to_csv(os.path.join(out, "trading_walkforward_TECHM_T1.csv"), index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--quick", action="store_true")
    g.add_argument("--full", action="store_true")
    a = ap.parse_args()
    mode = "quick" if a.quick else "full"
    print("[%s] estimated runtime ~%s" % (mode, "3 min" if mode == "full" else "1 min"))
    cfg = load_config()
    out, mdir = outdir(mode, "business"), outdir(mode, "models")
    L = ["# Business evaluation, TEST period (%s)\n" % mode]
    for key in sp.problem_list(cfg, "inventory"):
        if os.path.exists(os.path.join(outdir(mode, "runs"), "runs_inventory__%s.csv" % key)):
            df = inventory_block(cfg, mode, key, out, mdir)
            L += ["\n## Inventory %s (mean 90-day window profit; fractional orders)\n" % key,
                  md(df[~df.integer_orders][["strategy", "kind", "profit", "profit_ci_lo", "profit_ci_hi", "fill_rate", "avg_inventory", "orders_per_window", "vs_tuned_sS_diff", "vs_tuned_sS_p"]])]
    if os.path.exists(os.path.join(outdir(mode, "runs"), "runs_trading__TECHM.csv")):
        df, pr = trading_block(cfg, mode, out, mdir)
        L += ["\n## Trading TECHM, contiguous test replay (adjusted close; 5 bps)\n", md(df[["strategy", "n_seeds", "cum_return", "sharpe", "max_drawdown", "turnover_per_year", "exposure", "sum_excess_vs_bh", "sum_excess_lo", "sum_excess_hi"]])]
        wf = walk_forward(cfg, mode, out, mdir, pr)
        L += ["\n## Walk-forward (expanding window) R-T1 vs buy-and-hold\n", md(wf)]
    open(os.path.join(out, "BUSINESS.md"), "w", encoding="utf8").write("\n".join(L))
    print("wrote", out)


if __name__ == "__main__":
    sys.exit(main())
