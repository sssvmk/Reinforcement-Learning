"""Business evaluation of TRPO on the TEST period (never used for selection). Re-uses the metric code of evaluate_business.py.
Strategies: TRPO (tuned), VPG+GAE ablation, REINFORCE (loo_time, re-used from outputs/full), benchmarks tuned on train, plus a walk-forward for R-T1.

python evaluate_business_trpo.py --quick|--full
Outputs: outputs/trpo/<mode>/business_eval
"""
import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
from common import load_config, outdir, md, OUT
import policies as P
import setup_problems as sp
import problems as pb
import stats as st
from trpo.algorithm import trpo
from evaluate_business import trading_replay


def models(d, key, kind, meth):
    f = os.path.join(d, "%s__%s__%s.json" % (key, kind, meth))
    return json.load(open(f)) if os.path.exists(f) else None


def sources(mode):
    t = outdir("trpo", mode, "models")
    r = os.path.join(OUT, "full", "models")
    return [("TRPO", t, "trpo-trpo"), ("VPG+GAE", t, "trpo-vpg"), ("NG-fixed", t, "trpo-ng_fixed"), ("REINFORCE(loo_time)", r, "reinforce-loo_time")]


def inventory_block(cfg, mode, key, out):
    pr = sp.get_problem(cfg, "inventory", key)
    starts = pr.grid("test", 10)
    strat = []
    for kind in ("I1", "I2", "I3"):
        for lab, d, meth in sources(mode):
            m = models(d, key, kind, meth)
            if m:
                strat.append(("R-%s %s" % (kind, lab), P.make_stoch("inventory", kind), [np.asarray(v["theta_best"]) for v in m.values()], "det", "learned"))
    for nm, (pol, th) in pb.inventory_benchmarks(pr, "train").items():
        strat.append((nm, pol, [th], "mean", "benchmark (tuned on train)"))
    rows, perwin = [], {}
    for lab, pol, ths, md_, tag in strat:
        for integer in (False, True):
            D = [pr.run(pol, th, starts, np.random.default_rng(0), md_, detail=True, integer=integer) for th in ths]
            met = [pb.inventory_metrics(x, pr.meta["mean_price"]) for x in D]
            prof = np.mean([m["profit"] for m in met], axis=0)
            lo, hi = st.bootstrap_ci(prof)
            r = {"strategy": lab, "kind": tag, "integer_orders": integer, "n_seeds": len(ths), "profit": prof.mean(), "ci_lo": lo, "ci_hi": hi,
                 "seed_sd": np.std([m["profit"].mean() for m in met], ddof=1) if len(ths) > 1 else 0.0}
            for k in ("fill_rate", "avg_inventory", "orders_per_window", "order_cv", "bullwhip"):
                r[k] = float(np.mean([np.mean(m[k]) for m in met]))
            rows.append(r)
            if not integer:
                perwin[lab] = prof
    df = pd.DataFrame(rows)
    base = perwin.get("sS")
    if base is not None:
        for i, r in df.iterrows():
            if r.strategy in perwin and not r.integer_orders:
                df.loc[i, ["vs_sS_diff", "vs_sS_lo", "vs_sS_hi", "vs_sS_p"]] = st.paired_bootstrap(perwin[r.strategy] - base)
    df.to_csv(os.path.join(out, "inventory_test_%s.csv" % key), index=False)
    return df


def trading_block(cfg, mode, out):
    key = "TECHM"
    pr = sp.get_problem(cfg, "trading", key)
    strat = []
    for kind in ("T1", "T2"):
        for lab, d, meth in sources(mode):
            m = models(d, key, kind, meth)
            if m:
                strat.append(("R-%s %s" % (kind, lab), P.make_stoch("trading", kind), [np.asarray(v["theta_best"]) for v in m.values()], "det"))
    for nm, (pol, th) in pb.trading_benchmarks(pr).items():
        strat.append((nm, pol, [th], "mean"))
    rows, nets = [], {}
    for lab, pol, ths, md_ in strat:
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
    for i, r in df.iterrows():
        df.loc[i, ["sum_excess_vs_bh", "sum_excess_lo", "sum_excess_hi"]] = st.block_bootstrap_diff(nets[r.strategy], nets["buy_and_hold"])
    df.to_csv(os.path.join(out, "trading_test_TECHM.csv"), index=False)
    return df


def walk_forward(cfg, mode, out):
    data = sp.trading_data(cfg)
    n = len(data.close)
    hp = json.load(open(os.path.join(outdir("trpo", mode, "models"), "hparams_trpo.json")))["trading|TECHM|T1"]["trpo"]
    folds = [(int(n * a), int(n * b)) for a, b in ((0.5, 0.625), (0.625, 0.75), (0.75, 0.875), (0.875, 1.0))]
    n_seeds, eps = (3, 6000) if mode == "quick" else (5, 20000)
    pol = P.make_stoch("trading", "T1")
    rows = []
    for fi, (a, b) in enumerate(folds):
        s = int(a * 0.75)
        pr = pb.build_trading(data, "TECHM", cfg, None, splits={"train": (0, s), "val": (s, a), "test": (a, b)})
        nets = []
        for sd in range(n_seeds):
            rng = sp.seed_rng(sd, "wf", fi)
            r = trpo(pr, pol, pol.init(rng), rng, eps, B=32, delta=hp["delta"], lam=hp["lam"], gamma=cfg["trading"]["gamma"], diag=False)
            nets.append(trading_replay(pr, pol, r["theta_best"], "det")[0])
        rows.append({"fold": fi, "days": b - 1 - a, "net": np.mean(nets, axis=0), "bh": pr.env.R[a:b - 1]})
    net = np.concatenate([r["net"] for r in rows])
    bh = np.concatenate([r["bh"] for r in rows]).copy()
    bh[0] -= 5e-4
    df = pd.DataFrame([{"fold": r["fold"], "days": r["days"], "TRPO_T1_cum": float(np.prod(1 + r["net"]) - 1), "buy_hold_cum": float(np.prod(1 + r["bh"]) - 1)} for r in rows])
    df.loc[len(df)] = {"fold": "pooled", "days": len(net), "TRPO_T1_cum": float(np.prod(1 + net) - 1), "buy_hold_cum": float(np.prod(1 + bh) - 1)}
    df.to_csv(os.path.join(out, "trading_walkforward_TECHM_T1.csv"), index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--quick", action="store_true")
    g.add_argument("--full", action="store_true")
    mode = "quick" if ap.parse_args().quick else "full"
    cfg = load_config()
    out = outdir("trpo", mode, "business_eval")
    L = ["# TRPO business evaluation, TEST period (%s)\n" % mode,
         "Learned policies: mean over seeds of per-window results; benchmarks tuned on train. REINFORCE rows are re-used models from outputs/full.\n"]
    for key in sp.problem_list(cfg, "inventory"):
        df = inventory_block(cfg, mode, key, out)
        L += ["\n## Inventory %s (mean 90-day window profit; fractional orders)\n" % key,
              md(df[~df.integer_orders][["strategy", "kind", "n_seeds", "profit", "ci_lo", "ci_hi", "seed_sd", "fill_rate", "avg_inventory", "orders_per_window", "vs_sS_diff", "vs_sS_p"]]),
              "\nInteger-rounded orders:\n", md(df[df.integer_orders][["strategy", "profit", "fill_rate"]])]
    df = trading_block(cfg, mode, out)
    L += ["\n## Trading TECHM, contiguous test replay (adjusted close; 5 bps)\n", md(df[["strategy", "n_seeds", "cum_return", "sharpe", "max_drawdown", "turnover_per_year", "exposure", "sum_excess_vs_bh", "sum_excess_lo", "sum_excess_hi"]])]
    L += ["\n## Walk-forward (expanding window) TRPO R-T1 vs buy-and-hold\n", md(walk_forward(cfg, mode, out))]
    open(os.path.join(out, "BUSINESS.md"), "w", encoding="utf8").write("\n".join(L))
    print("wrote", out)


if __name__ == "__main__":
    sys.exit(main())
