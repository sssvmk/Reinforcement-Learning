"""Algorithm-level evaluation: learning curves, sample efficiency, stability, optimality gap, baseline statistics, gradient quality
(variance / SNR / cosine / bias), sensitivity. Writes ./outputs/<mode>/eval/.

python evaluate_reinforce.py --quick|--full
"""
import argparse
import glob
import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import load_config, outdir, md
import policies as P
import setup_problems as sp
import stats as st
from reinforce.algorithm import grad_estimate
from reinforce.baselines import BaselineState, rewards_to_go

TOL = 0.05


def classify(c, j0, jref, sigma=None):
    """Stability class of one run from its validation curve c (array of val_det)."""
    c = np.asarray(c, float)
    if (not np.isfinite(c).all()) or c[-1] < j0 - 1.0 or c.min() < -1e8:
        return "diverged"
    best = c.max()
    if c[-1] < j0 + 0.5 * (best - j0) and np.argmax(c) < 0.5 * len(c) and best > j0:
        return "collapsed"
    if best < j0 + 0.5 * (jref - j0):
        return "stalled"
    tgt = j0 + 0.95 * (jref - j0)
    tail = c[int(0.75 * len(c)):]
    if tail.min() >= tgt - TOL * abs(jref - j0):
        return "converged"
    return "unconverged"


def first_hit(eps, c, thr):
    h = np.where(np.asarray(c) >= thr)[0]
    return float(eps[h[0]]) if len(h) else np.nan


def curve_tables(runs, out, key):
    rows, stab, curves = [], [], []
    for pol, g in runs.groupby("policy"):
        M = g.groupby(["method", "episodes"]).val_det.mean().unstack(0)
        j0 = float(g[g.episodes == g.episodes.min()].val_det.mean())
        jref = float(M.max().max())
        for meth, gm in g.groupby("method"):
            seeds = {s: x.sort_values("episodes") for s, x in gm.groupby("seed")}
            fin, best, r90, r95, cls = [], [], [], [], []
            for s, x in seeds.items():
                c, e = x.val_det.to_numpy(), x.episodes.to_numpy()
                fin.append(c[-1]), best.append(c.max())
                r90.append(first_hit(e, c, j0 + 0.9 * (jref - j0))), r95.append(first_hit(e, c, j0 + 0.95 * (jref - j0)))
                cls.append(classify(c, j0, jref))
            mc = gm.groupby("episodes").val_det.agg(["mean", "std"])
            tot = mc.index.max()
            at = lambda f: float(mc["mean"].iloc[(np.abs(mc.index - f * tot)).argmin()])
            lo, hi = st.bootstrap_ci(fin)
            rows.append({"key": key, "policy": pol, "method": meth, "n_seeds": len(seeds), "J0": j0, "final_val_mean": np.mean(fin), "final_val_sd": np.std(fin, ddof=1) if len(fin) > 1 else np.nan,
                         "final_ci_lo": lo, "final_ci_hi": hi, "best_val_mean": np.mean(best), "val@10%": at(.1), "val@25%": at(.25), "val@50%": at(.5), "val@100%": at(1.0),
                         "episodes_to_90": np.nanmean(r90) if np.isfinite(r90).any() else np.nan, "reached_90_share": float(np.isfinite(r90).mean()),
                         "episodes_to_95": np.nanmean(r95) if np.isfinite(r95).any() else np.nan, "reached_95_share": float(np.isfinite(r95).mean()),
                         "optimality_gap_vs_best_known": jref - np.mean(best), "J_best_known": jref})
            for c_ in ("diverged", "collapsed", "stalled", "converged", "unconverged"):
                stab.append({"key": key, "policy": pol, "method": meth, "class": c_, "share": float(np.mean([x == c_ for x in cls]))})
            curves.append(mc.assign(method=meth, policy=pol).reset_index())
        # plot
        fig, ax = plt.subplots(figsize=(7, 4))
        for meth, gm in g.groupby("method"):
            mc = gm.groupby("episodes").val_det.agg(["mean", "std"])
            ax.plot(mc.index, mc["mean"], label=meth, ls="--" if meth.startswith("cm:") else "-")
            ax.fill_between(mc.index, mc["mean"] - mc["std"], mc["mean"] + mc["std"], alpha=0.08)
        lo_ = j0 + 0.0
        ax.set_ylim(min(j0, jref) - 0.2 * abs(jref - j0) - 0.1, jref + 0.15 * abs(jref - j0) + 0.1)
        ax.set_xlabel("training episodes / rollouts")
        ax.set_ylabel("validation J (norm. units)")
        ax.set_title("%s %s" % (key[:30], pol), fontsize=9)
        ax.legend(fontsize=6, ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "curves_%s_%s.png" % (key, pol)), dpi=110)
        plt.close(fig)
    pd.concat(curves).to_csv(os.path.join(out, "curves_%s.csv" % key), index=False)
    return pd.DataFrame(rows), pd.DataFrame(stab)


def baseline_stats(runs, key):
    rows = []
    for pol, g in runs.groupby("policy"):
        best = g.groupby(["method", "seed"]).val_det.max().unstack(0)
        if "reinforce:none" not in best:
            continue
        ps, names = [], []
        for m in best.columns:
            if m.startswith("reinforce:") and m != "reinforce:none":
                ps.append(st.wilcoxon(best[m], best["reinforce:none"])), names.append(m)
        adj = st.holm(ps) if ps else []
        for m, p, pa in zip(names, ps, adj):
            d = (best[m] - best["reinforce:none"])
            rows.append({"key": key, "policy": pol, "method": m, "vs": "none", "mean_diff": d.mean(), "p": p, "p_holm": pa})
        if "reinforce:loo_time" in best:
            ps, names = [], []
            for m in best.columns:
                if m.startswith("cm:"):
                    ps.append(st.wilcoxon(best["reinforce:loo_time"], best[m])), names.append(m)
            adj = st.holm(ps) if ps else []
            for m, p, pa in zip(names, ps, adj):
                rows.append({"key": key, "policy": pol, "method": "reinforce:loo_time", "vs": m, "mean_diff": (best["reinforce:loo_time"] - best[m]).mean(), "p": p, "p_holm": pa})
    return pd.DataFrame(rows)


def gradient_quality(cfg, mode, out, n_rep=200, B=32):
    """Variance / SNR / cosine / relative error / bias of each baseline's gradient estimator at fixed theta (initial and trained)."""
    rows = []
    mdir = outdir(mode, "models")
    for dom, key, kinds in (("inventory", cfg["inventory"]["series"][0], ["I1", "I2"]), ("trading", "TECHM", ["T1"])):
        try:
            pr = sp.get_problem(cfg, dom, key)
        except Exception as e:
            print("gradient quality skipped:", dom, e)
            continue
        gamma = cfg[dom]["gamma"]
        for kind in kinds:
            pol = P.make_stoch(dom, kind)
            pts = {"initial": pol.init(sp.seed_rng(0, key, kind))}
            mf = os.path.join(mdir, "%s__%s__reinforce-loo_time.json" % (key, kind))
            if os.path.exists(mf):
                pts["trained"] = np.asarray(json.load(open(mf))["0"]["theta_best"])
            for pname, th in pts.items():
                rng = np.random.default_rng(7)
                ref = np.mean([grad_estimate(pr, pol, th, pr.sample("train", 400, rng), rng, "loo_time", gamma)[0] for _ in range(max(1, 50 * B // 400))], axis=0)
                bst = BaselineState()
                o = pr.run(pol, th, pr.sample("train", 400, rng), rng, "sample", record=True)
                bst.fit(o["phi"], rewards_to_go(o["rew"] / pr.norm, gamma), pr.H)
                res = {}
                for bl in ("none", "loo_const", "loo_time", "linear", "loo_time_std"):
                    gs = np.array([grad_estimate(pr, pol, th, pr.sample("train", B, rng), rng, bl, gamma, 0.0, bst)[0] for _ in range(n_rep)])
                    res[bl] = gs
                none_mean, none_se = res["none"].mean(0), res["none"].std(0, ddof=1) / np.sqrt(n_rep)
                for bl, gs in res.items():
                    m, tr = gs.mean(0), gs.var(0, ddof=1).sum()
                    cos = (gs @ ref) / (np.linalg.norm(gs, axis=1) * np.linalg.norm(ref) + 1e-12)
                    se = gs.std(0, ddof=1) / np.sqrt(n_rep)
                    z = np.sqrt(((m - none_mean) ** 2).sum() / max((se ** 2).sum() + (none_se ** 2).sum(), 1e-30))
                    rows.append({"key": key, "policy": kind, "theta": pname, "baseline": bl, "trace_cov": tr, "snr": float((m ** 2).sum() / tr),
                                 "cos_to_ref": float(cos.mean()), "rel_l2_to_ref": float(np.linalg.norm(gs - ref, axis=1).mean() / np.linalg.norm(ref)),
                                 "bias_vs_none_z": float(z), "var_ratio_vs_none": float(tr / res["none"].var(0, ddof=1).sum())})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "gradient_quality.csv"), index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--quick", action="store_true")
    g.add_argument("--full", action="store_true")
    a = ap.parse_args()
    mode = "quick" if a.quick else "full"
    print("[%s] estimated runtime ~%s" % (mode, "2 min" if mode == "full" else "1 min"))
    cfg = load_config()
    out = outdir(mode, "eval")
    S, ST, BS = [], [], []
    for f in sorted(glob.glob(os.path.join(outdir(mode, "runs"), "runs_*.csv"))):
        key = os.path.basename(f).split("__", 1)[1][:-4]
        runs = pd.read_csv(f)
        s, stb = curve_tables(runs, out, key)
        S.append(s), ST.append(stb), BS.append(baseline_stats(runs, key))
    S, ST, BS = pd.concat(S), pd.concat(ST), pd.concat(BS)
    S.to_csv(os.path.join(out, "summary.csv"), index=False)
    ST.to_csv(os.path.join(out, "stability.csv"), index=False)
    BS.to_csv(os.path.join(out, "paired_tests.csv"), index=False)
    GQ = gradient_quality(cfg, mode, out)
    sens = os.path.join(outdir(mode, "runs"), "sensitivity.csv")
    SE = None
    if os.path.exists(sens):
        SE = pd.read_csv(sens).groupby(["domain", "key", "policy", "variable", "value"]).val_J.agg(["mean", "std", "count"]).reset_index()
        SE.to_csv(os.path.join(out, "sensitivity_summary.csv"), index=False)
    L = ["# REINFORCE evaluation (%s)\n" % mode, "## Summary (validation J, normalised units; selection by validation only)\n",
         md(S[["key", "policy", "method", "n_seeds", "final_val_mean", "final_val_sd", "best_val_mean", "episodes_to_90", "reached_90_share", "optimality_gap_vs_best_known"]]),
         "\n## Stability shares\n", md(ST[ST.share > 0].pivot_table(index=["key", "policy", "method"], columns="class", values="share", fill_value=0).reset_index()),
         "\n## Paired tests (Wilcoxon over seeds, Holm within family)\n", md(BS),
         "\n## Gradient quality (200 estimates of B=32; reference = loo_time, 50x batch)\n", md(GQ) if len(GQ) else "n/a\n"]
    if SE is not None:
        L += ["\n## Sensitivity (exploratory)\n", md(SE)]
    open(os.path.join(out, "EVALUATION.md"), "w", encoding="utf8").write("\n".join(L))
    print("wrote", out)


if __name__ == "__main__":
    sys.exit(main())
