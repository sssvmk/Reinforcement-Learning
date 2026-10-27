"""Algorithm-performance evaluation of TRPO and its ablations (reads outputs/trpo/<mode>; writes outputs/trpo/<mode>/algo_eval).

python evaluate_trpo.py --quick|--full
"""
import argparse
import glob
import os
import numpy as np
import pandas as pd
from common import outdir, md
import stats as S

SEC = 3600 * 0


def load(mode):
    base = outdir("trpo", mode)
    runs = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(base, "runs", "runs_*.csv"))], ignore_index=True)
    its = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(base, "iters", "iters_*.csv"))], ignore_index=True)
    sens = os.path.join(base, "runs", "sensitivity.csv")
    return runs, its, (pd.read_csv(sens) if os.path.exists(sens) else None)


def per_run(runs, its):
    G = ["domain", "key", "policy", "method", "seed"]
    fin = runs.sort_values("episodes").groupby(G).agg(final=("val_det", "last"), first=("val_det", "first"), best=("val_det", "max")).reset_index()
    ref = fin.groupby(["key", "policy"]).agg(ref=("final", "mean"), v0=("first", "mean")).reset_index()
    fin = fin.merge(ref, on=["key", "policy"])
    # episodes to 95% of the gain from initial value to the best method's mean final value
    rows = []
    for g, d in runs.groupby(G):
        r = fin[(fin.key == g[1]) & (fin.policy == g[2]) & (fin.method == g[3]) & (fin.seed == g[4])].iloc[0]
        thr = r.v0 + 0.95 * (r.ref - r.v0)
        hit = d.sort_values("episodes")
        hit = hit[hit.val_det >= thr]
        rows.append(hit.episodes.iloc[0] if len(hit) else np.nan)
    fin["eps95"] = rows
    its = its.copy()
    its["step"] = its["ret_after"] - its["ret_before"]
    wd = its.groupby(G).agg(worst_drop=("step", "min"), frac_improved=("surr_improve", lambda x: float((x > 0).mean())),
                            mean_kl=("kl", "mean"), max_kl=("kl", "max"), mean_nbt=("nbt", "mean"), frac_acc=("accepted", "mean"),
                            t_update=("t_update", "mean"), t_collect=("t_collect", "mean"), diverged=("diverged", "max") if "diverged" in its else ("kl", "size")).reset_index()
    fin = fin.merge(wd, on=G, how="left")
    gain = fin.ref - fin.v0
    fin["class"] = np.where(~np.isfinite(fin.final) | (fin.final < -1e6), "diverge",
                    np.where(fin.final < fin.v0 - 0.05 * gain.abs(), "collapse",
                    np.where(fin.final - fin.v0 < 0.5 * gain, "stall", "converge")))
    return fin


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--quick", action="store_true")
    g.add_argument("--full", action="store_true")
    ap.add_argument("--set", default="main", choices=["main", "neural", "qfree", "entropy"])
    a = ap.parse_args()
    mode = "quick" if a.quick else "full"
    mode = {"main": mode, "neural": "neural_" + mode, "qfree": "qfree", "entropy": "ent" if a.full else "ent_quick"}[a.set]
    out = outdir("trpo", mode, "algo_eval")
    runs, its, sens = load(mode)
    fin = per_run(runs, its)
    fin.to_csv(os.path.join(out, "per_run.csv"), index=False)
    txt = ["# TRPO algorithm evaluation (%s mode)\n" % mode,
           "Return units: validation return / reward normaliser (higher is better). `final` = last checkpoint, deterministic policy.\n"]

    summ = []
    for (key, pol, m), d in fin.groupby(["key", "policy", "method"]):
        lo, hi = S.bootstrap_ci(d.final.values)
        summ.append({"key": key, "policy": pol, "method": m, "n": len(d), "final": d.final.mean(), "ci_lo": lo, "ci_hi": hi,
                     "eps95_median": np.nanmedian(d.eps95) if d.eps95.notna().any() else np.nan, "reached95": d.eps95.notna().mean(),
                     "worst_drop_median": d.worst_drop.median(), "worst_drop_min": d.worst_drop.min(), "seed_sd": d.final.std(),
                     "frac_improved": d.frac_improved.mean(), "mean_KL": d.mean_kl.mean(), "max_KL": d.max_kl.max(), "backtracks": d.mean_nbt.mean(),
                     "t_update_ms": 1e3 * d.t_update.mean(), "t_collect_ms": 1e3 * d.t_collect.mean(),
                     "converge": (d["class"] == "converge").mean(), "stall": (d["class"] == "stall").mean(),
                     "collapse": (d["class"] == "collapse").mean(), "diverge": (d["class"] == "diverge").mean()})
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(out, "summary.csv"), index=False)
    txt += ["## Per problem / policy / variant\n", md(summ[["key", "policy", "method", "n", "final", "ci_lo", "ci_hi", "eps95_median", "reached95", "worst_drop_median",
                                                          "seed_sd", "converge", "stall", "collapse", "diverge"]]), ""]
    txt += ["## Update quality (trust-region check)\n", md(summ[["key", "policy", "method", "frac_improved", "mean_KL", "max_KL", "backtracks", "t_update_ms", "t_collect_ms"]]), ""]

    # surrogate-vs-realised correlation, CG vs exact solve, KL vs delta
    q = its[its.method.isin(["trpo:trpo", "trpo:trpo_ent"])]
    qq = []
    for (key, pol), d in q.groupby(["key", "policy"]):
        dd = d.dropna(subset=["surr_improve", "ret_after", "ret_before"])
        c = float(np.corrcoef(dd.surr_improve, dd.ret_after - dd.ret_before)[0, 1]) if len(dd) > 3 else np.nan
        qq.append({"key": key, "policy": pol, "corr_surrogate_vs_realised": c, "cos_CG_vs_exact": d.cos_exact.dropna().mean(),
                   "cg_residual": d.cg_res.mean(), "vf_r2": d.vf_r2.mean(), "frac_accepted": d.accepted.mean()})
    qq = pd.DataFrame(qq)
    qq.to_csv(os.path.join(out, "update_quality.csv"), index=False)
    txt += ["## TRPO update diagnostics\n", md(qq), ""]

    # paired tests (same seeds and theta0) of each ablation vs TRPO
    tests = []
    for (key, pol), d in fin.groupby(["key", "policy"]):
        ref = d[d.method == "trpo:trpo"].sort_values("seed")
        for m in sorted(set(d.method) - {"trpo:trpo"}):
            o = d[d.method == m].sort_values("seed")
            ss = sorted(set(ref.seed) & set(o.seed))
            x, y = ref.set_index("seed").loc[ss].final.values, o.set_index("seed").loc[ss].final.values
            tests.append({"key": key, "policy": pol, "vs": m, "diff_trpo_minus": float(np.mean(x - y)), "p_wilcoxon": S.wilcoxon(x, y), "cliffs_delta": S.cliffs_delta(x, y)})
    tests = pd.DataFrame(tests)
    if len(tests):
        tests["p_holm"] = S.holm(tests.p_wilcoxon.fillna(1.0).values)
    tests.to_csv(os.path.join(out, "ablation_tests.csv"), index=False)
    txt += ["## TRPO vs ablations (paired by seed, Wilcoxon, Holm-adjusted over all rows)\n", md(tests), ""]

    # learning curves (mean over seeds)
    curves = runs.groupby(["key", "policy", "method", "episodes"]).val_det.mean().reset_index()
    curves.to_csv(os.path.join(out, "learning_curves.csv"), index=False)

    if sens is not None:
        sg = sens.groupby(["domain", "key", "policy", "variable", "value"]).val_J.agg(["mean", "std", "count"]).reset_index()
        sg.to_csv(os.path.join(out, "sensitivity_summary.csv"), index=False)
        txt += ["## Sensitivity (one at a time; best validation return, mean over seeds)\n", md(sg), ""]

    open(os.path.join(out, "algo_eval.md"), "w", encoding="utf-8").write("\n".join(txt))
    print("wrote", out)


if __name__ == "__main__":
    main()
