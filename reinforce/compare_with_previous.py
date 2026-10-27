"""Compare TRPO with the previously built methods (REINFORCE variants and equal-budget derivative-free methods re-used from outputs/full).

python compare_with_previous.py --quick|--full
Reads outputs/trpo/<mode>/runs and outputs/full/runs; writes outputs/trpo/<mode>/comparison. Read-only on all earlier results (hash-checked).
"""
import argparse
import glob
import hashlib
import json
import os
import numpy as np
import pandas as pd
from common import outdir, md, OUT
import stats as S


def hash_dir(d):
    out = {}
    for root, _, files in os.walk(d):
        for f in files:
            p = os.path.join(root, f)
            with open(p, "rb") as fh:
                out[os.path.relpath(p, d)] = hashlib.sha256(fh.read()).hexdigest()
    return out


def read_runs(pattern):
    fs = glob.glob(pattern)
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--quick", action="store_true")
    g.add_argument("--full", action="store_true")
    ap.add_argument("--set", default="main", choices=["main", "neural", "qfree", "entropy"])
    a = ap.parse_args()
    mode = "quick" if a.quick else "full"
    prev_name = {"main": "full", "neural": "neural_full", "qfree": "qfree", "entropy": os.path.join("trpo", "full")}[a.set]
    mode = {"main": mode, "neural": "neural_" + mode, "qfree": "qfree", "entropy": "ent" if a.full else "ent_quick"}[a.set]
    REF = "trpo:trpo_ent" if a.set == "entropy" else "trpo:trpo"
    out = outdir("trpo", mode, "comparison")
    prev_dir = os.path.join(OUT, prev_name)
    h0 = hash_dir(prev_dir)
    new = read_runs(os.path.join(OUT, "trpo", mode, "runs", "runs_*.csv"))
    old = read_runs(os.path.join(prev_dir, "runs", "runs_*.csv"))
    if a.set == "entropy":
        old = old[old.method == "trpo:trpo"]
    else:
        keep = ["reinforce:loo_time", "reinforce:none", "cm:random", "cm:neldermead", "cm:spsa", "cm:fd_central"]
        old = old[old.method.isin(keep) | old.method.str.startswith("cm:")]
    old = old[old.policy.isin(new.policy.unique()) & old.key.isin(new.key.unique())]
    allr = pd.concat([new[new.method.isin(["trpo:trpo", "trpo:vpg", "trpo:ng_fixed", "trpo:trpo_ent"])], old], ignore_index=True)
    G = ["key", "policy", "method", "seed"]
    fin = allr.sort_values("episodes").groupby(G).agg(final=("val_det", "last"), first=("val_det", "first")).reset_index()
    ref = fin.groupby(["key", "policy"]).apply(lambda d: pd.Series({"ref": d.groupby("method").final.mean().max(), "v0": d["first"].mean()})).reset_index()
    fin = fin.merge(ref, on=["key", "policy"])
    e95 = []
    for r in fin.itertuples():
        d = allr[(allr.key == r.key) & (allr.policy == r.policy) & (allr.method == r.method) & (allr.seed == r.seed)].sort_values("episodes")
        h = d[d.val_det >= r.v0 + 0.95 * (r.ref - r.v0)]
        e95.append(h.episodes.iloc[0] if len(h) else np.nan)
    fin["eps95"] = e95
    fin.to_csv(os.path.join(out, "per_run_all_methods.csv"), index=False)

    rows = []
    for (key, pol), d in fin.groupby(["key", "policy"]):
        for m, dm in d.groupby("method"):
            lo, hi = S.bootstrap_ci(dm.final.values)
            rows.append({"key": key, "policy": pol, "method": m, "n": len(dm), "final": dm.final.mean(), "ci_lo": lo, "ci_hi": hi, "eps95_median": np.nanmedian(dm.eps95) if dm.eps95.notna().any() else np.nan,
                         "reached95": dm.eps95.notna().mean()})
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(out, "summary.csv"), index=False)

    tests = []
    for (key, pol), d in fin.groupby(["key", "policy"]):
        t = d[d.method == REF].set_index("seed").final
        for m in sorted(set(d.method) - {REF}):
            o = d[d.method == m].set_index("seed").final
            ss = sorted(set(t.index) & set(o.index))
            if len(ss) < 3:
                continue
            x, y = t.loc[ss].values, o.loc[ss].values
            tests.append({"key": key, "policy": pol, "vs": m, "n_pairs": len(ss), "diff_trpo_minus": float(np.mean(x - y)), "p_wilcoxon_paired": S.wilcoxon(x, y),
                          "p_mannwhitney": S.mannwhitney(x, y), "cliffs_delta": S.cliffs_delta(x, y)})
    tests = pd.DataFrame(tests)
    if len(tests):
        tests["p_holm"] = S.holm(tests.p_wilcoxon_paired.fillna(1.0).values)
        tests["verdict"] = np.where(tests.p_holm < 0.05, np.where(tests.diff_trpo_minus > 0, "TRPO better", "TRPO worse"), "no significant difference")
    tests.to_csv(os.path.join(out, "tests.csv"), index=False)

    wins = tests.groupby("vs").verdict.value_counts().unstack(fill_value=0) if len(tests) else pd.DataFrame()
    h1 = hash_dir(prev_dir)
    unchanged = h0 == h1
    txt = ["# TRPO vs previous methods (%s mode)\n" % mode,
           "Baselines are **re-used** from `outputs/%s` of this folder (same seeds, same theta0). " % prev_name +
           "TRPO counts pilot + training batches against the same budget; validation/diagnostic rollouts are excluded for all methods. "
           "Tuning effort differs: TRPO tunes delta x lambda (12 configs) and VPG lr (3); REINFORCE tuned lr x B x entropy (18).\n",
           "Previous results unchanged during comparison (hash check of outputs/full): **%s**\n" % unchanged,
           "## Summary (validation return, deterministic policy, last checkpoint)\n", md(summ), "",
           "## TRPO vs each method (paired by seed, Wilcoxon, Holm over all rows)\n", md(tests), "", "## Verdict counts\n", md(wins.reset_index()) if len(wins) else ""]
    open(os.path.join(out, "comparison.md"), "w", encoding="utf-8").write("\n".join(txt))
    assert unchanged, "previous results were modified"
    print("wrote", out)


if __name__ == "__main__":
    main()
