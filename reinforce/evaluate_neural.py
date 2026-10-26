"""Neural policies (R-I4, R-T3): validation summary and TEST-period results next to the linear REINFORCE policies and benchmarks.
python evaluate_neural.py --quick|--full   (reads outputs/neural_<mode>/ and outputs/<mode>/; writes outputs/neural_<mode>/eval/)"""
import argparse
import glob
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


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--quick", action="store_true")
    g.add_argument("--full", action="store_true")
    a = ap.parse_args()
    mode = "quick" if a.quick else "full"
    print("[%s] estimated runtime ~1 min" % mode)
    cfg = load_config()
    nm, lm = outdir("neural_" + mode, "models"), outdir(mode, "models")
    out = outdir("neural_" + mode, "eval")
    rows = []
    raw = {}
    for dom in ("inventory", "trading"):
        for key in sp.problem_list(cfg, dom):
            pr = sp.get_problem(cfg, dom, key)
            lin = {"inventory": ["I2", "I3"], "trading": ["T1", "T2"]}[dom]
            kinds = [("I4" if dom == "inventory" else "T3", nm)] + [(k, lm) for k in lin]
            for kind, d in kinds:
                for f in sorted(glob.glob(os.path.join(d, "%s__%s__%s.json" % (key, kind, "*" if d == nm else "reinforce-*")))):
                    meth = os.path.basename(f).split("__")[2][:-5]
                    if kind in ("I2", "I3", "T1", "T2") and meth not in ("reinforce-loo_time", "reinforce-none", "reinforce-linear"):
                        continue
                    m = json.load(open(f))
                    pol = P.make_stoch(dom, kind)
                    emode = "mean" if meth.startswith("cm-") else "det"
                    vals, tests = [], []
                    for v in m.values():
                        th = np.asarray(v["theta_best"])
                        vals.append(v["best_val"])
                        if dom == "inventory":
                            tests.append(pr.run(pol, th, pr.grid("test", 10), np.random.default_rng(0), emode)["total"].mean())
                        else:
                            lo, hi = pr.splits["test"]
                            o = te.simulate(pol, th, pr.env, np.array([lo]), hi - 1 - lo, np.random.default_rng(0), emode, detail=True)
                            tests.append(float(np.prod(1 + o["net"][0]) - 1))
                    lo_, hi_ = st.bootstrap_ci(tests)
                    rows.append({"key": key, "policy": kind, "baseline": meth.replace("reinforce-", ""), "n_seeds": len(m), "val_J_mean": np.mean(vals), "val_J_sd": np.std(vals, ddof=1),
                                 "test_mean": np.mean(tests), "test_ci_lo": lo_, "test_ci_hi": hi_, "test_unit": "profit/window" if dom == "inventory" else "cum. return"})
                    raw[(key, kind, meth)] = (vals, tests)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "neural_vs_linear.csv"), index=False)
    T = []
    for (key, kind, meth), (vals, tests) in raw.items():
        if kind in ("I4", "T3") and meth == "reinforce-loo_time":
            for (k2, kind2, meth2), (v2, t2) in raw.items():
                if k2 == key and kind2 != kind and meth2 == "reinforce-loo_time":
                    T.append({"key": key, "neural": kind, "vs": kind2, "val_p": st.mannwhitney(vals, v2), "test_p": st.mannwhitney(tests, t2),
                              "test_diff": float(np.mean(tests) - np.mean(t2)), "cliffs_test": st.cliffs_delta(tests, t2)})
    T = pd.DataFrame(T)
    if len(T):
        T["test_p_holm"] = st.holm(T.test_p.fillna(1).to_numpy())
    T.to_csv(os.path.join(out, "neural_vs_linear_tests.csv"), index=False)
    L = ["# Neural (R-I4, R-T3) vs linear REINFORCE policies and equal-budget comparison methods on the neural policy\n", md(df),
         "\n## Mann-Whitney: neural vs linear (loo_time baseline, over seeds)\n", md(T) if len(T) else "n/a"]
    sf = os.path.join(outdir("neural_" + mode, "runs"), "sensitivity.csv")
    if os.path.exists(sf):
        s = pd.read_csv(sf)
        SE = s.groupby(["policy", "variable", "value"]).agg(n=("val_J", "size"), val_J_mean=("val_J", "mean"), val_J_sd=("val_J", "std")).reset_index()
        SE.to_csv(os.path.join(out, "sensitivity_summary.csv"), index=False)
        L += ["\n## Sensitivity (exploratory; fewer seeds and episodes than the linear study)\n", md(SE)]
    open(os.path.join(out, "NEURAL.md"), "w", encoding="utf8").write("\n".join(L) + "\n")
    print(df.round(3).to_string())


if __name__ == "__main__":
    sys.exit(main())
