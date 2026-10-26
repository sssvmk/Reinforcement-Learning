"""Neural REINFORCE (R-I4, R-T3) vs FDPG and vs equal-budget comparison methods on the SAME neural policy.
(a) Within this project: REINFORCE (3 baselines) vs SPSA / random search on identical networks; seeds 0..N-1 share the initial parameters
    and windows (paired), tuning effort per method comes from small validation grids (see run_reinforce.py).
(b) Across projects: FDPG's saved thetas (best estimator per policy family by validation) re-evaluated in THIS environment (same windows, Q_max,
    normaliser, validation / test) vs the neural policies. Seeds are not paired across projects (Mann-Whitney); FDPG files are only read.
python compare_neural_fdpg.py --quick|--full      (writes outputs/neural_<mode>/compare/)"""
import argparse
import json
import os
import sys
import numpy as np
import pandas as pd

from common import load_config, resolve_fdpg_dir, outdir, md
import policies as P
import setup_problems as sp
import stats as st
from compare_with_fdpg import eval_theta, fdpg_models, hash_tree


def load(d, key, kind, meth):
    f = os.path.join(d, "%s__%s__%s.json" % (key, kind, meth))
    return json.load(open(f)) if os.path.exists(f) else None


def evaluate(pr, pol, m, mode):
    v, t = zip(*[eval_theta(pr, pol, np.asarray(x["theta_best"]), mode) for x in m.values()])
    return list(v), list(t)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--quick", action="store_true")
    g.add_argument("--full", action="store_true")
    a = ap.parse_args()
    mode = "quick" if a.quick else "full"
    cfg = load_config()
    nm = outdir("neural_" + mode, "models")
    out = outdir("neural_" + mode, "compare")
    fd = resolve_fdpg_dir(cfg)
    before = hash_tree(fd) if fd else None
    print("[neural_%s] estimated runtime ~2 min" % mode)
    rows, tests = [], []
    for dom in ("inventory", "trading"):
        nk = "I4" if dom == "inventory" else "T3"
        for key in sp.problem_list(cfg, dom):
            pr = sp.get_problem(cfg, dom, key)
            npol = P.make_stoch(dom, nk)
            res = {}
            for meth, emode in (("reinforce-none", "det"), ("reinforce-loo_time", "det"), ("reinforce-linear", "det"), ("cm-spsa", "mean"), ("cm-random", "mean")):
                m = load(nm, key, nk, meth)
                if m:
                    res[meth] = evaluate(pr, npol, m, emode)
            fres = {}
            if fd:
                for kind in (["I1", "I2", "I3"] if dom == "inventory" else ["T1", "T2"]):
                    fm = fdpg_models(fd, dom, key, kind)
                    if not fm:
                        continue
                    fpol = P.make_fdpg_policy(dom, kind)
                    best = None
                    for est, seeds in fm.items():
                        v, t = zip(*[eval_theta(pr, fpol, np.asarray(th, float), "mean") for th in seeds.values()])
                        if best is None or np.mean(v) > np.mean(best[1]):
                            best = (est, list(v), list(t))
                    fres["FDPG F-%s (%s)" % (kind, best[0])] = (best[1], best[2])
            for lab, (v, t) in {**{"neural " + k: x for k, x in res.items()}, **fres}.items():
                rows.append({"domain": dom, "key": key, "method": lab, "n_seeds": len(v), "val_J": np.mean(v), "val_sd": np.std(v, ddof=1) if len(v) > 1 else np.nan,
                             "test": np.mean(t), "test_sd": np.std(t, ddof=1) if len(t) > 1 else np.nan, "test_unit": "profit/window" if dom == "inventory" else "cum. return"})
            ref_lab = max(res, key=lambda k: np.mean(res[k][0])) if res else None       # best neural variant on VALIDATION
            if ref_lab:
                rv, rt = res[ref_lab]
                for lab, (v, t) in {**{"neural " + k: x for k, x in res.items() if k != ref_lab}, **fres}.items():
                    paired = lab.startswith("neural") and len(v) == len(rv)
                    tests.append({"domain": dom, "key": key, "reference": "neural " + ref_lab + " (best val)", "vs": lab, "val_diff": float(np.mean(rv) - np.mean(v)),
                                  "test_diff": float(np.mean(rt) - np.mean(t)), "test_p": st.mannwhitney(rt, t), "test_p_paired_wilcoxon": st.wilcoxon(rt, t) if paired else np.nan,
                                  "cliffs_test": st.cliffs_delta(rt, t)})
    D, T = pd.DataFrame(rows), pd.DataFrame(tests)
    if len(T):
        T["test_p_holm"] = st.holm(T.test_p.fillna(1).to_numpy())
    D.to_csv(os.path.join(out, "neural_vs_fdpg_summary.csv"), index=False)
    T.to_csv(os.path.join(out, "neural_vs_fdpg_tests.csv"), index=False)
    L = ["# Neural REINFORCE vs equal-budget comparison methods and FDPG\n",
         "Neural rows: R-I4 / R-T3 (PyTorch MLP). `cm-spsa` / `cm-random`: equal-rollout-budget derivative-free methods on the same network (mean policy), same seeds and initial parameters as REINFORCE. "
         "FDPG rows: FDPG's saved thetas (best estimator by validation) re-evaluated in this environment (Q_max = 7 mu for inventory, same windows and normaliser). "
         "Validation J is in norm units; test is profit per 90-day window (inventory) or contiguous cumulative return (TECHM).\n",
         md(D), "\n## Tests: best neural variant on validation vs each other row (Mann-Whitney; paired Wilcoxon where seeds are matched)\n", md(T) if len(T) else "n/a",
         "\nCaveats: FDPG policy families differ (F-I2 is (s,S), F-I3 calendar base-stock, F-T1 sigmoid position); tuning effort is not matched across projects; "
         "Nelder-Mead and central FD were dropped for the ~250-parameter network (infeasible at equal budget).\n"]
    open(os.path.join(out, "NEURAL_VS_FDPG.md"), "w", encoding="utf8").write("\n".join(L))
    if fd:
        assert hash_tree(fd) == before, "FDPG folder changed!"
        print("FDPG folder unchanged (%d files)" % len(before))
    print(D.round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
