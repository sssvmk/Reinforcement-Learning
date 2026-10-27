"""Validation / TEST-period evaluation of the TRPO extensions (neural policies, entropy bonus, Q_max-free). EXPLORATORY.
python evaluate_extensions_trpo.py --set neural|entropy|qfree [--quick|--full]
Reads outputs/trpo/<mode>/models and earlier model folders (read-only); writes outputs/trpo/<mode>/ext_eval.
Inventory: mean 90-day test-window profit (physical units). Trading: contiguous TEST cumulative return (5 bps cost, adjusted close).
Every row of one problem is evaluated in the same environment (qfree: no order cap for all rows).
"""
import argparse
import copy
import json
import os
import sys
import numpy as np
import pandas as pd
from common import load_config, resolve_fdpg_dir, outdir, md, OUT
import policies as P
import setup_problems as sp
import problems as pb
import stats as st
from compare_with_fdpg import eval_theta, fdpg_models

DIVERGED = -1e6


def thetas(d, key, kind, meth):
    f = os.path.join(d, "%s__%s__%s.json" % (key, kind, meth))
    if not os.path.exists(f):
        return None
    return [np.asarray(v["theta_best"]) for v in json.load(open(f)).values()]


def specs(setname, mode, domain):
    T = lambda m: os.path.join(OUT, "trpo", m, "models")
    R = lambda m: os.path.join(OUT, m, "models")
    inv = domain == "inventory"
    if setname == "neural":
        nk = "I4" if inv else "T3"
        lin = ["I1", "I2", "I3"] if inv else ["T1", "T2"]
        s = [(nk, "TRPO neural", T("neural_" + mode), "trpo-trpo"), (nk, "VPG+GAE neural", T("neural_" + mode), "trpo-vpg"),
             (nk, "REINFORCE neural (loo_time)", R("neural_full"), "reinforce-loo_time"), (nk, "REINFORCE neural (none)", R("neural_full"), "reinforce-none")]
        s += [(k, "TRPO linear", T("full"), "trpo-trpo") for k in lin]
        return s, "TRPO neural"
    if setname == "entropy":
        ks = ["I1", "I2", "I3"] if inv else ["T1", "T2"]
        em = "ent" if mode == "full" else "ent_quick"
        return [(k, "TRPO + entropy bonus", T(em), "trpo-trpo_ent") for k in ks] + [(k, "TRPO (plain)", T("full"), "trpo-trpo") for k in ks], "TRPO + entropy bonus"
    ks = ["I1", "I2"]
    return ([(k, "TRPO Q_max-free", T("qfree"), "trpo-trpo") for k in ks] + [(k, "VPG+GAE Q_max-free", T("qfree"), "trpo-vpg") for k in ks] +
            [(k, "REINFORCE Q_max-free (loo_time)", R("qfree"), "reinforce-loo_time") for k in ks] + [(k, "REINFORCE Q_max-free (none)", R("qfree"), "reinforce-none") for k in ks] +
            [(k, "TRPO capped 7mu", T("full"), "trpo-trpo") for k in ks]), "TRPO Q_max-free"


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--quick", action="store_true")
    g.add_argument("--full", action="store_true")
    ap.add_argument("--set", required=True, choices=["neural", "entropy", "qfree"])
    a = ap.parse_args()
    mode = "quick" if a.quick else "full"
    mdir = {"neural": "neural_" + mode, "entropy": "ent" if a.full else "ent_quick", "qfree": "qfree"}[a.set]
    qmax = None if a.set == "qfree" else "keep"
    cfg = load_config()
    out = outdir("trpo", mdir, "ext_eval")
    fd = resolve_fdpg_dir(cfg) if a.set == "qfree" else None
    rows, tests, L = [], [], ["# TRPO extension evaluation: %s (%s) - EXPLORATORY\n" % (a.set, mode),
                              "Validation J (normalised) and TEST result, mean over seeds, deterministic policy. Rows differ in tuning effort; seeds are paired with the reference in the "
                              "optimiser (same theta0) but evaluation here uses seed-level means, so tests are Mann-Whitney over seeds. Diverged = validation J < -1e6.\n"]
    for domain in ("inventory", "trading"):
        if a.set == "qfree" and domain == "trading":
            continue
        sl, refl = specs(a.set, mode, domain)
        for key in sp.problem_list(cfg, domain):
            pr = sp.get_problem(cfg, domain, key)
            res = {}
            for kind, lab, d, meth in sl:
                ths = thetas(d, key, kind, meth)
                if not ths:
                    continue
                pol = P.make_stoch(domain, kind)
                vt = [eval_theta(pr, pol, th, "det", qmax=qmax) for th in ths]
                v, t = np.array([x[0] for x in vt]), np.array([x[1] for x in vt])
                r = {"domain": domain, "key": key, "policy": kind, "method": lab, "n_seeds": len(ths), "val_J": float(np.median(v)), "val_J_mean": float(np.mean(v)),
                     "test": float(np.median(t)), "test_mean": float(np.mean(t)), "test_sd": float(np.std(t, ddof=1)) if len(t) > 1 else 0.0, "n_diverged": int((v < DIVERGED).sum())}
                if a.set == "qfree" and domain == "inventory":
                    pr2 = copy.copy(pr)
                    pr2.env = pr.env.with_qmax(None)
                    r["share_orders_above_7mu"] = float(np.mean([(pr2.run(pol, th, pr2.grid("test", 10), np.random.default_rng(0), "det", detail=True)["order"] > 7 * pr.meta["mu"]).mean() for th in ths]))
                rows.append(r)
                res[(kind, lab)] = (v, t)
            if a.set == "qfree" and fd and domain == "inventory":
                for kind in ("I1", "I2"):
                    fm = fdpg_models(fd, "inventory", key, kind)
                    if not fm:
                        continue
                    fpol = P.make_fdpg_policy("inventory", kind)
                    best = None
                    for est, seeds in fm.items():
                        vt = [eval_theta(pr, fpol, np.asarray(th, float), "mean", qmax=None) for th in seeds.values()]
                        v, t = np.array([x[0] for x in vt]), np.array([x[1] for x in vt])
                        if best is None or np.median(v) > np.median(best[1]):
                            best = (est, v, t)
                    rows.append({"domain": domain, "key": key, "policy": kind, "method": "FDPG (%s, best on val)" % best[0], "n_seeds": len(best[1]), "val_J": float(np.median(best[1])),
                                 "val_J_mean": float(np.mean(best[1])), "test": float(np.median(best[2])), "test_mean": float(np.mean(best[2])), "test_sd": float(np.std(best[2], ddof=1)),
                                 "n_diverged": int((best[1] < DIVERGED).sum())})
            if domain == "trading":
                for nm, (pol, th) in pb.trading_benchmarks(pr).items():
                    vj, tv = eval_theta(pr, pol, th, "mean")
                    rows.append({"domain": domain, "key": key, "policy": "-", "method": nm, "n_seeds": 1, "val_J": vj, "val_J_mean": vj, "test": tv, "test_mean": tv, "test_sd": 0.0, "n_diverged": 0})
            for (kind, lab), (v, t) in res.items():
                if lab == refl:
                    continue
                rf = res.get((kind, refl))
                if rf is not None and len(v) > 2:
                    tests.append({"domain": domain, "key": key, "policy": kind, "ref": refl, "vs": lab, "val_p": st.mannwhitney(rf[0], v), "test_p": st.mannwhitney(rf[1], t),
                                  "test_median_diff": float(np.median(rf[1]) - np.median(t)), "cliffs_test": st.cliffs_delta(rf[1], t)})
    df, T = pd.DataFrame(rows), pd.DataFrame(tests)
    if len(T):
        T["test_p_holm"] = st.holm(T.test_p.fillna(1).to_numpy())
    df.to_csv(os.path.join(out, "summary.csv"), index=False)
    T.to_csv(os.path.join(out, "tests.csv"), index=False)
    L += ["Medians are reported next to means because a diverged seed ruins a mean.\n", md(df.round(4)), "\n## Reference vs each other row (Mann-Whitney over seeds, Holm over all rows)\n", md(T.round(4)) if len(T) else "n/a"]
    open(os.path.join(out, "EXT_EVAL.md"), "w", encoding="utf8").write("\n".join(L) + "\n")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
