"""Q_max-free REINFORCE variant (R-I1 / R-I2 trained and evaluated WITHOUT the order cap, as FDPG did) vs the capped REINFORCE policies
(Q_max = 7 mu) and vs FDPG's saved FD thetas, all evaluated in the Q_max-free environment on identical windows.
python evaluate_qfree.py      (reads outputs/qfree/ and outputs/full/; FDPG files are only read as text/JSON; writes outputs/qfree/eval/)"""
import json
import os
import sys
import numpy as np
import pandas as pd

from common import load_config, resolve_fdpg_dir, outdir, md
import policies as P
import setup_problems as sp
import stats as st
from compare_with_fdpg import eval_theta, fdpg_models


def load(d, key, kind, meth):
    f = os.path.join(d, "%s__%s__%s.json" % (key, kind, meth))
    return json.load(open(f)) if os.path.exists(f) else None


def main():
    cfg = load_config()
    qd, cd = outdir("qfree", "models"), outdir("full", "models")
    out = outdir("qfree", "eval")
    fd = resolve_fdpg_dir(cfg)
    print("[qfree] estimated runtime ~2 min")
    rows, tests = [], []
    for key in sp.problem_list(cfg, "inventory"):
        pr = sp.get_problem(cfg, "inventory", key)
        mu = pr.meta["mu"]
        for kind in ("I1", "I2"):
            pol = P.make_stoch("inventory", kind)
            res = {}
            for lab, d, meth in (("REINFORCE Q_max-free (loo_time)", qd, "reinforce-loo_time"), ("REINFORCE Q_max-free (none)", qd, "reinforce-none"),
                                 ("REINFORCE capped 7mu (loo_time)", cd, "reinforce-loo_time"), ("REINFORCE capped 7mu (none)", cd, "reinforce-none")):
                m = load(d, key, kind, meth)
                if not m:
                    continue
                v, t, big = [], [], []
                for x in m.values():
                    th = np.asarray(x["theta_best"])
                    vj, tv = eval_theta(pr, pol, th, "det", qmax=None)        # same environment for every row: no cap
                    v.append(vj); t.append(tv)
                    import copy
                    pr2 = copy.copy(pr); pr2.env = pr.env.with_qmax(None)
                    D = pr2.run(pol, th, pr2.grid("test", 10), np.random.default_rng(0), "det", detail=True)
                    big.append(float((D["order"] > 7 * mu).mean()))
                res[lab] = (v, t)
                rows.append({"key": key, "policy": kind, "method": lab, "n_seeds": len(m), "val_J_nocap_env": np.mean(v), "val_J_sd": np.std(v, ddof=1),
                             "test_profit_nocap_env": np.mean(t), "test_sd": np.std(t, ddof=1), "share_orders_above_7mu": np.mean(big)})
            fm = fdpg_models(fd, "inventory", key, kind) if fd else None
            if fm and kind in ("I1", "I2"):
                fpol = P.make_fdpg_policy("inventory", kind)
                best = None
                for est, seeds in fm.items():
                    vt = [eval_theta(pr, fpol, np.asarray(th, float), "mean", qmax=None) for th in seeds.values()]
                    v, t = zip(*vt)
                    if best is None or np.mean(v) > np.mean(best[1]):
                        best = (est, v, t)
                res["FDPG best estimator"] = (best[1], best[2])
                rows.append({"key": key, "policy": kind, "method": "FDPG F-%s (%s, by validation)" % (kind, best[0]), "n_seeds": len(best[1]), "val_J_nocap_env": np.mean(best[1]),
                             "val_J_sd": np.std(best[1], ddof=1), "test_profit_nocap_env": np.mean(best[2]), "test_sd": np.std(best[2], ddof=1), "share_orders_above_7mu": np.nan})
            ref = res.get("REINFORCE Q_max-free (loo_time)")
            for lab, (v, t) in res.items():
                if ref is not None and lab != "REINFORCE Q_max-free (loo_time)":
                    tests.append({"key": key, "policy": kind, "vs": lab, "val_p": st.mannwhitney(ref[0], v), "test_p": st.mannwhitney(ref[1], t),
                                  "test_diff_(qfree_minus_other)": float(np.mean(ref[1]) - np.mean(t)), "cliffs_test": st.cliffs_delta(ref[1], t)})
    df, T = pd.DataFrame(rows), pd.DataFrame(tests)
    if len(T):
        T["test_p_holm"] = st.holm(T.test_p.fillna(1).to_numpy())
    df.to_csv(os.path.join(out, "qmax_free_summary.csv"), index=False)
    T.to_csv(os.path.join(out, "qmax_free_tests.csv"), index=False)
    L = ["# Q_max-free REINFORCE (R-I1, R-I2)\n",
         "All rows are evaluated in the SAME environment (no order cap, same validation / test windows, same normaliser): validation J in norm units, test profit per 90-day window. "
         "FDPG rows use FDPG's saved thetas (mean-policy mode) of the estimator that was best on validation here; FDPG's tuning effort and policy parameterisation differ, and seeds are not paired.\n",
         md(df), "\n## Mann-Whitney: Q_max-free REINFORCE (loo_time) vs each other row (over seeds)\n", md(T) if len(T) else "n/a"]
    open(os.path.join(out, "QMAX_FREE.md"), "w", encoding="utf8").write("\n".join(L) + "\n")
    print(df.round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
