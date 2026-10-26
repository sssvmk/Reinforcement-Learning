"""Compare REINFORCE with the FDPG project. FDPG files are only READ as text/CSV/JSON (never imported, executed or unpickled);
every FDPG file is hashed before and after and the script asserts nothing changed.

python compare_with_fdpg.py --quick|--full
Evidence A: FDPG's own reported CSVs.  Evidence B: FDPG's saved parameters re-evaluated in THIS environment (+ reproduction check).
"""
import argparse
import hashlib
import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import load_config, resolve_fdpg_dir, outdir, md
import policies as P
import setup_problems as sp
import problems as pb
import trading_env as te
import stats as st


def hash_tree(d):
    out = {}
    for root, dirs, files in os.walk(d):
        dirs[:] = [x for x in dirs if x != "__pycache__"]
        for f in files:
            p = os.path.join(root, f)
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for blk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(blk)
            out[os.path.relpath(p, d)] = h.hexdigest()
    return out


def read_fdpg_texts(d):
    out = {}
    for root, dirs, files in os.walk(d):
        for f in files:
            if f.lower().endswith((".csv", ".json", ".yaml", ".yml", ".md", ".txt")):
                with open(os.path.join(root, f), "r", encoding="utf8", errors="ignore") as fh:
                    out[os.path.relpath(os.path.join(root, f), d)] = fh.read()
    return out


def fdpg_models(d, domain, key, kind):
    p = os.path.join(d, "outputs", "models", "%s__%s__%s__full.json" % (domain, key, kind))
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding="utf8"))


def rollouts_to(curve, frac=0.9):
    """First checkpoint where mean val J reaches J0 + frac*(Jbest - J0) (J0 = first checkpoint)."""
    j0, jb = curve.iloc[0], curve.max()
    hit = np.where(curve.to_numpy() >= j0 + frac * (jb - j0))[0]
    return float(curve.index[hit[0]]) if len(hit) else np.nan


def eval_theta(pr, pol, th, mode, split="val", qmax="keep"):
    """Validation J (windows, normalised) and test result (physical): inventory mean window profit / trading contiguous cumulative return."""
    env = pr.env
    if pr.domain == "inventory" and qmax is None:
        env = env.with_qmax(None)
    import copy
    pr2 = copy.copy(pr)
    pr2.env = env
    vj = pr2.J(pol, th, pr2.grid("val", 5 if pr.domain == "inventory" else 15), np.random.default_rng(0), mode)
    if pr.domain == "inventory":
        tv = float(pr2.run(pol, th, pr2.grid("test", 10), np.random.default_rng(0), mode)["total"].mean())
    else:
        lo, hi = pr.splits["test"]
        o = te.simulate(pol, th, env, np.array([lo]), hi - 1 - lo, np.random.default_rng(0), mode, detail=True)
        tv = float(np.prod(1 + o["net"][0]) - 1)
    return vj, tv


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--quick", action="store_true")
    g.add_argument("--full", action="store_true")
    a = ap.parse_args()
    mode = "quick" if a.quick else "full"
    print("[%s] estimated runtime ~1 min" % mode)
    cfg = load_config()
    d = resolve_fdpg_dir(cfg)
    out = outdir(mode, "compare")
    if not d or cfg["fdpg_comparison"]["enabled"] is False:
        open(os.path.join(out, "COMPARISON.md"), "w").write("FDPG folder not available: comparison PENDING\n")
        print("FDPG absent -> PENDING")
        return 0
    before = hash_tree(d)
    mdir = os.path.join(outdir(mode, "models"))
    rows_B, rows_A, repro, tests = [], [], [], []
    curves_plot = []
    domains = [x for x in ("inventory", "trading") if os.path.isdir(os.path.join(outdir(mode, "runs")))]
    for dom in ("inventory", "trading"):
        for key in sp.problem_list(cfg, dom):
            runf = os.path.join(outdir(mode, "runs"), "runs_%s__%s.csv" % (dom, key))
            if not os.path.exists(runf):
                continue
            runs = pd.read_csv(runf)
            pr = sp.get_problem(cfg, dom, key)
            for kind in sp.POLICIES[dom]:
                fm = fdpg_models(d, dom, key, kind)
                if fm is None:
                    rows_B.append({"domain": dom, "key": key, "policy": kind, "status": "PENDING (FDPG models missing)"})
                    continue
                fpol = P.make_fdpg_policy(dom, kind)
                # --- FDPG estimator per validation in THIS environment
                est_rows = []
                for est, seeds in fm.items():
                    vs, ts = zip(*[eval_theta(pr, fpol, np.asarray(th, float), "mean") for th in seeds.values()])
                    est_rows.append((est, np.mean(vs), vs, ts))
                best = max(est_rows, key=lambda r: r[1])
                # --- reproduction check (FDPG settings: Q_max=None, mean mode)
                rep = fdpg_reported(d, dom, key, kind)
                if rep is not None:
                    est_name = rep["est"] if rep["est"] in fm else best[0]
                    mine = float(np.mean([eval_theta(pr, fpol, np.asarray(th, float), "mean", qmax=None)[1] for th in fm[est_name].values()]))
                    ok = abs(mine - rep["value"]) <= 0.02 * abs(rep["value"])
                    repro.append({"domain": dom, "key": key, "policy": kind, "estimator": est_name, "metric": rep["metric"], "fdpg_reported": rep["value"],
                                  "reproduced_here": mine, "rel_diff": (mine - rep["value"]) / abs(rep["value"]), "status": "REPRODUCED" if ok else "NOT REPRODUCED"})
                # --- REINFORCE thetas
                for meth in ("reinforce:loo_time", "reinforce:none"):
                    mf = os.path.join(mdir, "%s__%s__%s.json" % (key, kind, meth.replace(":", "-")))
                    if not os.path.exists(mf):
                        continue
                    m = json.load(open(mf))
                    rpol = P.make_stoch(dom, kind)
                    rv, rt = zip(*[eval_theta(pr, rpol, np.asarray(v["theta_best"]), "det") for v in m.values()])
                    p_val = st.mannwhitney(rv, best[2])
                    p_test = st.mannwhitney(rt, best[3])
                    tests.append({"domain": dom, "key": key, "policy": kind, "method": meth, "fdpg_estimator": best[0], "val_mean_R": np.mean(rv), "val_mean_F": np.mean(best[2]),
                                  "test_mean_R": np.mean(rt), "test_mean_F": np.mean(best[3]), "p_val": p_val, "p_test": p_test, "cliffs_test": st.cliffs_delta(rt, best[3]),
                                  "same_family": kind in ("I1", "T1")})
                rows_B.append({"domain": dom, "key": key, "policy": kind, "fdpg_best_estimator": best[0], "fdpg_val_J": best[1], "fdpg_test": float(np.mean(best[3])), "status": "OK"})
                # --- Evidence A: FDPG's own curves / summary vs ours
                sm = os.path.join(d, "outputs", "algo_eval", "full", "summary_%s__%s__%s.csv" % (dom, key, kind))
                cv = os.path.join(d, "outputs", "algo_eval", "full", "curves_%s__%s__%s.csv" % (dom, key, kind))
                if os.path.exists(sm) and os.path.exists(cv):
                    s = pd.read_csv(sm)
                    s = s[s.method.isin(["forward", "central", "spsa", "regression", "gaussian"])]
                    bs = s.sort_values("final_val_mean", ascending=False).iloc[0]
                    c = pd.read_csv(cv)
                    cb = c[c.method == bs.method].set_index("rollouts").val_mean
                    mine = runs[(runs.policy == kind) & (runs.method == "reinforce:loo_time")]
                    mc = mine.groupby("episodes").val_det.mean()
                    rows_A.append({"domain": dom, "key": key, "policy": kind, "fdpg_method": bs.method, "fdpg_final_val": bs.final_val_mean,
                                   "fdpg_rollouts_to_90": rollouts_to(cb), "reinforce_final_val": float(mc.iloc[-1]), "reinforce_best_mean_val": float(mc.max()),
                                   "reinforce_rollouts_to_90": rollouts_to(mc), "caveat": "normalisation differs (Q_max, norm rule): indicative only"})
                    curves_plot.append((dom, key, kind, cb, mine))
    pd.DataFrame(rows_A).to_csv(os.path.join(out, "evidence_A_reported.csv"), index=False)
    pd.DataFrame(rows_B).to_csv(os.path.join(out, "evidence_B_fdpg_reevaluated.csv"), index=False)
    pd.DataFrame(repro).to_csv(os.path.join(out, "reproduction_check.csv"), index=False)
    T = pd.DataFrame(tests)
    if len(T):
        T["p_test_holm"] = np.nan
        for m, ix in T.groupby("method").groups.items():
            T.loc[ix, "p_test_holm"] = st.holm(T.loc[ix, "p_test"].fillna(1).to_numpy())
    T.to_csv(os.path.join(out, "mannwhitney_REINFORCE_vs_FDPG.csv"), index=False)
    # curve overlays
    for dom, key, kind, cb, mine in curves_plot:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.plot(cb.index, cb.values, label="FDPG best estimator (own normalisation)", color="C1")
        for meth, col in (("reinforce:loo_time", "C0"), ("reinforce:none", "C2")):
            g = mine[mine.method == meth].groupby("episodes").val_det
            if len(g.mean()):
                ax.plot(g.mean().index, g.mean().values, label=meth, color=col)
                ax.fill_between(g.mean().index, (g.mean() - g.std()).values, (g.mean() + g.std()).values, alpha=0.15, color=col)
        ax.set_xlabel("training rollouts / episodes")
        ax.set_ylabel("validation J (norm units)")
        ax.set_title("%s %s %s" % (dom, key[:22], kind), fontsize=9)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "overlay_%s_%s_%s.png" % (dom, key, kind)), dpi=110)
        plt.close(fig)
    L = ["# REINFORCE vs FDPG\n", "Hash check: %d FDPG files, asserted unchanged after the run.\n" % len(before),
         "## Reproduction check (FDPG saved thetas re-run here, Q_max=None, FDPG windows; tolerance 2%)\n", md(pd.DataFrame(repro)) if repro else "none\n",
         "\n## Evidence B: re-evaluated in this environment (same windows, Q_max, normalisation); REINFORCE vs FDPG-best-by-validation\n", md(T) if len(T) else "none\n",
         "\n## Evidence A: FDPG's reported results vs ours (indicative)\n", md(pd.DataFrame(rows_A)) if rows_A else "none\n",
         "\nNotes: seeds are not paired across projects (Mann-Whitney); R-I2/R-I3/R-T2 are not the same policy families as F-I2/F-I3/F-T2 (`same_family` False); "
         "FDPG used no Q_max or discount; tuning effort differs.\n"]
    open(os.path.join(out, "COMPARISON.md"), "w", encoding="utf8").write("\n".join(L))
    after = hash_tree(d)
    assert after == before, "FDPG folder changed!"
    json.dump(before, open(os.path.join(outdir("audit"), "fdpg_hashes_after_%s.json" % mode), "w"))
    print("FDPG folder unchanged (%d files). Wrote %s" % (len(after), out))
    return 0


def fdpg_reported(d, dom, key, kind):
    """FDPG's reported test number for its FD policy from business_eval/full (text CSV)."""
    base = os.path.join(d, "outputs", "business_eval", "full")
    try:
        if dom == "inventory":
            t = pd.read_csv(os.path.join(base, "inventory_test_%s_%s.csv" % (key, kind)))
            r = t[(t.strategy.str.startswith("FD-")) & (t.integer_orders == False)].iloc[0]
            return {"est": r.strategy.split("(")[1].rstrip(")"), "metric": "mean test-window profit", "value": float(r.profit)}
        t = pd.read_csv(os.path.join(base, "trading_test_%s_%s.csv" % (key, kind)))
        r = t[t.strategy.str.startswith("FD-")].iloc[0]
        return {"est": r.strategy.split("(")[1].rstrip(")"), "metric": "test cumulative return", "value": float(r.cum_return_mean)}
    except Exception:
        return None


if __name__ == "__main__":
    sys.exit(main())
