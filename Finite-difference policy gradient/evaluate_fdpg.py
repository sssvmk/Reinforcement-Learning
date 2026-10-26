"""Script 1: algorithm performance of the finite-difference policy-gradient methods -> ./outputs/algo_eval/

  python evaluate_fdpg.py --quick | --full
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import copy
import glob
import hashlib
import itertools
import json
import pickle
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

import run_fdpg as R
from audit_data import HERE, load_config
from fdpg.estimators import ESTIMATORS, central
from fdpg.optimizer import HParams, Scaler, project

OUT = os.path.join(HERE, "outputs", "algo_eval")
FD = list(ESTIMATORS)
TOL = 0.05


def md(df, floatfmt="{:.3f}"):
    cols = list(df.columns)
    rows = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        rows.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(rows)


def boot_ci(x, rng, n=2000):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    m = rng.choice(x, size=(n, len(x))).mean(axis=1)
    return tuple(np.percentile(m, [2.5, 97.5]))


def step(curve, grid, col):
    i = np.searchsorted(curve[:, 0], grid, side="right") - 1
    return curve[np.clip(i, 0, None), col]


def holm(p):
    p = np.asarray(p, float); m = len(p); order = np.argsort(p); adj = np.empty(m); run = 0.0
    for k, i in enumerate(order):
        run = max(run, (m - k) * p[i]); adj[i] = min(1.0, run)
    return adj


def paired(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.isfinite(d)]
    if len(d) < 2 or np.allclose(d, 0):
        return 1.0, 0.0, 0.0
    p = wilcoxon(d, zero_method="wilcox").pvalue
    ranks = pd.Series(np.abs(d[d != 0])).rank().to_numpy()
    rb = (ranks[d[d != 0] > 0].sum() - ranks[d[d != 0] < 0].sum()) / ranks.sum()
    return float(p), float(rb), float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else 0.0


def compare_family(table, methods, label):
    """table: dict method -> per-seed array (paired by index). Holm over all pairs."""
    rows = []
    for a, b in itertools.combinations(methods, 2):
        if a in table and b in table:
            n = min(len(table[a]), len(table[b]))
            p, rb, dz = paired(table[a][:n], table[b][:n])
            rows.append({"outcome": label, "A": a, "B": b, "mean_diff_A_minus_B": float(np.nanmean(np.asarray(table[a][:n]) - np.asarray(table[b][:n]))),
                         "p": p, "rank_biserial": rb, "cohens_dz": dz})
    if rows:
        adj = holm([r["p"] for r in rows])
        for r, q in zip(rows, adj):
            r["p_holm"], r["n_comparisons"] = float(q), len(rows)
    return rows


# ------------------------------------------------------------------ gradient quality
def reference_gradient(prob, u):
    Jref = prob.objective(split="train", fixed_stride=5)
    return central(Jref, u, np.random.default_rng(0), eps=0.02)


def draw_grads(prob, u, est, reps, N, eps, hp, crn=True, m=None, K=None, seed=0):
    J = prob.objective(N=N)
    return np.array([ESTIMATORS[est](J, u, np.random.default_rng(seed * 100003 + r), eps=eps, m=m or hp.m, K=K or hp.K or None, crn=crn)
                     for r in range(reps)])


def grad_metrics(G, gref):
    nr = np.linalg.norm(gref) + 1e-12
    cos = G @ gref / (np.linalg.norm(G, axis=1) * nr + 1e-12)
    rel = np.linalg.norm(G - gref, axis=1) / nr
    sign = (np.sign(G) == np.sign(gref)).mean(axis=1)
    bias = np.linalg.norm(G.mean(axis=0) - gref)
    return {"cosine": cos.mean(), "rel_l2": rel.mean(), "sign_agree": sign.mean(), "bias_l2": bias,
            "variance": float(G.var(axis=0, ddof=1).sum()), "cos_all": cos}


def gradient_quality(prob, recs, hp, reps, key, rng):
    c0 = sorted([r for r in recs if r["method"] == "central"], key=lambda r: r["seed"])[0]
    pts = {"start": c0["u0"], "mid": c0["traj"][len(c0["traj"]) // 2], "final": c0["u_final_internal"]}
    rows, cosines = [], {}
    for pn, u in pts.items():
        gref = reference_gradient(prob, u)
        for est in FD:
            G = draw_grads(prob, u, est, reps, prob.N, hp.eps, hp, seed=1)
            m = grad_metrics(G, gref)
            cosines[(pn, est)] = m.pop("cos_all")
            rows.append({"point": pn, "estimator": est, **m, "ref_norm": float(np.linalg.norm(gref))})
        if prob.policy.n == 1:   # compare with derivative of a dense J scan
            lo, hi = prob.policy.ilo[0], prob.policy.ihi[0]
            Jref = prob.objective(split="train", fixed_stride=5)
            g = np.linspace(lo, hi, 401)
            Js = np.array([Jref(np.array([x]), Jref.fixed_paths) for x in g])
            j = int(np.argmin(np.abs(g - u[0])))
            rows.append({"point": pn, "estimator": "dense_grid_derivative", "cosine": np.nan, "rel_l2": np.nan, "sign_agree": np.nan,
                         "bias_l2": float(abs(np.gradient(Js, g)[j] - gref[0])), "variance": 0.0, "ref_norm": float(abs(gref[0]))})
    return pd.DataFrame(rows), cosines, pts


def sensitivity(prob, recs, hp, reps, key, mode):
    u = sorted([r for r in recs if r["method"] == "central"], key=lambda r: r["seed"])[0]["u0"]
    gref = reference_gradient(prob, u)
    n = prob.policy.n
    rows = []
    for est in ("forward", "central", "spsa"):
        for eps in (0.01, 0.03, 0.1, 0.3, 1.0):
            m = grad_metrics(draw_grads(prob, u, est, reps, prob.N, eps, hp, seed=2), gref)
            rows.append({"param": "eps", "value": eps, "estimator": est, "bias_l2": m["bias_l2"], "variance": m["variance"], "cosine": m["cosine"]})
    for est in ("central", "spsa", "regression"):
        for N in (8, 16, 32, 64, 128):
            m = grad_metrics(draw_grads(prob, u, est, reps, N, hp.eps, hp, seed=3), gref)
            rows.append({"param": "N_paths", "value": N, "estimator": est, "bias_l2": m["bias_l2"], "variance": m["variance"], "cosine": m["cosine"]})
    for mm in (1, 2, 4, 8, 16):
        m = grad_metrics(draw_grads(prob, u, "spsa", reps, prob.N, hp.eps, hp, m=mm, seed=4), gref)
        rows.append({"param": "spsa_m", "value": mm, "estimator": "spsa", "bias_l2": m["bias_l2"], "variance": m["variance"], "cosine": m["cosine"]})
    for k in (n, 2 * n, 4 * n, 8 * n):
        m = grad_metrics(draw_grads(prob, u, "regression", reps, prob.N, hp.eps, hp, K=k, seed=5), gref)
        rows.append({"param": "regression_K", "value": k, "estimator": "regression", "bias_l2": m["bias_l2"], "variance": m["variance"], "cosine": m["cosine"]})
    # run-level sensitivity: step size and parameter scaling (central, final validation return)
    budget = R.MODES[mode]["budget"] // 2
    seeds = range(3 if mode == "quick" else 5)
    for mult in (0.25, 1.0, 4.0):
        v = [R.train_one(prob, "central", 900 + s, HParams(**{**hp.asdict(), "a": hp.a * mult}), budget)["val_final"] for s in seeds]
        rows.append({"param": "step_size_a_mult", "value": mult, "estimator": "central", "final_val_mean": float(np.mean(v)), "final_val_sd": float(np.std(v))})
    for c in (0.25, 1.0, 4.0):
        p2 = copy.copy(prob); pol = copy.copy(prob.policy); pol.scale = prob.policy.scale * c; p2.policy = pol
        v = [R.train_one(p2, "central", 900 + s, hp, budget)["val_final"] for s in seeds]
        rows.append({"param": "param_scale_mult", "value": c, "estimator": "central", "final_val_mean": float(np.mean(v)), "final_val_sd": float(np.std(v))})
    return pd.DataFrame(rows)


def crn_ablation(prob, recs, hp, reps, key):
    u = sorted([r for r in recs if r["method"] == "central"], key=lambda r: r["seed"])[0]["u0"]
    gref = reference_gradient(prob, u)
    rows = []
    for est in ("forward", "central", "spsa", "regression", "gaussian"):
        for crn in (True, False):
            m = grad_metrics(draw_grads(prob, u, est, reps, prob.N, hp.eps, hp, crn=crn, seed=6), gref)
            rows.append({"estimator": est, "crn": crn, "variance": m["variance"], "cosine": m["cosine"], "bias_l2": m["bias_l2"]})
    df = pd.DataFrame(rows)
    fin = []
    for est in ("central", "spsa"):
        for crn, name in ((True, est), (False, est + "_nocrn")):
            v = [r["val_final"] for r in recs if r["method"] == name]
            fin.append({"estimator": est, "crn": crn, "final_val_mean": float(np.mean(v)), "final_val_sd": float(np.std(v)), "n": len(v)})
    return df, pd.DataFrame(fin)


# ------------------------------------------------------------------ run-level analysis
def run_level(recs, budget, rng):
    methods = sorted({r["method"] for r in recs}, key=lambda m: (m not in FD, m))
    by = {m: sorted([r for r in recs if r["method"] == m], key=lambda r: r["seed"]) for m in methods}
    grid = np.unique(np.r_[0, np.geomspace(max(budget / 200, 50), budget, 60)])
    allv0 = np.mean([r["curve"][0, 1] for r in recs])
    best_mean = max(np.mean([r["val_final"] for r in by[m]]) for m in methods)
    rows, curves, stab = [], {}, []
    to95 = {}; final = {}
    for m in methods:
        V = np.array([step(r["curve"], grid, 1) for r in by[m]])
        D = np.array([step(r["curve"], grid, 2) for r in by[m]])
        curves[m] = (V, D)
        row = {"method": m, "n_seeds": len(by[m])}
        for f in (0.9, 0.95):
            tgt = allv0 + f * (best_mean - allv0)
            t = []
            for r in by[m]:
                hit = np.where(r["curve"][:, 1] >= tgt)[0]
                t.append(r["curve"][hit[0], 0] if len(hit) else 2.0 * budget)
            row[f"rollouts_to_{int(f*100)}pct_mean"] = float(np.mean(t))
            row[f"reached_{int(f*100)}pct_share"] = float(np.mean(np.array(t) < 2.0 * budget))
            if f == 0.95:
                to95[m] = np.array(t)
        for frac in (0.1, 0.25, 0.5, 1.0):
            row[f"val@{int(frac*100)}%budget"] = float(np.mean([step(r["curve"], [frac * budget], 1)[0] for r in by[m]]))
        fv = np.array([r["val_final"] for r in by[m]])
        final[m] = fv
        ci = boot_ci(fv, rng)
        row.update({"final_val_mean": fv.mean(), "final_val_sd": fv.std(ddof=1) if len(fv) > 1 else 0.0, "final_val_ci_lo": ci[0], "final_val_ci_hi": ci[1],
                    "final_train_mean": float(np.mean([r["train_final"] for r in by[m]])),
                    "runtime_per_iter_ms": float(np.mean([1000 * r["runtime"] / max(r["iters"], 1) for r in by[m]])),
                    "rollouts_mean": float(np.mean([r["rollouts"] for r in by[m]]))})
        U = np.array([r["u_final_internal"] for r in by[m]])
        row["final_param_sd_mean"] = float(U.std(axis=0).mean()) if len(U) > 1 else 0.0
        rows.append(row)
        cls = []
        for r in by[m]:
            c = r["curve"][:, 1]; k = max(2, len(c) // 4)
            if not np.isfinite(r["val_final"]) or r["val_final"] < c[0] - TOL: cls.append("diverge")
            elif r["val_final"] - c[0] < TOL: cls.append("stall")
            elif r["at_bound"]: cls.append("bound")
            elif c[-k:].max() - c[-k:].min() < TOL: cls.append("converged")
            else: cls.append("unconverged")
        stab.append({"method": m, **{k: float(np.mean([c == k for c in cls])) for k in ("converged", "unconverged", "bound", "stall", "diverge")}})
    return pd.DataFrame(rows), pd.DataFrame(stab), curves, grid, to95, final, by


def optimality_gap(prob, recs):
    pol = prob.policy
    if pol.n == 0 or pol.n > 3:
        return None
    Jv = prob.objective(split="val", fixed_stride=prob.vs)
    pts = {1: 201, 2: 41, 3: 11}[pol.n]
    axes = [np.linspace(l, h, pts) for l, h in zip(pol.ilo, pol.ihi)]
    best = max(Jv(np.array(p), Jv.fixed_paths) for p in itertools.product(*axes))
    best = max([best] + [r["val_final"] for r in recs])
    rows = []
    for m in sorted({r["method"] for r in recs}):
        g = np.array([best - r["val_final"] for r in recs if r["method"] == m])
        rows.append({"method": m, "gap_mean": g.mean(), "gap_sd": g.std(ddof=1) if len(g) > 1 else 0.0, "grid_best_val": best})
    return pd.DataFrame(rows)


def plots(key, curves, grid, stab, gq, sens, by, outdir):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for m, (V, D) in curves.items():
        for a, X in zip(ax, (V, D)):
            mu, sd = X.mean(axis=0), X.std(axis=0)
            a.plot(grid, mu, label=m, ls="--" if m not in FD else "-"); a.fill_between(grid, mu - sd, mu + sd, alpha=0.1)
    ax[0].set_title(f"{key}: validation J"); ax[1].set_title("train J (fixed diagnostic windows)")
    for a in ax: a.set_xscale("symlog", linthresh=100); a.set_xlabel("training rollouts")
    ax[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(outdir, f"learning_curves_{key}.png"), dpi=110); plt.close(fig)
    c = by["central"][0]
    if c["traj"] is not None:
        fig, a = plt.subplots(figsize=(6, 4)); a.plot(c["traj"]); a.set_xlabel("logged iteration"); a.set_ylabel("internal parameter")
        a.set_title(f"{key}: parameter trajectories (central, seed 0)"); fig.tight_layout(); fig.savefig(os.path.join(outdir, f"param_traj_{key}.png"), dpi=110); plt.close(fig)
    fig, a = plt.subplots(figsize=(6, 4)); s = gq[gq.point == "start"]; s = s[s.estimator.isin(FD)]
    a.bar(s.estimator, s.cosine); a.set_ylabel("cosine vs reference gradient"); a.set_title(f"{key}: gradient quality (start)")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"gradient_cosine_{key}.png"), dpi=110); plt.close(fig)
    e = sens[sens.param == "eps"]
    fig, a = plt.subplots(figsize=(6, 4))
    for est, g in e.groupby("estimator"):
        a.plot(g.value, g.bias_l2 ** 2 + g.variance, marker="o", label=est)
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("eps"); a.set_ylabel("bias^2 + variance (MSE)"); a.legend()
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"eps_bias_variance_{key}.png"), dpi=110); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--quick", action="store_true"); g.add_argument("--full", action="store_true")
    a = ap.parse_args(argv)
    mode = "quick" if a.quick else "full"
    plan = os.path.join(HERE, "analysis_plan.yaml")
    if mode == "full" and not os.path.exists(plan):
        print("analysis_plan.yaml missing: freeze it before a full run"); return 1
    plan_hash = hashlib.sha256(open(plan, "rb").read()).hexdigest()[:16]
    reps = 10 if mode == "quick" else 30
    od = os.path.join(OUT, mode); os.makedirs(od, exist_ok=True)
    cfg = load_config()
    rng = np.random.default_rng(0)
    files = sorted(glob.glob(os.path.join(HERE, "outputs", "runs", f"*__{mode}.pkl")))
    if not files:
        print("no runs found; run run_fdpg.py first"); return 1
    md_parts = [f"# Algorithm-performance summary ({mode})", f"analysis_plan.yaml sha256[:16] = {plan_hash}", ""]
    all_stats = []
    for f in files:
        key = os.path.basename(f).replace(f"__{mode}.pkl", "")
        domain, series, policy = key.split("__")
        d = pickle.load(open(f, "rb")); recs = d["records"]; hp = HParams(**d["hp"]); budget = d["budget"]
        n_scale = R.MODES[mode]["n_scale"]
        prob = R.get_problem(cfg, domain, series, policy, n_scale)
        print("evaluating", key, flush=True)
        summ, stab, curves, grid, to95, final, by = run_level(recs, budget, rng)
        gq, cosines, pts = gradient_quality(prob, recs, hp, reps, key, rng)
        sens = sensitivity(prob, recs, hp, reps, key, mode)
        crn_g, crn_f = crn_ablation(prob, recs, hp, reps, key)
        gap = optimality_gap(prob, recs)
        for name, df in (("summary", summ), ("stability", stab), ("gradient_quality", gq), ("sensitivity", sens), ("crn_gradient", crn_g), ("crn_final", crn_f)):
            df.to_csv(os.path.join(od, f"{name}_{key}.csv"), index=False)
        if gap is not None:
            gap.to_csv(os.path.join(od, f"optimality_gap_{key}.csv"), index=False)
        pd.concat([pd.DataFrame({"method": m, "rollouts": grid, "val_mean": V.mean(axis=0), "val_sd": V.std(axis=0),
                                 "train_mean": D.mean(axis=0), "train_sd": D.std(axis=0)}) for m, (V, D) in curves.items()]).to_csv(os.path.join(od, f"curves_{key}.csv"), index=False)
        cmp_methods = [m for m in FD + ["random", "neldermead"] if m in final]
        rows = compare_family(final, cmp_methods, "final_validation_return") + compare_family(to95, cmp_methods, "rollouts_to_95pct") \
            + compare_family({e: cosines[("start", e)] for e in FD}, FD, "gradient_cosine_start")
        stats = pd.DataFrame(rows); stats.to_csv(os.path.join(od, f"paired_tests_{key}.csv"), index=False)
        plots(key, curves, grid, stab, gq, sens, by, od)
        md_parts += [f"## {key}", f"tuned hyperparameters (validation only): eps={hp.eps}, a={hp.a}; budget={budget} rollouts; seeds={len(by['central'])}", "",
                     "### Final validation return, sample efficiency, runtime", md(summ[["method", "final_val_mean", "final_val_sd", "final_val_ci_lo", "final_val_ci_hi",
                      "rollouts_to_90pct_mean", "rollouts_to_95pct_mean", "reached_95pct_share", "val@25%budget", "val@100%budget", "runtime_per_iter_ms"]]), "",
                     "### Stability (share of seeds)", md(stab), "", "### Gradient quality vs reference (rows: point x estimator)", md(gq[["point", "estimator", "cosine", "rel_l2", "sign_agree", "bias_l2", "variance"]]), "",
                     "### Common random numbers", md(crn_g), "", md(crn_f), ""]
        if gap is not None:
            md_parts += ["### Optimality gap vs exhaustive grid on validation paths", md(gap), ""]
        sig = stats[stats.p_holm < 0.05] if len(stats) else stats
        md_parts += [f"### Paired Wilcoxon tests (Holm; {len(stats)} comparisons in total, {len(sig)} significant at 0.05)", md(sig[["outcome", "A", "B", "mean_diff_A_minus_B", "p_holm", "rank_biserial", "cohens_dz"]]) if len(sig) else "none significant", ""]
        md_parts += ["### Sensitivity (see CSV; plots saved)", md(sens.head(60).fillna("")), ""]
    open(os.path.join(od, "SUMMARY.md"), "w").write("\n".join(md_parts))
    print("wrote", os.path.join(od, "SUMMARY.md"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
