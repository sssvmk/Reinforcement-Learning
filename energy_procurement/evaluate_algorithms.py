"""Compare the policy-gradient algorithms: learning curves, gap to reference, sample efficiency, stability, sensitivity,
paired tests. Reads outputs/pg (cold start) and outputs/pg_warm (warm start) produced by run_pg.py.
  python evaluate_algorithms.py --quick   # first 3 seeds, 500 bootstrap resamples (~1 min)
  python evaluate_algorithms.py --full    # all seeds, 10,000 resamples (~3 min)"""
import argparse
import itertools
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
import data
from metrics import holm
from pg.training_plots import plot_compare
from run_pg import summarize

ROOT = data.OUT
OUT = os.path.join(ROOT, "eval", "algorithms")


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(((a[:, None] > b[None]).sum() - (a[:, None] < b[None]).sum()) / (len(a) * len(b)))


def boot_ci(x, n_boot, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = np.array([np.median(rng.choice(x, len(x))) for _ in range(n_boot)])
    return np.median(x), np.quantile(m, .025), np.quantile(m, .975)


def load_curves(root, obj, algo, max_seeds):
    runs = []
    d = os.path.join(root, obj, algo)
    for s in sorted(os.listdir(d))[:max_seeds]:
        p = os.path.join(d, s, "log.csv")
        if os.path.exists(p):
            L = pd.read_csv(p).dropna(subset=["val_skill"])
            runs.append((L.train_days.to_numpy(), L.val_skill.to_numpy()))
    return runs


def evaluate(root, label, max_seeds, n_boot):
    S = summarize(root)
    S = S.sort_values(["objective", "algo", "seed"])
    S = S[S.seed < max_seeds]
    out = os.path.join(OUT, label)
    os.makedirs(out, exist_ok=True)
    rows = []
    for (o, a), d in S.groupby(["objective", "algo"]):
        r = {"objective": o, "algo": a, "n_seeds": len(d)}
        for k in ("best_val_val_skill", "best_val_test_skill", "final_val_skill", "best_val_test_gap_rel", "best_val_test_mean_eur"):
            m, lo, hi = boot_ci(d[k], n_boot)
            r.update({f"{k}_median": m, f"{k}_lo": lo, f"{k}_hi": hi})
        r["test_skill_iqr"] = float(np.subtract(*np.percentile(d.best_val_test_skill, [75, 25])))
        r["reached_95"] = int(d.days_to_95.notna().sum()); r["reached_99"] = int(d.days_to_99.notna().sum())
        r["days_to_95_median"] = d.days_to_95.median(); r["days_to_99_median"] = d.days_to_99.median()
        r["worst_drop_median"] = d.worst_drop.median(); r["diverged"] = int(d.diverged.sum())
        r["sigma_collapsed"] = int(d.sigma_collapsed.sum()); r["wall_s_median"] = d.wall_s.median()
        rows.append(r)
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(out, "algorithm_summary.csv"), index=False)

    # paired Wilcoxon (same seeds) on test skill, Holm within objective
    pr = []
    for o, d in S.groupby("objective"):
        piv = d.pivot(index="seed", columns="algo", values="best_val_test_skill")
        res = []
        for a, b in itertools.combinations(piv.columns, 2):
            x, y = piv[a].dropna(), piv[b].dropna()
            ii = x.index.intersection(y.index)
            if len(ii) < 4 or np.allclose(x[ii], y[ii]):
                p = np.nan
            else:
                p = wilcoxon(x[ii], y[ii]).pvalue
            res.append({"objective": o, "a": a, "b": b, "n": len(ii), "median_diff": float(np.median(x[ii] - y[ii])) if len(ii) else np.nan,
                        "cliffs_delta": cliffs_delta(x[ii], y[ii]) if len(ii) else np.nan, "p": p})
        R = pd.DataFrame(res)
        if not len(R):
            continue
        ok = R.p.notna()
        R.loc[ok, "p_holm"] = holm(R.p[ok].to_numpy())
        pr.append(R)
    if pr:
        pd.concat(pr).to_csv(os.path.join(out, "paired_tests.csv"), index=False)

    # learning curves per objective
    for o in S.objective.unique():
        curves = {a: load_curves(root, o, a, max_seeds) for a in sorted(S[S.objective == o].algo.unique())}
        plot_compare({k: v for k, v in curves.items() if v}, os.path.join(out, f"learning_curves_{o}.png"),
                     f"{label}: validation skill vs rollouts ({o})")
    # test skill distribution
    fig, ax = plt.subplots(1, S.objective.nunique(), figsize=(5 * S.objective.nunique(), 4.5), squeeze=False)
    for j, (o, d) in enumerate(S.groupby("objective")):
        algs = sorted(d.algo.unique())
        ax[0, j].boxplot([d[d.algo == a].best_val_test_skill for a in algs], labels=algs)
        ax[0, j].axhline(1, ls=":", c="gray"); ax[0, j].axhline(0, ls=":", c="r")
        ax[0, j].set_title(f"{o}: test skill (1 = classical reference, 0 = TSO forecast)"); ax[0, j].tick_params(axis="x", rotation=45)
    fig.tight_layout(); fig.savefig(os.path.join(out, "test_skill_boxplots.png"), dpi=100); plt.close(fig)
    # skill at fixed budgets (sample efficiency)
    eff = []
    for (o, a), d in S.groupby(["objective", "algo"]):
        runs = load_curves(root, o, a, max_seeds)
        for B in (1000, 3000, 10000, 30000):
            v = [np.interp(B, r[0], r[1]) for r in runs if r[0].max() >= B * 0.99]
            if v:
                eff.append({"objective": o, "algo": a, "rollouts": B, "median_val_skill": float(np.median(v)), "n": len(v)})
    pd.DataFrame(eff).to_csv(os.path.join(out, "skill_at_budget.csv"), index=False)
    return T, S


def sensitivity():
    rows = []
    for tag, root in (("cold", os.path.join(ROOT, "pg")), ("warm", os.path.join(ROOT, "pg_warm"))):
        p = os.path.join(root, "grid.csv")
        if os.path.exists(p):
            g = pd.read_csv(p)
            g["start"] = tag
            rows.append(g)
    if not rows:
        return
    G = pd.concat(rows)
    G.to_csv(os.path.join(OUT, "lr_sensitivity.csv"), index=False)
    algs = sorted(G.algo.unique())
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    for j, st in enumerate(("cold", "warm")):
        for a in algs:
            d = G[(G.start == st) & (G.algo == a)].groupby("lr").val_skill.mean()
            ax[j].plot(d.index, d.values, "o-", label=a)
        ax[j].set_xscale("log"); ax[j].set_yscale("symlog", linthresh=0.5); ax[j].set_title(f"{st} start: validation skill vs learning rate")
        ax[j].set_xlabel("learning rate (TRPO: KL delta)"); ax[j].grid(alpha=.3)
    ax[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "lr_sensitivity.png"), dpi=100); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--quick", action="store_true")
    g.add_argument("--full", action="store_true")
    a = ap.parse_args()
    max_seeds, n_boot = (10, 10000) if a.full else (3, 500)
    os.makedirs(OUT, exist_ok=True)
    for label, root in (("cold", os.path.join(ROOT, "pg")), ("warm", os.path.join(ROOT, "pg_warm")),
                        ("v2_tuned_cond", os.path.join(ROOT, "pg_v2"))):
        if os.path.isdir(root) and any(os.path.isdir(os.path.join(root, o)) and         o != "grid" for o in os.listdir(root)):
            T, S = evaluate(root, label, max_seeds, n_boot)
            print(label); print(T[["objective", "algo", "n_seeds", "best_val_test_skill_median", "best_val_test_skill_lo",
                                   "best_val_test_skill_hi", "reached_95", "diverged"]].round(3).to_string())
    sensitivity()


if __name__ == "__main__":
    main()
