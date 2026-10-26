"""Aggregates ./outputs/results into tables (summary.md, CSVs) and plots."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import ROOT
from ranking_metrics import metrics_for

EST = ["fqe", "self", "ce_ope", "wis", "dr", "behavior", "random"]
MET = ["spearman", "kendall", "regret@1_norm", "p@3", "top1_in_true_top3", "pairwise_acc", "mae", "bias"]


def _load(res):
    runs = []
    for mf in sorted(res.glob("*_meta.json")):
        base = mf.name[:-len("_meta.json")]
        meta = json.loads(mf.read_text())
        tag = base.rsplit("_", 1)[-1]
        runs.append({"base": base, "tag": tag, "meta": meta, "cand": pd.read_csv(res / f"{base}_candidates.csv"),
                     "metrics": pd.read_csv(res / f"{base}_metrics.csv"),
                     "fam": pd.read_csv(res / f"{base}_family_error.csv"),
                     "hist": json.loads((res / f"{base}_fqe_history.json").read_text()),
                     "merr": pd.read_csv(res / f"{base}_model_error.csv")})
    return runs


def _ms(x):
    x = pd.Series(x).dropna()
    if not len(x):
        return "n/a"
    return f"{x.mean():.3f} ± {x.std(ddof=1) if len(x) > 1 else 0:.3f}"


def _md(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def make_report(cfg):
    res = (Path(cfg["out_dir"]) if Path(cfg["out_dir"]).is_absolute() else ROOT / cfg["out_dir"]) / "results"
    plots = res.parent / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    runs = _load(res)
    if not runs:
        print("no results to report")
        return
    main = [r for r in runs if r["tag"] == "main" and r["meta"]["cond"] == "medium"]
    abl = [r for r in runs if r["tag"] != "main"]
    md = ["# Results summary (auto-generated)\n", f"main runs: {len(main)} (series x repetitions); ablation runs: {len(abl)}\n"]

    # 1. estimator x metric (mean ± std over runs)
    rows = []
    for e in EST:
        row = {"estimator": e}
        for m in MET:
            row[m] = _ms([r["metrics"].query("estimator==@e and scope=='all'")[m].iloc[0]
                          for r in main if m in r["metrics"] and len(r["metrics"].query("estimator==@e and scope=='all'"))])
        rows.append(row)
    t1 = pd.DataFrame(rows)
    t1.to_csv(res / "table_estimator_metrics.csv", index=False)
    md += ["## Estimator x metric (overall pool, mean ± std over series x repetitions)\n", _md(t1), ""]

    # per series
    per = []
    for r in main:
        for e in ["fqe", "self", "ce_ope", "wis", "dr"]:
            m = r["metrics"].query("estimator==@e and scope=='all'").iloc[0]
            per.append({"run": r["base"], "estimator": e, "spearman": round(m["spearman"], 3),
                        "regret@1_norm": round(m["regret@1_norm"], 3), "p@3": round(m["p@3"], 3)})
    md += ["## Per run (series x repetition)\n", _md(pd.DataFrame(per)), ""]

    # 2. family leaderboard + per-family estimator error
    cand = pd.concat([r["cand"].assign(run=r["base"]) for r in main])
    lead = cand.groupby("family").agg(n=("name", "count"), best_true=("truth", "max"), mean_true=("truth", "mean"),
                                      worst_true=("truth", "min")).round(2).reset_index()
    md += ["## Method-family leaderboard (true test-period value; best/mean/worst across candidates and runs)\n", _md(lead), ""]
    fam = pd.concat([r["fam"] for r in main]).groupby(["estimator", "family"]).agg(
        bias=("bias", "mean"), mae=("mae", "mean"), within_spearman=("within_spearman", "mean")).round(2).reset_index()
    md += ["## Value error and within-family Spearman per estimator and family (bias = estimate - truth)\n", _md(fam), ""]
    fair = []
    for e in ["fqe", "ce_ope", "self"]:
        within = fam[(fam.estimator == e)]["within_spearman"].mean()
        overall = np.nanmean([r["metrics"].query("estimator==@e and scope=='all'")["spearman"].iloc[0] for r in main])
        fair.append({"estimator": e, "mean within-family Spearman": round(within, 3), "across-family (overall) Spearman": round(overall, 3)})
    md += ["## Cross-family fairness check\n", _md(pd.DataFrame(fair)), ""]

    # per-family ranking quality of estimators (within scope)
    pf = []
    for e in ["fqe", "self", "ce_ope", "wis", "dr"]:
        for r in main:
            for _, m in r["metrics"][(r["metrics"].estimator == e) & (r["metrics"].scope != "all")].iterrows():
                pf.append({"estimator": e, "family": m["scope"], "spearman": m["spearman"], "regret@1_norm": m["regret@1_norm"]})
    if pf:
        pfd = pd.DataFrame(pf).groupby(["estimator", "family"]).mean().round(3).reset_index()
        md += ["## Within-family ranking quality (families with >= 4 candidates)\n", _md(pfd), ""]

    # 3. ablations
    if abl:
        a_rows = []
        for r in abl:
            for e in ["fqe", "ce_ope", "self", "wis", "dr"]:
                m = r["metrics"].query("estimator==@e and scope=='all'").iloc[0]
                a_rows.append({"ablation": r["tag"], "estimator": e, "spearman": round(m["spearman"], 3),
                               "regret@1_norm": round(m["regret@1_norm"], 3), "mae": round(m["mae"], 2),
                               "n_cand": int(m["n"])})
        ad = pd.DataFrame(a_rows)
        ad.to_csv(res / "table_ablations.csv", index=False)
        md += ["## Ablations (primary series, compact pool without MCTS, 1 FQE seed)\n", _md(ad), ""]

    # 4. with vs without bad anchors; hyperparameter views
    anchors = {"never_order", "order_max"}
    wa = []
    for r in main:
        c = r["cand"]
        c2 = c[~c["name"].isin(anchors)].reset_index(drop=True)
        for e in ["fqe", "ce_ope", "self", "dr"]:
            m1 = metrics_for(c[e].to_numpy(), c["truth"].to_numpy(), ci=False)
            m2 = metrics_for(c2[e].to_numpy(), c2["truth"].to_numpy(), ci=False)
            wa.append({"run": r["base"], "estimator": e, "spearman_all": round(m1["spearman"], 3),
                       "spearman_no_anchors": round(m2["spearman"], 3)})
    md += ["## Pool with vs. without clearly-bad anchors (never_order, order_max)\n", _md(pd.DataFrame(wa)), ""]
    hp = cand.copy()
    hp = hp[hp["name"].str.contains("dyna_|cql_|mcts_|dqn|ce_")][["run", "name", "truth", "truth_lo", "truth_hi", "fqe", "self", "ce_ope"]].round(2)
    hp.to_csv(res / "table_hyperparameter_views.csv", index=False)
    md += ["## Hyper-parameter sweeps (n_min / lambda, n_planning, CQL alpha, MCTS budget, DQN steps): full table in table_hyperparameter_views.csv\n"]
    d = cand[cand["family"] == "MCTS"][["name", "truth", "fqe", "fqe_distilled"]].round(2)
    if len(d):
        d["gap(FQE_sub - FQE_distilled)"] = (d["fqe"] - d["fqe_distilled"]).round(2)
        md += ["## MCTS: FQE on subsample vs. distilled policy\n", _md(d), ""]
    (res / "summary.md").write_text("\n".join(md), encoding="utf-8")

    # ---------------- plots
    r0 = main[0] if main else runs[0]
    fig, axs = plt.subplots(2, 3, figsize=(14, 8))
    for ax, e in zip(axs.ravel(), ["fqe", "self", "ce_ope", "wis", "dr", "random"]):
        c = r0["cand"]
        for fam_, g in c.groupby("family"):
            ax.scatter(g["truth"], g[e], label=fam_, s=18)
        lim = [min(c["truth"].min(), c[e].min()), max(c["truth"].max(), c[e].max())]
        ax.plot(lim, lim, "k--", lw=0.6)
        ax.set_title(f"{e} vs truth"); ax.set_xlabel("true value"); ax.set_ylabel("estimate")
    axs[0, 0].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(plots / "scatter_estimated_vs_true.png", dpi=110); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    top = r0["cand"].sort_values("truth", ascending=False)["name"].tolist()
    for nme in top[:3] + top[-2:]:
        h = r0["hist"][nme]
        ax.plot([x["round"] for x in h], [x["value"] for x in h], label=nme)
    ax.set_xlabel("FQE round"); ax.set_ylabel("estimated value"); ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(plots / "fqe_value_vs_round.png", dpi=110); plt.close(fig)

    if abl:
        fig, axs = plt.subplots(1, 2, figsize=(11, 4))
        sub = ad[ad.ablation.str.startswith(("cov_", "size_"))]
        for e in ["fqe", "ce_ope", "self", "dr"]:
            s_ = sub[sub.estimator == e]
            axs[0].plot(s_["ablation"], s_["spearman"], "o-", label=e)
            axs[1].plot(s_["ablation"], s_["regret@1_norm"], "o-", label=e)
        axs[0].set_ylabel("Spearman"); axs[1].set_ylabel("normalised regret@1")
        for a_ in axs:
            a_.tick_params(axis="x", rotation=45)
        axs[0].legend(); fig.tight_layout(); fig.savefig(plots / "ranking_vs_size_coverage.png", dpi=110); plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    cc = pd.concat([r["cand"] for r in main])
    err = (cc["fqe"] - cc["truth"]).abs()
    axs[0].scatter(cc["support"], err, s=14); axs[0].set_xlabel("logged-support fraction of greedy actions")
    axs[0].set_ylabel("|FQE - truth|")
    rho = cc[["support"]].assign(err=err).corr(method="spearman").iloc[0, 1]
    axs[0].set_title(f"FQE error vs coverage (Spearman {rho:.2f})")
    me = pd.concat([r["merr"] for r in main])
    v = me[me.visited]
    axs[1].scatter(np.log1p(v["n"]), v["tv"], s=6, alpha=0.4); axs[1].set_xlabel("log(1+n) of (s,a)"); axs[1].set_ylabel("TV(P_hat, P_true)")
    axs[1].set_title("CE model error vs. data")
    ce_err = (cc["ce_ope"] - cc["truth"]).abs()
    axs[2].scatter(cc["model_tv"], ce_err, s=14); axs[2].set_xlabel("model TV error at policy's (s,a)")
    axs[2].set_ylabel("|CE-OPE - truth|"); axs[2].set_title("Model error vs CE-OPE error")
    fig.tight_layout(); fig.savefig(plots / "coverage_and_model_error.png", dpi=110); plt.close(fig)
    print("report written to", res / "summary.md")
