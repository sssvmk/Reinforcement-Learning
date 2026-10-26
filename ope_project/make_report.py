"""Build tables and figures from outputs/results/full_*.csv.  python make_report.py [full|quick]"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

mode = sys.argv[1] if len(sys.argv) > 1 else "full"   # "full", "quick", "full_inv", "quick_inv"
SFX = "_inv" if mode.endswith("_inv") else ""
R = Path("outputs/results")
FIG = Path("outputs/figures")
FIG.mkdir(parents=True, exist_ok=True)
pk = pd.read_csv(R / f"{mode}_rep_metrics.csv")
rows = pd.read_csv(R / f"{mode}_agent_rows.csv")
dg = pd.read_csv(R / f"{mode}_diagnostics.csv")
order = ["TD", "GTD2", "TDC", "ETD", "ETD_clip", "ETD_noreset", "LSTD", "FQE", "WIS", "OracleProj", "BehaviorValue", "Random"]
mets = ["nrmse_J", "spearman", "nregret1", "prec3", "pair_acc", "diverged_frac", "pick_test_cum"]


def table(df):
    g = df.groupby("method")[mets].agg(["median"]).droplevel(1, axis=1)
    return g.reindex([m for m in order if m in g.index])


scope = ("M5 inventory results: primary series FOODS_2_197_CA_4 plus two replication series; costs are assumptions."
         if SFX else "All trading results rest on ONE instrument (TECHM).")
out = ["# Results (auto-generated)\n", scope + " Medians over repetitions.\n"]
if SFX:
    out += ["## Replication series (base condition)\n"]
    for f in sorted(pk[pk.cond.str.startswith("replicate")].cond.unique()):
        out += [f"### {f}\n", table(pk[pk.cond == f]).round(4).to_markdown(), "\n"]
base = pk[(pk.cond == "base") & (pk.fold == "primary")]
out += ["## Primary fold, base condition\n", table(base).round(4).to_markdown(), "\n"]
for f in sorted(pk.fold.unique()):
    if f.startswith("wf"):
        out += [f"## Walk-forward fold {f}\n", table(pk[(pk.cond == "base") & (pk.fold == f)]).round(4).to_markdown(), "\n"]
abl = pk[pk.cond.str.startswith("ablation")]
for met in ("nrmse_J", "spearman", "nregret1"):
    p = abl.groupby(["cond", "method"])[met].median().unstack().reindex(columns=[m for m in order if m in abl.method.unique()])
    out += [f"## Ablations: median {met}\n", p.round(3).to_markdown(), "\n"]
st = {}
b0 = pk[(pk.cond == "base") & (pk.fold == "primary")]
from stats import paired_table, friedman
for met in ("nrmse_J", "spearman", "nregret1"):
    sc = {m: g.sort_values("rep")[met].to_numpy() for m, g in b0.groupby("method")}
    st[met] = dict(friedman=friedman(sc), pairs=paired_table(sc))
for met, s in st.items():
    out += [f"## Paired tests on {met} (primary, base)\n", f"Friedman: {s['friedman']}\n"]
    keep = [r for r in s["pairs"] if {r["a"], r["b"]} <= {"TD", "GTD2", "TDC", "ETD", "ETD_clip", "LSTD", "FQE", "WIS"}]
    out += [pd.DataFrame(keep).round(4).to_markdown(index=False), "\n"]
d0 = dg[(dg.cond == "base") & (dg.fold == "primary")]
out += ["## ETD emphasis diagnostics (primary, base; across repetitions)\n",
        d0.drop(columns=["cond", "fold", "rep"]).describe().round(3).to_markdown(), "\n"]
(R.parent / f"results_{mode}.md").write_text("\n".join(out), encoding="utf8")

# ---- figures
b = rows[(rows.cond == "base") & (rows.fold == "primary") & (rows.rep == 0)]
meths = ["TD", "GTD2", "TDC", "ETD_clip", "LSTD", "FQE", "WIS"]
fig, ax = plt.subplots(1, len(meths), figsize=(3 * len(meths), 3), sharex=True)
for a, m in zip(ax, meths):
    s = b[b.method == m]
    a.scatter(s.J_true, s.J_final, s=10)
    lim = [s.J_true.min(), s.J_true.max()]
    a.plot(lim, lim, "k--", lw=.7)
    a.set_title(m)
    a.set_xlabel("true J")
ax[0].set_ylabel("estimated J")
fig.suptitle("Estimated vs true J (rep 0; ETD plotted as ETD_clip; unclipped ETD is off-scale)")
fig.tight_layout(); fig.savefig(FIG / f"j_scatter{SFX}.png", dpi=110); plt.close(fig)

cv = np.load(R / f"{mode}_curves.npz")
fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))
for k, a in zip(("J", "msve", "mspbe"), ax):
    for m in ["TD", "GTD2", "TDC", "ETD", "ETD_clip"]:
        key = f"{m}|{k}"
        if key in cv:
            y = cv[key]
            a.plot(np.arange(1, len(y) + 1), y, label=m)
    a.set_title({"J": "J estimate (median over agents/reps)", "msve": "MSVE", "mspbe": "MSPBE"}[k])
    a.set_xlabel("checkpoint")
    if k != "J":
        a.set_yscale("log")
ax[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(FIG / f"learning_curves{SFX}.png", dpi=110); plt.close(fig)

for cond_set, name in ((["ablation:coverage_eps_b_0.1", "base", "ablation:coverage_eps_b_0.6"], "coverage"),
                       (["ablation:logsize_100", "base", "ablation:logsize_2000"], "logsize")):
    sub = pk[pk.cond.isin(cond_set) & (pk.fold == "primary")]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    for a, met in zip(ax, ("spearman", "nregret1")):
        p = sub.groupby(["cond", "method"])[met].median().unstack()
        p = p.reindex(cond_set)[[m for m in ["TD", "GTD2", "TDC", "ETD_clip", "LSTD", "FQE", "WIS"] if m in p.columns]]
        p.plot(ax=a, marker="o")
        a.set_xticks(range(3)); a.set_xticklabels([c.replace("ablation:", "") for c in cond_set], rotation=15, fontsize=7)
        a.set_title(met)
    fig.tight_layout(); fig.savefig(FIG / f"ranking_vs_{name}{SFX}.png", dpi=110); plt.close(fig)

tun = json.loads((R / f"{mode}_tuning.json").read_text())
k0 = next(iter(tun))
fig, ax = plt.subplots(1, 4, figsize=(13, 3), sharey=True)
for a, m in zip(ax, ("TD", "GTD2", "TDC", "ETD")):
    df = pd.DataFrame([{**p, "score": min(s, 10)} for p, s in tun[k0][m]])
    for key, g in df.groupby("kappa" if m in ("GTD2", "TDC") else "lam"):
        a.plot(g.alpha, g.score, marker="o", label=f"{'kappa' if m in ('GTD2','TDC') else 'lam'}={key}")
    a.set_xscale("log"); a.set_title(f"{m}: tuning-log objective / w=0 value (capped at 10)"); a.legend(fontsize=6)
fig.tight_layout(); fig.savefig(FIG / f"stepsize_sensitivity{SFX}.png", dpi=110); plt.close(fig)

fig, ax = plt.subplots(figsize=(5, 3.2))
for m in ["TD", "GTD2", "TDC", "ETD_clip", "LSTD", "FQE", "WIS"]:
    s = base[base.method == m]
    ax.scatter(s.pick_J_true, s.pick_test_cum, label=m, s=14)
ax.set_xlabel("true J of the agent each method picks (OPE period)"); ax.set_ylabel("test-period cum. return of the pick")
ax.legend(fontsize=6)
fig.tight_layout(); fig.savefig(FIG / f"picks_performance{SFX}.png", dpi=110); plt.close(fig)

fig, ax = plt.subplots(figsize=(5, 3.2))
dall = dg[dg.fold == "primary"]
ax.scatter(dall.gamma2_Erho2, dall.F_var, s=10)
ax.set_xlabel("gamma^2 E[rho^2] (>1 means unbounded emphasis variance)"); ax.set_ylabel("var of followon trace F")
ax.set_yscale("log"); ax.axvline(1, color="k", ls="--", lw=.7); ax.set_title("ETD emphasis variance (per rep/condition)")
fig.tight_layout(); fig.savefig(FIG / f"etd_emphasis_variance{SFX}.png", dpi=110); plt.close(fig)
print("wrote", R.parent / f"results_{mode}.md", "and figures in", FIG)
