"""Run the four classical agents + benchmarks on validation and test; write tables, schedules, frontiers."""
import argparse
import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import data
from problem import Problem
from agents import forecast as F, scenarios, newsvendor as nv, cvar_sp, mean_variance as mv, benchmarks as bm
from metrics import agent_metrics
from risk import cvar

OUTA = os.path.join(data.OUT, "agents")


def add(rows, P, name, kind, param, split, q, idx, q_ref, extra=None):
    m = agent_metrics(P, q, idx, q_ref)
    rows.append({"agent": name, "kind": kind, "param": param, "split": split, **m, **(extra or {})})


def run(quick=False, n_jobs=14, rho=None, tag=""):
    os.makedirs(OUTA, exist_ok=True)
    P = Problem(rho=rho)
    fc = F.make_forecasts(P)
    splits = {"val": P.idx("val"), "test": P.idx("test")}
    S = 100 if quick else 300
    if quick:
        splits = {k: v[:30] for k, v in splits.items()}
    rows, sched, times = [], {}, {}
    scs = {k: scenarios.Scenarios(P, k, fc) for k in ("tso", "bias", "gbm")}

    # decision-focused calibration of tau offsets (finite-difference PG on train days, cross-fit forecasts)
    th, hist = nv.calibrate_tau_shift(P, scs["gbm"], P.idx("train"), steps=15 if quick else 40)
    np.save(os.path.join(OUTA, f"tau_shift{tag}.npy"), th)

    for split, idx in splits.items():
        ref = P.Lhat[idx]
        for n, q in bm.benchmarks(P, idx).items():
            add(rows, P, "benchmark", n, "", split, q, idx, ref); sched[f"{split}|benchmark|{n}"] = q
        for kind, sc in scs.items():
            qA, tau = nv.newsvendor_A(P, sc, idx)
            add(rows, P, "newsvendor_A", kind, "", split, qA, idx, ref, {"tau_mean": tau.mean()})
            sched[f"{split}|nvA|{kind}"] = qA
            t0 = time.time(); qB = nv.newsvendor_B(P, sc, idx, S=None); times[f"nvB_{kind}"] = (time.time() - t0) / len(idx)
            add(rows, P, "newsvendor_B", kind, "", split, qB, idx, ref); sched[f"{split}|nvB|{kind}"] = qB
        qAc, _ = nv.newsvendor_A(P, scs["gbm"], idx, th)
        add(rows, P, "newsvendor_A", "gbm_calibrated", "", split, qAc, idx, ref); sched[f"{split}|nvA|gbm_calibrated"] = qAc

        # CVaR programs
        for kind in ("gbm", "bias"):
            for alpha in ((0.9, 0.95) if kind == "gbm" else (0.95,)):
                Q, t = cvar_sp.solve_days(P, scs[kind], idx, alpha=alpha, S=S, n_jobs=n_jobs)
                for lam, q in Q.items():
                    add(rows, P, "cvar_sp", kind, f"alpha={alpha};lambda={lam}", split, q, idx, ref,
                        {"alpha": alpha, "lambda": lam, "solve_s": t[lam]})
                    sched[f"{split}|cvar|{kind}|{alpha}|{lam}"] = q
        # mean-variance
        for kind in ("gbm", "bias"):
            q0 = nv.newsvendor_B(P, scs[kind], idx, S=S)
            t0 = time.time()
            M = mv.mv_days(P, scs[kind], idx, q0, S=S)
            times[f"mv_{kind}"] = (time.time() - t0) / len(idx)
            for k, q in M.items():
                add(rows, P, "mean_variance", kind, f"kappa={k}", split, q, idx, ref, {"kappa": k})
                sched[f"{split}|mv|{kind}|{k}"] = q
        for H, m_, s_, c_ in mv.hedge_ratio_frontier(P, idx):
            q = np.clip(H * P.Lhat[idx], 0, P.cap[idx])
            add(rows, P, "hedge_ratio", "H", f"H={H:.3f}", split, q, idx, ref, {"H": H})
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(OUTA, f"agent_results{tag}.csv"), index=False)
    np.savez(os.path.join(OUTA, f"schedules{tag}.npz"), **sched)
    json.dump(times, open(os.path.join(OUTA, f"timing{tag}.json"), "w"))
    plot_frontiers(tab, OUTA, tag)
    closed_form_check(P, scs["bias"])
    return tab


def plot_frontiers(tab, outdir, tag=""):
    fg, ax = plt.subplots(2, 2, figsize=(13, 9))
    for j, split in enumerate(("val", "test")):
        t = tab[tab.split == split]
        for kind, c in (("gbm", "C0"), ("bias", "C1")):
            for alpha, ls in ((0.95, "o-"), (0.9, "s--")):
                s = t[(t.agent == "cvar_sp") & (t.kind == kind) & (t.alpha == alpha)].sort_values("lambda")
                if len(s):
                    ax[0, j].plot(s.cvar95, s.mean_norm, ls, color=c, label=f"CVaR SP {kind} a={alpha}")
            s = t[(t.agent == "mean_variance") & (t.kind == kind)].sort_values("kappa")
            ax[1, j].plot(s.std_norm, s.mean_norm, "o-", color=c, label=f"MV {kind}")
        h = t[t.agent == "hedge_ratio"].sort_values("H")
        ax[1, j].plot(h.std_norm, h.mean_norm, "k.-", alpha=.5, label="hedge ratio H")
        for n, m in t[t.agent == "benchmark"].set_index("kind").iterrows() if False else t[t.agent == "benchmark"].set_index("kind").iterrows():
            if n in ("tso_forecast", "bias_corrected", "oracle"):
                ax[0, j].scatter(m.cvar95, m.mean_norm, marker="x", s=60, label=n)
                ax[1, j].scatter(m.std_norm, m.mean_norm, marker="x", s=60, label=n)
        ax[0, j].set_xlabel("CVaR95 of normalized daily cost"); ax[0, j].set_ylabel("mean normalized cost")
        ax[1, j].set_xlabel("std of normalized daily cost"); ax[1, j].set_ylabel("mean normalized cost")
        ax[0, j].set_title(f"Risk-return frontier ({split})"); ax[1, j].set_title(f"Mean-variance frontier ({split})")
        ax[0, j].legend(fontsize=7); ax[1, j].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, f"frontiers{tag}.png"), dpi=110); plt.close()


def closed_form_check(P, sc, n_days=40):
    """rho=1 linear instance: numerical (unbounded) vs closed-form minimum-variance / mean-variance hedge."""
    from scipy.optimize import minimize
    rows = []
    for d in P.idx("val")[:n_days]:
        Ls, pAs = sc.draw(d)
        for kappa in (1.0, 4.0, 16.0):
            qc = mv.closed_form_linear(Ls, pAs, P.pDA[d], kappa)
            f = lambda q: (lambda C: (C.mean() + kappa * C.var()))((pAs * Ls).sum(1) - ((pAs - P.pDA[d]) * q).sum(1))
            r = minimize(f, P.Lhat[d], method="BFGS")
            rows.append({"day": str(P.dates[d].date()), "kappa": kappa, "obj_closed": f(qc), "obj_numeric": r.fun,
                         "max_abs_diff_q": np.abs(qc - r.x).max()})
    pd.DataFrame(rows).to_csv(os.path.join(OUTA, "closed_form_vs_numeric.csv"), index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--rho", type=float, default=None)
    ap.add_argument("--jobs", type=int, default=14)
    a = ap.parse_args()
    t0 = time.time()
    tab = run(a.quick, a.jobs, a.rho, tag="_quick" if a.quick else "")
    pd.set_option("display.width", 220, "display.max_rows", 400)
    print(tab[tab.split == "test"][["agent", "kind", "param", "mean_norm", "std_norm", "cvar95", "savings_pct", "hedge_ratio"]].round(4).to_string())
    print("elapsed", time.time() - t0)
