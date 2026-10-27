"""Extensions: (1) neural (PyTorch) vs GBM quantile forecasters in the procurement agents, static and with quarterly
rolling retraining of the forecaster itself.

  python run_extensions.py --static            # val/test savings with GBM and NN scenario models
  python run_extensions.py --rolling           # quarterly rolling-origin retraining of GBM and NN forecasters
  python run_extensions.py --quick             # smoke run
"""
import argparse
import os
import time
import numpy as np
import pandas as pd
import data
from problem import Problem
from agents import forecast as F, forecast_nn as N, scenarios, newsvendor as nv, cvar_sp, mean_variance as mv
from metrics import block_bootstrap_ci
from risk import cvar

OUT = os.path.join(data.OUT, "eval", "extensions")
os.makedirs(OUT, exist_ok=True)


def policies(P, sc, idx, S, n_jobs):
    q_nv = nv.newsvendor_B(P, sc, idx, S=S)
    Q, _ = cvar_sp.solve_days(P, sc, idx, alpha=0.95, S=S, n_jobs=n_jobs)
    M = mv.mv_days(P, sc, idx, q_nv, S=S)
    return {"Newsvendor B": q_nv, "CVaR SP (a=.95, lam=1)": Q[1.0], "Mean-variance (kappa=8)": M[8.0]}


def row(P, label, pol, q, idx, extra, n_boot=2000):
    Cr, C = P.cost(P.Lhat[idx], idx), P.cost(q, idx)
    pt, lo, hi = block_bootstrap_ci(Cr - C, 7, n_boot)
    return {**extra, "forecaster": label, "policy": pol, "n_days": len(idx), "savings_eur_day": pt, "ci_lo": lo, "ci_hi": hi,
            "savings_pct": 100 * pt / Cr.mean(), "annualized_eur": 365 * pt, "cvar95_eur": cvar(C, .95), "cvar95_tso_eur": cvar(Cr, .95)}


def static(P, S, n_jobs, n_days=None):
    bundles = {"GBM": F.make_forecasts(P), "Neural MLP": N.make_forecasts(P)}
    rows, fq = [], []
    for split in ("val", "test"):
        idx = P.idx(split)[:n_days]
        for lab, fc in bundles.items():
            sc = scenarios.Scenarios(P, "gbm", fc)
            for pol, q in policies(P, sc, idx, S, n_jobs).items():
                rows.append(row(P, lab, pol, q, idx, {"split": split}))
            m = {nm: F.metrics(tg[idx], fc[nm + "_q"][idx]) for nm, tg in (("eL", P.eL), ("g", P.g))}
            fq.append({"split": split, "forecaster": lab, **{f"{nm}_{k}": m[nm][k] for nm in m for k in ("pinball_mean", "mae_median", "cov80", "cov90")}})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "static_savings.csv"), index=False)
    pd.DataFrame(fq).to_csv(os.path.join(OUT, "forecast_quality.csv"), index=False)
    print(pd.DataFrame(rows)[["split", "forecaster", "policy", "savings_eur_day", "ci_lo", "ci_hi", "savings_pct", "cvar95_eur"]].round(1).to_string())
    print(pd.DataFrame(fq).round(4).to_string())


def rolling(P, S, n_jobs, quarters):
    base = {"GBM": F.make_forecasts(P), "Neural MLP": N.make_forecasts(P)}
    fit_fn = {"GBM": lambda *a: F.fit_predict(*a, max_iter=80), "Neural MLP": N.fit_predict}
    dates = P.dates
    rows = []
    for q in quarters:
        te = np.where(dates.to_period("Q") == q)[0]
        fit = np.where((dates < q.start_time) & (np.arange(P.D) >= 10))[0]
        Cr = P.cost(P.Lhat[te], te)
        for lab in base:
            t0 = time.time()
            fc = {k: v.copy() for k, v in base[lab].items() if k != "time_s"}
            for nm, tg in (("eL", P.eL), ("g", P.g)):
                qq, mu = fit_fn[lab](P, tg, fit, te)
                fc[nm + "_q"][te], fc[nm + "_mean"][te] = qq, mu
            sc = scenarios.Scenarios(P, "gbm", fc, fit=fit)
            for pol, qty in policies(P, sc, te, S, n_jobs).items():
                C = P.cost(qty, te)
                rows.append({"quarter": str(q), "forecaster": lab, "policy": pol, "n_days": len(te), "mean_eur": C.mean(),
                             "savings_eur_day": (Cr - C).mean(), "cvar95_eur": cvar(C, .95), "fit_days": len(fit)})
            print(f"rolling {q} {lab}: {time.time() - t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "rolling_forecaster.csv"), index=False)
    R = pd.DataFrame(rows)
    piv = R.pivot_table(index="quarter", columns=["forecaster", "policy"], values="savings_eur_day")
    print(piv.round(1).to_string())
    print(R.groupby(["forecaster", "policy"]).savings_eur_day.agg(["mean", "min", "max"]).round(1).to_string())
    import matplotlib
    matplotlib.use("Agg")
    ax = piv.plot(kind="bar", figsize=(12, 5), title="Rolling-retrained forecaster: savings vs TSO (EUR/day)")
    ax.legend(fontsize=6)
    ax.figure.tight_layout(); ax.figure.savefig(os.path.join(OUT, "rolling_forecaster.png"), dpi=100)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--static", action="store_true")
    ap.add_argument("--rolling", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    P = Problem()
    S = 100 if a.quick else 300
    if a.static or a.quick:
        static(P, S, a.jobs, 30 if a.quick else None)
    if a.rolling or a.quick:
        qs = pd.period_range("2018Q3", "2018Q3" if a.quick else "2018Q4", freq="Q") if a.quick else pd.period_range("2017Q1", "2018Q4", freq="Q")
        rolling(P, S, a.jobs, qs)
