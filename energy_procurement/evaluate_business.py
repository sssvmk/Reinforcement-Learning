"""Business evaluation on held-out days: savings vs buying the TSO forecast, risk, behaviour, forecast value, robustness.
  python evaluate_business.py --quick   # test-year table, plots, sensitivity on 60 days, no rolling origin (~3 min)
  python evaluate_business.py --full    # + quarterly rolling-origin retraining 2017Q1-2018Q4 and full sensitivity (~25-40 min)"""
import argparse
import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import data
from problem import Problem
from agents import forecast as F, scenarios, newsvendor as nv, cvar_sp, mean_variance as mv
from metrics import agent_metrics, block_bootstrap_ci
from risk import cvar

OUT = os.path.join(data.OUT, "eval", "business")
SCHED = os.path.join(data.OUT, "agents", "schedules.npz")

POLICIES = {  # label -> schedule key (split filled in)
    "TSO forecast": "{s}|benchmark|tso_forecast",
    "Bias-corrected forecast": "{s}|benchmark|bias_corrected",
    "Oracle (q = actual load)": "{s}|benchmark|oracle",
    "Buy nothing day-ahead": "{s}|benchmark|buy_nothing",
    "Buy at cap": "{s}|benchmark|buy_cap",
    "Newsvendor A (GBM)": "{s}|nvA|gbm",
    "Newsvendor B (GBM)": "{s}|nvB|gbm",
    "Newsvendor B (bias scenarios)": "{s}|nvB|bias",
    "CVaR SP (GBM, a=.95, lam=1)": "{s}|cvar|gbm|0.95|1.0",
    "CVaR SP (GBM, a=.95, lam=4)": "{s}|cvar|gbm|0.95|4.0",
    "Mean-variance (GBM, kappa=8)": "{s}|mv|gbm|8.0",
    "Mean-variance (GBM, kappa=32)": "{s}|mv|gbm|32.0",
}


def savings_table(P, Z, split, n_boot):
    idx = P.idx(split)
    ref = Z[POLICIES["TSO forecast"].format(s=split)]
    C_ref = P.cost(ref, idx)
    rows = []
    for lbl, k in POLICIES.items():
        key = k.format(s=split)
        if key not in Z.files:
            continue
        q = Z[key]
        C = P.cost(q, idx)
        pt, lo, hi = block_bootstrap_ci(C_ref - C, 7, n_boot)
        m = agent_metrics(P, q, idx, ref)
        rows.append({"policy": lbl, "mean_cost_eur_day": C.mean(), "savings_eur_day": pt, "ci_lo": lo, "ci_hi": hi,
                     "savings_pct": 100 * pt / C_ref.mean(), "annualized_eur": 365 * pt, "annualized_lo": 365 * lo,
                     "annualized_hi": 365 * hi, "std_eur": C.std(), "cvar95_eur": cvar(C, .95), "worst_day_eur": C.max(),
                     "max_monthly_eur": m["max_monthly_eur"], "hedge_ratio": m["hedge_ratio"], "shortfall_share": m["shortfall_share"],
                     "surplus_share": m["surplus_share"], "share_days_cheaper": m["share_days_cheaper"]})
    return pd.DataFrame(rows)


def pg_table(split, tso_cost):
    p = os.path.join(data.OUT, "pg", "summary.csv")
    if not os.path.exists(p):
        return pd.DataFrame()
    S = pd.read_csv(p)
    rows = []
    for (o, a), d in S.groupby(["objective", "algo"]):
        sav = tso_cost - d[f"best_val_{split}_mean_eur"]
        rows.append({"objective": o, "algo": a, "n_seeds": len(d), "savings_eur_day_median": sav.median(),
                     "savings_lo": sav.quantile(.025), "savings_hi": sav.quantile(.975), "savings_pct_median": 100 * sav.median() / tso_cost,
                     "annualized_median": 365 * sav.median(), "skill_median": d[f"best_val_{split}_skill"].median()})
    return pd.DataFrame(rows)


def behaviour(P, Z, split):
    idx = P.idx(split)
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    for lbl in ("TSO forecast", "Newsvendor B (GBM)", "CVaR SP (GBM, a=.95, lam=4)", "Mean-variance (GBM, kappa=32)"):
        q = Z[POLICIES[lbl].format(s=split)]
        ax[0].plot(range(24), (q / P.Lhat[idx]).mean(0), label=lbl)
        ax[1].plot(range(24), (q <= 1e-9).mean(0), label=lbl)
    ax[0].set_title("mean purchase / TSO forecast by hour"); ax[1].set_title("share of days with zero purchase by hour")
    ax[0].legend(fontsize=7); ax[0].set_xlabel("hour"); ax[1].set_xlabel("hour")
    months = P.dates[idx].month
    q = Z[POLICIES["Newsvendor B (GBM)"].format(s=split)]
    sav = P.cost(Z[POLICIES["TSO forecast"].format(s=split)], idx) - P.cost(q, idx)
    ax[2].bar(range(1, 13), [sav[months == m].sum() for m in range(1, 13)])
    ax[2].set_title("newsvendor B (GBM): monthly savings vs TSO (EUR)"); ax[2].set_xlabel("month")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f"behaviour_{split}.png"), dpi=100); plt.close(fig)


def savings_plot(T, split):
    T = T[~T.policy.isin(["Buy at cap"])]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    y = np.arange(len(T))
    ax.barh(y, T.annualized_eur / 1e3, xerr=[(T.annualized_eur - T.annualized_lo) / 1e3, (T.annualized_hi - T.annualized_eur) / 1e3], color="C0", alpha=.8)
    ax.set_yticks(y); ax.set_yticklabels(T.policy, fontsize=8); ax.invert_yaxis()
    ax.set_xlabel("annualised saving vs TSO-forecast purchasing (kEUR / year, 10 MW customer)")
    ax.set_title(f"{split} year, 95% block-bootstrap CI (7-day blocks)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f"savings_{split}.png"), dpi=100); plt.close(fig)


def rolling_origin(P, fc, n_jobs):
    """Refit the scenario model each quarter on all earlier days (bias scenarios) and evaluate the next quarter."""
    dates = P.dates
    qs = pd.period_range("2017Q1", "2018Q4", freq="Q")
    rows = []
    for q in qs:
        te = np.where(dates.to_period("Q") == q)[0]
        fit = np.where((dates < q.start_time) & (np.arange(P.D) >= 10))[0]
        sc = scenarios.Scenarios(P, "bias", fc, fit=fit)
        t0 = time.time()
        q_tso = P.Lhat[te]
        q_nv = nv.newsvendor_B(P, sc, te, S=300)
        Q, _ = cvar_sp.solve_days(P, sc, te, alpha=0.95, S=300, n_jobs=n_jobs)
        M = mv.mv_days(P, sc, te, q_nv, S=300)
        for lbl, qq in (("TSO forecast", q_tso), ("Newsvendor B (bias, rolling)", q_nv), ("CVaR SP lam=1 (bias, rolling)", Q[1.0]),
                        ("Mean-variance kappa=8 (bias, rolling)", M[8.0])):
            C, Cr = P.cost(qq, te), P.cost(q_tso, te)
            rows.append({"quarter": str(q), "policy": lbl, "n_days": len(te), "mean_eur": C.mean(), "savings_eur_day": (Cr - C).mean(),
                         "cvar95_eur": cvar(C, .95), "fit_days": len(fit)})
        print(f"rolling {q}: {time.time() - t0:.0f}s", flush=True)
    return pd.DataFrame(rows)


def sensitivity(split, n_days):
    """Savings of the bias-scenario newsvendor vs TSO under alternative market, cost and forecast-cutoff assumptions."""
    rows = []
    for name, kw, kind in (("base bias scenarios (rho=0, pi=0, cutoff=2)", {}, "bias"), ("rho=0.5", {"rho": 0.5}, "bias"),
                           ("rho=0.8", {"rho": 0.8}, "bias"), ("pi=10 EUR/MWh shortfall penalty", {"pi": 10.0}, "bias"),
                           ("GBM scenarios, cutoff=2 (base)", {}, "gbm"), ("GBM scenarios, cutoff=1", {"cutoff_days": 1}, "gbm"),
                           ("GBM scenarios, cutoff=3", {"cutoff_days": 3}, "gbm")):
        P = Problem(**kw)
        sc = scenarios.Scenarios(P, kind, F.make_forecasts(P) if kind == "gbm" else None)
        idx = P.idx(split)[:n_days]
        q = nv.newsvendor_B(P, sc, idx, S=300)
        Cr, C = P.cost(P.Lhat[idx], idx), P.cost(q, idx)
        pt, lo, hi = block_bootstrap_ci(Cr - C, 7, 1000)
        rows.append({"scenario": name, "savings_eur_day": pt, "lo": lo, "hi": hi, "savings_pct": 100 * pt / Cr.mean(), "n_days": len(idx)})
    for s in (0.5, 2.0):  # customer size scales cost linearly
        r0 = rows[0]
        rows.append({"scenario": f"customer size x{s} (linear scaling)", "savings_eur_day": r0["savings_eur_day"] * s,
                     "lo": r0["lo"] * s, "hi": r0["hi"] * s, "savings_pct": r0["savings_pct"], "n_days": r0["n_days"]})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--quick", action="store_true")
    g.add_argument("--full", action="store_true")
    ap.add_argument("--jobs", type=int, default=12)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    n_boot = 5000 if a.full else 500
    P = Problem()
    fc = F.make_forecasts(P)
    Z = np.load(SCHED)
    for split in ("test", "val"):
        T = savings_table(P, Z, split, n_boot)
        T.to_csv(os.path.join(OUT, f"policy_table_{split}.csv"), index=False)
        savings_plot(T, split); behaviour(P, Z, split)
        tso_cost = float(T[T.policy == "TSO forecast"].mean_cost_eur_day.iloc[0])
        pg = pg_table(split, tso_cost)
        if len(pg):
            pg.to_csv(os.path.join(OUT, f"pg_table_{split}.csv"), index=False)
        if split == "test":
            print(T[["policy", "savings_eur_day", "ci_lo", "ci_hi", "savings_pct", "annualized_eur", "cvar95_eur"]].round(1).to_string())
    T = pd.read_csv(os.path.join(OUT, "policy_table_test.csv")).set_index("policy")
    ladder = ["TSO forecast", "Bias-corrected forecast", "Newsvendor B (bias scenarios)", "Newsvendor B (GBM)", "Oracle (q = actual load)"]
    T.loc[ladder, ["savings_eur_day", "ci_lo", "ci_hi", "savings_pct", "annualized_eur"]].to_csv(os.path.join(OUT, "forecast_value_test.csv"))
    S = sensitivity("test", 365 if a.full else 60)
    S.to_csv(os.path.join(OUT, "sensitivity_test.csv"), index=False)
    print(S.round(2).to_string())
    if a.full:
        R = rolling_origin(P, fc, a.jobs)
        R.to_csv(os.path.join(OUT, "rolling_origin.csv"), index=False)
        piv = R.pivot(index="quarter", columns="policy", values="savings_eur_day")
        ax = piv.drop(columns="TSO forecast").plot(kind="bar", figsize=(10, 4.5), title="Rolling-origin savings vs TSO forecast (EUR/day)")
        ax.figure.tight_layout(); ax.figure.savefig(os.path.join(OUT, "rolling_origin.png"), dpi=100)
        print(piv.round(1).to_string())


if __name__ == "__main__":
    main()
