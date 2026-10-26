"""Run the OPE experiments.  python run_experiment.py --quick | --full  [--yes]"""
from __future__ import annotations
import argparse, copy, hashlib, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from common import load_config, check_m5, make_folds
from data_trading import load_all
from domains import TradingDomain
import engine
from metrics import value_errors, ranking_metrics, trading_performance
from stats import paired_table, friedman

ROOT = Path(__file__).parent
OUT = ROOT / "outputs"
SEC_PER_REP = 22.0  # measured on 16 cores (300 windows, 20 epochs); used only for the printed estimate


def lock_plan():
    h = hashlib.sha256((ROOT / "analysis_plan.yaml").read_bytes()).hexdigest()
    f = OUT / "analysis_plan.sha256"
    OUT.mkdir(exist_ok=True)
    if f.exists():
        if f.read_text().split()[0] != h:
            sys.exit("analysis_plan.yaml changed after it was locked; refusing to run (delete the .sha256 to re-lock "
                     "and report the deviation).")
    else:
        f.write_text(f"{h}  analysis_plan.yaml  locked {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    return h


def conditions(cfg, mode):
    e = cfg["experiment"]
    base = dict(eps_b=0.3, eps_t=0.1, n_windows=e["ope_windows"], dp=0.0)
    ab = [("coverage_eps_b_0.1", dict(eps_b=0.1)), ("coverage_eps_b_0.6", dict(eps_b=0.6)),
          ("logsize_100", dict(n_windows=100)), ("logsize_2000", dict(n_windows=2000)),
          ("gamma_0.9", dict(gamma=0.9)), ("gamma_0.98", dict(gamma=0.98)), ("eps_t_0.3", dict(eps_t=0.3)),
          ("dense_features", dict(dense=True)),
          ("decay_schedule", dict(schedule=(cfg["estimators"]["decay_p"], cfg["estimators"]["decay_k0"]))),
          ("lambda_0", dict(lam=0.0)), ("lambda_0.9", dict(lam=0.9)), ("vform_12_30", dict(vform=1)),
          ("protocol_B", dict(protocol="B")), ("cost_20bps", dict(cost_bps=20.0)),
          ("kappa_0.1_stress", dict(kappa=cfg["estimators"]["kappa_stress"]))]
    if mode == "quick":
        q = e["quick"]
        base.update(n_windows=q["ope_windows"], epochs=q["epochs"])
        return [("base", "primary", base, q["reps"])] + [("ablation:" + n, "primary", {**base, **d}, 1) for n, d in ab[:1]]
    out = [("base", "primary", base, e["reps"])]
    out += [("base", f"wf{i}", base, e["reps"]) for i in (1, 2, 3)]
    out += [("ablation:" + n, "primary", {**base, **d}, 5) for n, d in ab]
    return out


def inv_conditions(cfg, mode):
    e = cfg["experiment"]
    base = dict(eps_b=0.3, eps_t=0.1, n_windows=e["ope_windows"], dp=0.0)
    ab = [("coverage_eps_b_0.1", dict(eps_b=0.1)), ("coverage_eps_b_0.6", dict(eps_b=0.6)),
          ("logsize_100", dict(n_windows=100)), ("logsize_2000", dict(n_windows=2000)),
          ("gamma_0.9", dict(gamma=0.9)), ("eps_t_0.3", dict(eps_t=0.3)),
          ("dense_features", dict(dense=True)), ("lambda_0", dict(lam=0.0)), ("lambda_0.9", dict(lam=0.9)),
          ("lead_time_1", dict(lead=1)), ("lead_time_5", dict(lead=5)),
          ("kappa_0.1_stress", dict(kappa=cfg["estimators"]["kappa_stress"]))]
    if mode == "quick":
        q = e["quick"]
        base.update(n_windows=q["ope_windows"], epochs=q["epochs"])
        return [("base", "primary", base, q["reps"])] + [("ablation:" + n, "primary", {**base, **d}, 1) for n, d in ab[:1]]
    out = [("base", "primary", base, e["reps"])]
    out += [("base", f"wf{i}", base, e["reps"]) for i in (1, 2)]
    out += [("replicate:" + s, s, base, e["reps"]) for s in ("intermittent", "seasonal")]
    out += [("ablation:" + n, "primary", {**base, **d}, 5) for n, d in ab]
    return out


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--quick", action="store_true")
    g.add_argument("--full", action="store_true")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation after the runtime estimate")
    ap.add_argument("--domain", choices=["trading", "inventory"], default="trading")
    a = ap.parse_args()
    mode = "quick" if a.quick else "full"
    invd = a.domain == "inventory"
    pre = f"{mode}_inv" if invd else mode
    cfg = load_config()
    seed0 = cfg["experiment"]["seed"]
    inv = check_m5(cfg)
    print("inventory domain:", inv)
    plan_hash = lock_plan()
    conds = inv_conditions(cfg, mode) if invd else conditions(cfg, mode)
    nrep = sum(c[3] for c in conds)
    est = nrep * SEC_PER_REP * (1.0 if mode == "full" else 0.5) + 60 * len({(c[1]) for c in conds})
    print(f"[{mode}/{a.domain}] {len(conds)} conditions, {nrep} repetitions, estimated runtime ~{est/60:.0f} min")
    if invd:
        import data_m5
        from domains_inventory import InvDomain
        series = data_m5.load_inventory_series(cfg, str(OUT))
        cfg_inv = copy.deepcopy(cfg)
        cfg_inv["splits"] = cfg["inventory"]["splits"]
        cfg_inv["walk_forward"] = cfg["inventory"]["walk_forward"]
        folds = make_folds(series["primary"].N, cfg_inv)
        cfg = cfg_inv
    else:
        data, _ = load_all(cfg)
        d = data[0]
        folds = make_folds(d.N, cfg)
    long_rows, picks, diag_rows, curve_acc = [], [], [], {}
    tuned_cache, tune_tables, pool_cache = {}, {}, {}
    t_start = time.time()
    for cname, fname, cond, reps in conds:
        c2 = copy.deepcopy(cfg)
        if "gamma" in cond:
            c2["mdp"]["gamma"] = cond["gamma"]
        if "cost_bps" in cond:
            c2["trading"]["cost_bps"] = cond["cost_bps"]
        if "epochs" in cond:
            c2["estimators"]["epochs"] = cond["epochs"]
        if invd:
            ser = series["primary" if fname.startswith(("primary", "wf")) else fname]
            fo = (folds[fname] if fname in folds else make_folds(ser.N, cfg)["primary"])
            dom = InvDomain(ser, fo, c2, lead_time=cond.get("lead"), tag=cname)
            ptag = f"{fname}_g{dom.gamma}_L{dom.L}"
            if ptag not in pool_cache:
                pool_cache[ptag] = dom.build_pool(1)
            pool = pool_cache[ptag]
        else:
            dom = TradingDomain(d, folds[fname], c2)
            ptag = f"g{dom.gamma}_c{dom.cost_bps}"
            pool = dom.build_pool(1, cache_dir=OUT / "pools" / ptag)
        key = (fname, ptag,  cond["eps_b"], cond["eps_t"], cond["n_windows"] if mode == "quick" else 0)
        if key not in tuned_cache:
            te = cfg["experiment"]["quick"]["tune_epochs"] if mode == "quick" else cfg["estimators"]["tune_epochs"]
            nw = 60 if mode == "quick" else 150
            tuned_cache[key], tune_tables[str(key)] = engine.tune_hparams(
                dom, pool, c2, nw, cond["eps_b"], cond["eps_t"], seed0 + 7, te)
        chosen = tuned_cache[key]
        cnd = dict(cond)
        cnd["schedule"] = cond.get("schedule", (0.0, cfg["estimators"]["decay_k0"]))
        for r in range(reps):
            t0 = time.time()
            res = engine.run_rep(dom, pool, chosen, cnd, c2, seed0 + 1000 * r + 17)
            df = pd.DataFrame(res["rows"])
            df["cond"], df["fold"], df["rep"] = cname, fname, r
            long_rows.append(df)
            Jt = res["Jtrue"]
            if invd:
                test_perf = [dom.test_perf(ag) for ag in pool]
            else:
                test_perf = [trading_performance(dom.daily_returns(ag, "test")) for ag in pool]
            for m, g in df.groupby("method"):
                g = g.set_index("agent").loc[[a.name for a in pool]]
                ok_div = float(g["diverged"].mean())
                est_J = g["J_final"].to_numpy(float)
                ve = value_errors(est_J[np.isfinite(est_J)], Jt[np.isfinite(est_J)]) if np.isfinite(est_J).sum() > 2 else {}
                rk = ranking_metrics(est_J, Jt)
                pick = int(np.nanargmax(np.where(np.isfinite(est_J), est_J, -np.inf))) if np.isfinite(est_J).any() else 0
                picks.append(dict(cond=cname, fold=fname, rep=r, method=m, diverged_frac=ok_div,
                                  nrmse_J=ve.get("nrmse", np.nan), rmse_J=ve.get("rmse", np.nan),
                                  bias_J=ve.get("signed", np.nan), spearman=rk["spearman"], kendall=rk["kendall"],
                                  nregret1=rk["nregret1"], regret1=rk["regret1"], prec3=rk["prec_k"],
                                  pair_acc=rk["pair_acc"], pick=pool[pick].name, pick_J_true=float(Jt[pick]),
                                  pick_test_cum=test_perf[pick].get("cum_return", np.nan),
                                  pick_test_sharpe=test_perf[pick].get("sharpe", np.nan),
                                  msve_median=float(g["msve_final"].median()) if "msve_final" in g else np.nan))
            for row in res["diag"]:
                diag_rows.append(dict(cond=cname, fold=fname, rep=r, **row))
            if cname == "base" and fname == "primary":
                for (an, m), cv in res["curves"].items():
                    for k, v in cv.items():
                        curve_acc.setdefault((m, k), []).append(v)
            print(f"  {cname:>26} {fname:>7} rep {r+1}/{reps}  {time.time()-t0:5.1f}s  "
                  f"elapsed {(time.time()-t_start)/60:.1f} min", flush=True)
    res_dir = OUT / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(long_rows).to_csv(res_dir / f"{pre}_agent_rows.csv", index=False)
    pk = pd.DataFrame(picks)
    pk.to_csv(res_dir / f"{pre}_rep_metrics.csv", index=False)
    pd.DataFrame(diag_rows).to_csv(res_dir / f"{pre}_diagnostics.csv", index=False)
    np.savez(res_dir / f"{pre}_curves.npz", **{f"{m}|{k}": np.nanmedian(np.stack(v), axis=(0, 1)) if np.stack(v).ndim > 2
                                               else np.nanmedian(np.stack(v), axis=0) for (m, k), v in curve_acc.items()})
    (res_dir / f"{pre}_tuning.json").write_text(json.dumps(
        {k: {m: [(p, s) for p, s in v] for m, v in t.items()} for k, t in tune_tables.items()}, indent=1, default=float))
    # primary-comparison statistics
    b = pk[(pk.cond == "base") & (pk.fold == "primary")]
    stats_out = {}
    for met in ("nrmse_J", "spearman", "nregret1"):
        sc = {m: g.sort_values("rep")[met].to_numpy() for m, g in b.groupby("method")}
        stats_out[met] = dict(friedman=friedman(sc), pairs=paired_table(sc))
    (res_dir / f"{pre}_stats.json").write_text(json.dumps(stats_out, indent=1, default=float))
    print(f"done in {(time.time()-t_start)/60:.1f} min; plan sha256 {plan_hash[:12]}; results in {res_dir}")


if __name__ == "__main__":
    main()

