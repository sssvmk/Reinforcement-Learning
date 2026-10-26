"""Train REINFORCE (5 baselines) and the equal-budget comparison methods; tune hyper-parameters on validation only.

python run_reinforce.py --domain inventory|trading|both --quick|--full [--sensitivity]
Outputs (./outputs only): models/, runs/, hparams, sensitivity/.
"""
import argparse
import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "1")
from common import load_config, outdir
import policies as P
import setup_problems as sp
from reinforce.algorithm import reinforce
from reinforce import comparison_methods as cm

MODES = {
    "quick": dict(episodes=6000, seeds=3, tune_seeds=1, tune_frac=0.5, grid=dict(lr=[0.03, 0.1], B=[32], ent=[0.0, 0.01]), sens_seeds=2),
    "full": dict(episodes=40000, seeds=10, tune_seeds=2, tune_frac=0.5, grid=dict(lr=[0.01, 0.03, 0.1], B=[16, 64], ent=[0.0, 0.001, 0.01]), sens_seeds=5),
}
MODES["neural_quick"] = dict(MODES["quick"], seeds=2, grid=dict(lr=[0.03], B=[32], ent=[0.0, 0.01]), sens_episodes=3000)
MODES["neural_full"] = dict(MODES["full"], seeds=10, tune_seeds=1, grid=dict(lr=[0.01, 0.03], B=[32], ent=[0.0, 0.01]), sens_seeds=2, sens_episodes=20000)
MODES["qfree"] = dict(MODES["full"], seeds=10)
MODES["ropen"] = dict(MODES["full"], seeds=10)
NEURAL_BASELINES = ["none", "loo_time", "linear"]
PER_EP = {"I4": 3.5e-3, "T3": 1.9e-2, "I1": 2e-4, "I2": 2e-4, "I3": 5e-4, "T1": 4e-4, "T2": 7.5e-4}       # measured seconds per training episode (single core)


def _pol_gamma(cfg, domain):
    return cfg[domain]["gamma"]


def task(t):
    cfg = load_config()
    pr = sp.get_problem(cfg, t["domain"], t["key"], t.get("horizon"))
    pol = P.make_stoch(t["domain"], t["kind"])
    if t.get("sigma0") is not None and hasattr(pol, "sigma0"):
        pol.sigma0 = t["sigma0"]
    rng0 = sp.seed_rng(t["seed"], t["key"], t["kind"])
    theta0 = pol.init(rng0)
    rng = sp.seed_rng(t["seed"], t["key"], t["kind"], "train")
    t0 = time.time()
    if t["type"] == "rf":
        h = t["hp"]
        r = reinforce(pr, pol, theta0, rng, t["episodes"], lr=h["lr"], B=h["B"], ent_coef=h["ent"], baseline=t["baseline"],
                      gamma=h.get("gamma", _pol_gamma(cfg, t["domain"])), clip=h.get("clip", 5.0), schedule="cosine", keep_theta=False)
    else:
        r = cm.METHODS[t["method"]](pr, pol, theta0, rng, t["episodes"], B=_nb(cfg, t["domain"]), **t["hp"])
    r["secs"] = time.time() - t0
    return {"task": t, "res": r}


def _nb(cfg, domain):
    return cfg[domain]["n_paths"] // 2 if domain == "trading" else 32      # paths per evaluation for the equal-budget methods


def run_pool(tasks, workers):
    if workers <= 1 or len(tasks) < 2:
        return [task(t) for t in tasks]
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(task, tasks, chunksize=1))


def summarise_tuning(results, keyfn):
    d = {}
    for r in results:
        d.setdefault(keyfn(r["task"]), []).append(r["res"]["best_val"])
    return {k: float(np.mean(v)) for k, v in d.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="both", choices=["inventory", "trading", "both"])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--quick", action="store_true")
    g.add_argument("--full", action="store_true")
    ap.add_argument("--sensitivity", action="store_true", help="also run the sensitivity study on the primary problems")
    ap.add_argument("--neural", action="store_true", help="optional neural policies R-I4 / R-T3 only (own output folder, reduced design)")
    ap.add_argument("--qmax-free", action="store_true", help="inventory R-I1/R-I2 without the Q_max cap (like FDPG); outputs/qfree")
    ap.add_argument("--return-source", choices=["close", "open"], default="close", help="'open': trading returns from next-open to following-open on the unadjusted scale; outputs/ropen")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    mode = "quick" if a.quick else "full"
    if a.neural:
        mode = "neural_" + mode
        sp.POLICIES = {"inventory": ["I4"], "trading": ["T3"]}
        sp.CM_POLICIES = {"inventory": ["I4"], "trading": ["T3"]}
        sp.BASELINES = NEURAL_BASELINES
        cm.GRIDS = {k: v for k, v in cm.GRIDS.items() if k in ("random", "spsa")}      # Nelder-Mead / central FD are not usable with ~250 parameters
    if a.qmax_free:
        assert a.full and not a.neural
        os.environ["QMAX_FREE"] = "1"
        mode = "qfree"
        a.domain = "inventory"
        sp.POLICIES = {"inventory": ["I1", "I2"], "trading": []}
        sp.CM_POLICIES = {"inventory": ["I1", "I2"], "trading": []}
        sp.BASELINES = ["none", "loo_time"]
    if a.return_source == "open":
        assert a.full and not a.neural
        os.environ["TRADING_RETURN"] = "open"
        mode = "ropen"
        a.domain = "trading"
        sp.POLICIES = {"inventory": [], "trading": ["T1", "T2"]}
        sp.CM_POLICIES = {"inventory": [], "trading": ["T1"]}
        sp.BASELINES = ["none", "loo_time"]
    M = MODES[mode]
    cfg = load_config()
    workers = a.workers or cfg["workers"]
    domains = ["inventory", "trading"] if a.domain == "both" else [a.domain]
    if "inventory" in domains and cfg["inventory"]["enabled"] is not True:
        import data_m5
        if not data_m5.status(cfg["m5_dir"])[0]:
            print("M5 incomplete -> inventory PENDING")
            domains.remove("inventory")
    out = outdir(mode, "runs")
    mdir = outdir(mode, "models")

    # ---- runtime estimate
    n_hp = int(np.prod([len(v) for v in M["grid"].values()]))
    est = 0.0
    for d in domains:
        for key in sp.problem_list(cfg, d):
            for k in sp.POLICIES[d]:
                est += PER_EP[k] * (M["episodes"] * M["seeds"] * len(sp.BASELINES) + M["episodes"] * M["tune_frac"] * n_hp * M["tune_seeds"])
            est += len(sp.CM_POLICIES[d]) * 4 * M["seeds"] * M["episodes"] * 3e-4
    print("[%s] estimated runtime ~%.0f min (%d workers; excludes sensitivity)" % (mode, est / workers / 60 * 1.3, workers), flush=True)
    T0 = time.time()

    # ---- phase 1: tune REINFORCE hyper-parameters on validation (loo_time baseline), reused for all baselines
    ghp = [dict(zip(M["grid"], v)) for v in itertools.product(*M["grid"].values())]
    tune_tasks = []
    for d in domains:
        for key in sp.problem_list(cfg, d):
            for k in sp.POLICIES[d]:
                for hi, hp in enumerate(ghp):
                    for s in range(M["tune_seeds"]):
                        tune_tasks.append(dict(type="rf", domain=d, key=key, kind=k, baseline="loo_time", hp=hp, hi=hi, seed=100 + s,
                                               episodes=int(M["episodes"] * M["tune_frac"])))
    print("tuning REINFORCE:", len(tune_tasks), "runs", flush=True)
    res = run_pool(tune_tasks, workers)
    sc = summarise_tuning(res, lambda t: (t["domain"], t["key"], t["kind"], t["hi"]))
    best_hp, tuning_table = {}, []
    for (d, key, k, hi), v in sc.items():
        tuning_table.append({"domain": d, "key": key, "policy": k, **ghp[hi], "val_J": v})
        cur = best_hp.get((d, key, k))
        if cur is None or v > cur[0]:
            best_hp[(d, key, k)] = (v, ghp[hi])
    pd.DataFrame(tuning_table).to_csv(os.path.join(mdir, "hparams_rf_grid.csv"), index=False)
    json.dump({"|".join(k): v[1] for k, v in best_hp.items()}, open(os.path.join(mdir, "hparams_rf.json"), "w"), indent=1)

    # ---- phase 1b: tune comparison methods
    ctasks = []
    for d in domains:
        for key in sp.problem_list(cfg, d):
            for k in sp.CM_POLICIES[d]:
                for m, grid in cm.GRIDS.items():
                    for hi, hp in enumerate(grid):
                        for s in range(M["tune_seeds"]):
                            ctasks.append(dict(type="cm", domain=d, key=key, kind=k, method=m, hp=hp, hi=hi, seed=100 + s, episodes=int(M["episodes"] * M["tune_frac"])))
    print("tuning comparison methods:", len(ctasks), "runs", flush=True)
    res = run_pool(ctasks, workers)
    sc = summarise_tuning(res, lambda t: (t["domain"], t["key"], t["kind"], t["method"], t["hi"]))
    best_cm = {}
    for (d, key, k, m, hi), v in sc.items():
        cur = best_cm.get((d, key, k, m))
        if cur is None or v > cur[0]:
            best_cm[(d, key, k, m)] = (v, cm.GRIDS[m][hi])
    json.dump({"|".join(k): v[1] for k, v in best_cm.items()}, open(os.path.join(mdir, "hparams_cm.json"), "w"), indent=1)

    # ---- phase 2: final runs
    final = []
    for d in domains:
        for key in sp.problem_list(cfg, d):
            for k in sp.POLICIES[d]:
                for b in sp.BASELINES:
                    for s in range(M["seeds"]):
                        final.append(dict(type="rf", domain=d, key=key, kind=k, baseline=b, hp=best_hp[(d, key, k)][1], seed=s, episodes=M["episodes"]))
            for k in sp.CM_POLICIES[d]:
                for m in cm.GRIDS:
                    for s in range(M["seeds"]):
                        final.append(dict(type="cm", domain=d, key=key, kind=k, method=m, hp=best_cm[(d, key, k, m)][1], seed=s, episodes=M["episodes"]))
    print("final runs:", len(final), flush=True)
    res = run_pool(final, workers)
    rows, models = [], {}
    for r in res:
        t, x = r["task"], r["res"]
        meth = ("reinforce:" + t["baseline"]) if t["type"] == "rf" else ("cm:" + t["method"])
        for e in x["log"]:
            rows.append({"domain": t["domain"], "key": t["key"], "policy": t["kind"], "method": meth, "seed": t["seed"], "episodes": e["episodes"],
                         "val_det": e["val_det"], "val_stoch": e.get("val_stoch", np.nan), "train_J": e.get("train_J", np.nan),
                         "sigma": e.get("sigma", np.nan), "gnorm": e.get("gnorm", np.nan)})
        mk = "%s__%s__%s" % (t["key"], t["kind"], meth.replace(":", "-"))
        models.setdefault(mk, {})[str(t["seed"])] = {"theta_best": np.asarray(x["theta_best"]).tolist(), "best_val": x["best_val"], "secs": x["secs"],
                                                   "theta_final": np.asarray(x.get("theta_final", x["theta_best"])).tolist(), "hp": t["hp"]}
    for mk, v in models.items():
        json.dump(v, open(os.path.join(mdir, mk + ".json"), "w"))
    df = pd.DataFrame(rows)
    for (d, key), g in df.groupby(["domain", "key"]):
        g.to_csv(os.path.join(out, "runs_%s__%s.csv" % (d, key)), index=False)
    print("main phase done in %.1f min" % ((time.time() - T0) / 60), flush=True)

    if a.sensitivity:
        sensitivity(cfg, domains, M, mode, workers, best_hp, a.neural)
    print("done; total %.1f min" % ((time.time() - T0) / 60))


def sensitivity(cfg, domains, M, mode, workers, best_hp, neural=False):
    """One-at-a-time sensitivity of REINFORCE (loo_time) on the primary problems: gamma, sigma0, grad clip, learning rate, horizon, B.
    Neural variant: R-I4 / R-T3, fewer seeds and episodes (M['sens_seeds'], M['sens_episodes']), no B sweep."""
    tasks = []
    eps = M.get("sens_episodes", M["episodes"]) if neural else M["episodes"]
    for d in domains:
        key = sp.problem_list(cfg, d)[0]
        kinds = (["I4"] if d == "inventory" else ["T3"]) if neural else (["I2"] if d == "inventory" else ["T1"])
        for k in kinds:
            base = dict(best_hp[(d, key, k)][1])
            vars_ = {"gamma": [0.9, 0.99, 1.0], "clip": [1.0, 5.0, 25.0], "lr": [base["lr"] / 3, base["lr"], base["lr"] * 3], "B": [16, 32, 64, 128], "sigma0": [0.25, 0.5, 1.0],
                     "horizon": [30, cfg[d]["horizon"], 180] if d == "inventory" else [125, 250, 500]}
            for name, vals in vars_.items():
                if name == "sigma0" and d != "inventory":
                    continue
                if name == "B" and neural:
                    continue
                for v in vals:
                    for s in range(M["sens_seeds"]):
                        hp = dict(base)
                        t = dict(type="rf", domain=d, key=key, kind=k, baseline="loo_time", seed=200 + s, episodes=eps, hp=hp, var=name, val=v)
                        if name in ("gamma", "clip", "lr", "B"):
                            hp[name] = v
                        elif name == "sigma0":
                            t["sigma0"] = v
                        else:
                            t["horizon"] = int(v)
                        tasks.append(t)
    print("sensitivity:", len(tasks), "runs", flush=True)
    res = run_pool(tasks, workers)
    rows = [{"domain": r["task"]["domain"], "key": r["task"]["key"], "policy": r["task"]["kind"], "variable": r["task"]["var"], "value": r["task"]["val"],
             "seed": r["task"]["seed"], "val_J": r["res"]["best_val"], "final_val_det": r["res"]["log"][-1]["val_det"]} for r in res]
    pd.DataFrame(rows).to_csv(os.path.join(outdir(mode, "runs"), "sensitivity.csv"), index=False)


if __name__ == "__main__":
    sys.exit(main())
