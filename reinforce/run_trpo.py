"""Train TRPO and its ablations; tune on validation only. Re-uses REINFORCE / comparison-method runs from outputs/full as equal-budget baselines.

python run_trpo.py --domain inventory|trading|both --quick|--full [--sensitivity] [--workers N]
Outputs (./outputs/trpo/<mode>): models/, runs/ (validation curves), iters/ (per-iteration diagnostics), hparams, sensitivity.csv.
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
from trpo.algorithm import trpo
from trpo.ablations import VARIANTS, variant_kwargs

MODES = {
    "quick": dict(episodes=6000, seeds=3, tune_seeds=1, tune_frac=0.5, delta=[0.01, 0.03], lam=[0.97], lr=[0.03, 0.1], sens_seeds=2),
    "full": dict(episodes=40000, seeds=10, tune_seeds=2, tune_frac=0.5, delta=[0.003, 0.01, 0.03, 0.1], lam=[0.9, 0.97, 1.0], lr=[0.01, 0.03, 0.1], sens_seeds=5),
}
MODES["neural_quick"] = dict(MODES["quick"], episodes=3000, seeds=2, delta=[0.01, 0.03], lam=[0.97], lr=[0.03])
MODES["neural_full"] = dict(MODES["full"], tune_seeds=1, delta=[0.01, 0.03, 0.1], lam=[0.9, 0.97], lr=[0.01, 0.03])
MODES["qfree"] = dict(MODES["full"])
MODES["ent"] = dict(MODES["full"], ent=[0.003, 0.01, 0.03])
MODES["ent_quick"] = dict(MODES["quick"], ent=[0.01, 0.03])
PER_EP = {"I1": 3e-4, "I2": 3e-4, "I3": 1e-3, "T1": 5e-4, "T2": 1e-3, "I4": 7e-3, "T3": 3.5e-2}      # rough seconds per training episode incl. diagnostics (single core)
MAIN_VARIANTS = ["trpo", "vpg", "ng_fixed", "trpo_mc", "trpo_nostd"]
B_DEFAULT = 32


def task(t):
    cfg = load_config()
    pr = sp.get_problem(cfg, t["domain"], t["key"], t.get("horizon"))
    pol = P.make_stoch(t["domain"], t["kind"])
    if t.get("sigma0") is not None and hasattr(pol, "sigma0"):
        pol.sigma0 = t["sigma0"]
    theta0 = pol.init(sp.seed_rng(t["seed"], t["key"], t["kind"]))
    rng = sp.seed_rng(t["seed"], t["key"], t["kind"], "train")
    kw = dict(t["kw"])
    kw.setdefault("gamma", cfg[t["domain"]]["gamma"])
    kw.setdefault("diag", t.get("diag", True))
    r = trpo(pr, pol, theta0, rng, t["episodes"], **kw)
    return {"task": t, "res": r}


def run_pool(tasks, workers):
    if workers <= 1 or len(tasks) < 2:
        return [task(t) for t in tasks]
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(task, tasks, chunksize=1))


def mean_by(results, keyfn):
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
    ap.add_argument("--sensitivity", action="store_true")
    ap.add_argument("--neural", action="store_true", help="neural policies R-I4 / R-T3 (TRPO and VPG only; outputs/trpo/neural_<mode>)")
    ap.add_argument("--qmax-free", action="store_true", help="R-I1/R-I2 without the Q_max cap (full mode; outputs/trpo/qfree)")
    ap.add_argument("--entropy", action="store_true", help="TRPO + entropy bonus on the linear policies, coefficient tuned on validation (outputs/trpo/ent)")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    mode = "quick" if a.quick else "full"
    variants, extra, pols_over = MAIN_VARIANTS, {}, None
    if a.neural:
        mode = "neural_" + mode
        variants, extra, pols_over = ["trpo", "vpg"], {"diag_mult": 1}, {"inventory": ["I4"], "trading": ["T3"]}
    if a.qmax_free:
        assert a.full and not a.neural and not a.entropy
        os.environ["QMAX_FREE"] = "1"
        mode, a.domain = "qfree", "inventory"
        variants, pols_over = ["trpo", "vpg"], {"inventory": ["I1", "I2"], "trading": []}
    if a.entropy:
        assert not a.neural and not a.qmax_free
        mode = "ent" if a.full else "ent_quick"
        variants = ["trpo_ent"]
    M = MODES[mode]
    cfg = load_config()
    workers = a.workers or cfg["workers"]
    domains = ["inventory", "trading"] if a.domain == "both" else [a.domain]
    mdir, rdir, idir = outdir("trpo", mode, "models"), outdir("trpo", mode, "runs"), outdir("trpo", mode, "iters")
    pols = {d: (pols_over[d] if pols_over else [k for k in sp.POLICIES[d] if k in PER_EP and k not in ("I4", "T3")]) for d in domains}

    n_tr = len(M["delta"]) * len(M["lam"])
    est = 0.0
    for d in domains:
        for key in sp.problem_list(cfg, d):
            for k in pols[d]:
                tune = len(M["ent"]) if a.entropy else (n_tr + len(M["lr"]))
                est += PER_EP[k] * (M["episodes"] * M["tune_frac"] * tune * M["tune_seeds"] + M["episodes"] * M["seeds"] * len(variants))
    print("[%s] estimated runtime ~%.0f min (%d workers; excludes sensitivity)" % (mode, est / workers / 60 * 1.3, workers), flush=True)
    T0 = time.time()
    half = int(M["episodes"] * M["tune_frac"])

    # ---- phase 1: tune TRPO (delta x lambda) and VPG (lr) on validation only; entropy mode: tune only ent_coef at TRPO's tuned (delta, lambda)
    tt = []
    prev = {}
    if a.entropy:
        prev = json.load(open(os.path.join(outdir("trpo", "full" if a.full else "quick", "models"), "hparams_trpo.json")))
    for d in domains:
        for key in sp.problem_list(cfg, d):
            for k in pols[d]:
                if a.entropy:
                    h = prev["%s|%s|%s" % (d, key, k)]["trpo"]
                    for e in M["ent"]:
                        for s in range(M["tune_seeds"]):
                            tt.append(dict(domain=d, key=key, kind=k, seed=100 + s, episodes=half, diag=False, tune="ent", hp=dict(ent_coef=e),
                                           kw=dict(B=B_DEFAULT, delta=h["delta"], lam=h["lam"], ent_coef=e, n_eval=10)))
                    continue
                for de, la in itertools.product(M["delta"], M["lam"]):
                    for s in range(M["tune_seeds"]):
                        tt.append(dict(domain=d, key=key, kind=k, seed=100 + s, episodes=half, diag=False, tune="trpo", hp=dict(delta=de, lam=la),
                                       kw=dict(B=B_DEFAULT, delta=de, lam=la, n_eval=10, **extra)))
                for lr in M["lr"]:
                    for s in range(M["tune_seeds"]):
                        tt.append(dict(domain=d, key=key, kind=k, seed=100 + s, episodes=half, diag=False, tune="vpg", hp=dict(lr=lr),
                                       kw=dict(B=B_DEFAULT, variant="vpg", lr=lr, n_eval=10, **extra)))
    print("tuning:", len(tt), "runs", flush=True)
    res = run_pool(tt, workers)
    sc = mean_by(res, lambda t: (t["domain"], t["key"], t["kind"], t["tune"], json.dumps(t["hp"], sort_keys=True)))
    best, table = {}, []
    for (d, key, k, tn, hp), v in sc.items():
        table.append({"domain": d, "key": key, "policy": k, "tuned": tn, "hp": hp, "val_J": v})
        c = best.get((d, key, k, tn))
        if c is None or v > c[0]:
            best[(d, key, k, tn)] = (v, json.loads(hp))
    pd.DataFrame(table).to_csv(os.path.join(mdir, "hparams_trpo_grid.csv"), index=False)
    hps = {}
    for (d, key, k, tn), (v, hp) in best.items():
        hps.setdefault("%s|%s|%s" % (d, key, k), {})[tn] = hp
    if a.entropy:
        for kk, v in hps.items():
            v.update({"trpo": prev[kk]["trpo"], "vpg": prev[kk]["vpg"]})
    json.dump(hps, open(os.path.join(mdir, "hparams_trpo.json"), "w"), indent=1)

    # ---- phase 2: final runs of all variants (validation-selected hyper-parameters; ablations share TRPO's)
    final = []
    for d in domains:
        for key in sp.problem_list(cfg, d):
            for k in pols[d]:
                h = hps["%s|%s|%s" % (d, key, k)]
                base = dict(B=B_DEFAULT, delta=h["trpo"]["delta"], lam=h["trpo"]["lam"], lr=h["vpg"]["lr"], **extra)
                if a.entropy:
                    base["ent_coef"] = h["ent"]["ent_coef"]
                for v in variants:
                    for s in range(M["seeds"]):
                        final.append(dict(domain=d, key=key, kind=k, variant_name=v, seed=s, episodes=M["episodes"], kw=variant_kwargs(v, base)))
    print("final runs:", len(final), flush=True)
    res = run_pool(final, workers)
    rows, its, models = [], [], {}
    for r in res:
        t, x = r["task"], r["res"]
        base = {"domain": t["domain"], "key": t["key"], "policy": t["kind"], "method": "trpo:" + t["variant_name"], "seed": t["seed"]}
        for e in x["log"]:
            rows.append({**base, "episodes": e["episodes"], "val_det": e["val_det"], "val_stoch": e["val_stoch"], "sigma": e["sigma"]})
        for e in x["it_log"]:
            its.append({**base, **e})
        mk = "%s__%s__trpo-%s" % (t["key"], t["kind"], t["variant_name"])
        models.setdefault(mk, {})[str(t["seed"])] = {"theta_best": np.asarray(x["theta_best"]).tolist(), "theta_final": np.asarray(x["theta_final"]).tolist(),
                                                  "best_val": x["best_val"], "secs": x["secs"], "hp": {k_: v_ for k_, v_ in t["kw"].items()}}
    for mk, v in models.items():
        json.dump(v, open(os.path.join(mdir, mk + ".json"), "w"))
    for (d, key), g in pd.DataFrame(rows).groupby(["domain", "key"]):
        g.to_csv(os.path.join(rdir, "runs_%s__%s.csv" % (d, key)), index=False)
    for (d, key), g in pd.DataFrame(its).groupby(["domain", "key"]):
        g.to_csv(os.path.join(idir, "iters_%s__%s.csv" % (d, key)), index=False)
    print("main phase done in %.1f min" % ((time.time() - T0) / 60), flush=True)

    if a.sensitivity and mode in ("quick", "full"):
        sensitivity(cfg, domains, M, mode, workers, hps, pols)
    print("done; total %.1f min" % ((time.time() - T0) / 60))


def sensitivity(cfg, domains, M, mode, workers, hps, pols):
    """One-at-a-time sweeps of TRPO on the primary problems (first series; I2 / T1) around the tuned setting."""
    tasks = []
    for d in domains:
        key = sp.problem_list(cfg, d)[0]
        k = "I2" if d == "inventory" else "T1"
        if k not in pols[d]:
            continue
        h = hps["%s|%s|%s" % (d, key, k)]["trpo"]
        base = dict(B=B_DEFAULT, delta=h["delta"], lam=h["lam"])
        g0 = cfg[d]["gamma"]
        sweeps = {"delta": [0.003, 0.01, 0.03, 0.1], "lam": [0.0, 0.9, 0.97, 1.0], "gamma": [0.9, g0, 1.0], "B": [8, 16, 32, 64, 128],
                  "damp": [1e-4, 1e-2, 1e-1, 1.0], "cg_iters": [2, 5, 10, 20], "sigma0": [0.25, 0.5, 1.0], "std": [True, False],
                  "horizon": [30, cfg[d]["horizon"], 180] if d == "inventory" else [125, 250, 500]}
        for name, vals in sweeps.items():
            if name == "sigma0" and d != "inventory":
                continue
            for v in vals:
                for s in range(M["sens_seeds"]):
                    kw = dict(base, n_eval=10)
                    t = dict(domain=d, key=key, kind=k, seed=200 + s, episodes=M["episodes"], diag=False, var=name, val=v, kw=kw)
                    if name == "sigma0":
                        t["sigma0"] = v
                    elif name == "horizon":
                        t["horizon"] = int(v)
                    else:
                        kw[name] = v
                    tasks.append(t)
    print("sensitivity:", len(tasks), "runs", flush=True)
    res = run_pool(tasks, workers)
    rows = [{"domain": r["task"]["domain"], "key": r["task"]["key"], "policy": r["task"]["kind"], "variable": r["task"]["var"], "value": r["task"]["val"],
             "seed": r["task"]["seed"], "val_J": r["res"]["best_val"], "final_val_det": r["res"]["log"][-1]["val_det"],
             "min_ret_step": np.nan} for r in res]
    pd.DataFrame(rows).to_csv(os.path.join(outdir("trpo", mode, "runs"), "sensitivity.csv"), index=False)


if __name__ == "__main__":
    sys.exit(main())
