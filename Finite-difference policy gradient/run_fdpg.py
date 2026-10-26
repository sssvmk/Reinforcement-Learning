"""Stage 3: train FD policy-gradient estimators and baselines. Writes ./outputs/runs, models, logs.

  python run_fdpg.py --domain trading --quick      (smoke test, minutes)
  python run_fdpg.py --domain all --full           (full study)
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse
import itertools
import json
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import numpy as np

import data_m5
import data_trading as dt
from audit_data import load_config, HERE
from fdpg.baselines import grid_search, nelder_mead, random_search
from fdpg.estimators import ESTIMATORS
from fdpg.optimizer import HParams, Scaler, project, run_fd
from problems import build_inventory_problem, build_trading_problem

OUT = os.path.join(HERE, "outputs")
FD_METHODS = list(ESTIMATORS) + ["central_nocrn", "spsa_nocrn"]
BASELINES = ["grid", "random", "neldermead"]
POLICIES = {"trading": ["T1", "T2"], "inventory": ["I1", "I2", "I3"]}
MODES = {
    "quick": dict(seeds=3, budget=6000, tune_seeds=1, eps=[0.1, 0.3], a=[1.0, 4.0], n_scale=0.5),
    "full": dict(seeds=10, budget=40000, tune_seeds=2, eps=[0.05, 0.15, 0.4], a=[0.5, 2.0, 8.0], n_scale=1.0),
}
_CACHE = {}


def ticker(path):
    return os.path.basename(path).split("_")[0]


def list_series(cfg, domain, log=print):
    """Series ids for a domain; inventory returns [] (with a log line) while M5 files are missing."""
    if domain == "trading":
        res = []
        for p in cfg["trading_files"]:
            df = dt.read_trading_safe(p, log)
            if df is None:
                continue
            try:
                dt.build_features(df, cfg["trading"]["warmup_rows"], cfg["trading"]["allow_adjusted_close"])
                res.append(ticker(p))
            except dt.SchemaError as e:
                log(f"EXCLUDED {p}: {e}")
        return res
    ok, missing, found = data_m5.status(cfg["m5_dir"])
    if cfg["inventory"]["enabled"] is False or (cfg["inventory"]["enabled"] == "auto" and not ok):
        log("INVENTORY PENDING / disabled: " + "; ".join(missing))
        return []
    prof = data_m5.profile_candidates(found)
    return list(data_m5.choose_series(prof).values())


def get_problem(cfg, domain, series, policy, n_scale=1.0, splits=None):
    key = (domain, series, policy, n_scale, None if splits is None else tuple(splits.items()))
    if key in _CACHE:
        return _CACHE[key]
    if domain == "trading":
        path = next(p for p in cfg["trading_files"] if ticker(p) == series)
        df = dt.read_trading_safe(path)
        data = dt.build_features(df, cfg["trading"]["warmup_rows"], cfg["trading"]["allow_adjusted_close"])
        tcfg = dict(cfg["trading"]); tcfg["n_paths"] = max(8, int(tcfg["n_paths"] * n_scale))
        prob = build_trading_problem(data, policy, tcfg, splits=splits, name=series)
    else:
        _, _, found = data_m5.status(cfg["m5_dir"])
        df = data_m5.load_series(found, series, os.path.join(OUT, "series"))
        icfg = dict(cfg["inventory"]); icfg["n_paths"] = max(8, int(icfg["n_paths"] * n_scale))
        prob = build_inventory_problem(df, policy, icfg, name=series)
    _CACHE.clear()
    _CACHE[key] = prob
    return prob


def val_curve(prob, trace, Jv, Jd, n_pts=30):
    """Post-hoc validation curve for baselines: (rollouts, val, diag) at <= n_pts checkpoints."""
    idx = np.unique(np.r_[np.unique(np.geomspace(1, len(trace), n_pts).astype(int)) - 1, len(trace) - 1])
    return [(trace[i][0], Jv(trace[i][1], Jv.fixed_paths), Jd(trace[i][1], Jd.fixed_paths)) for i in idx]


def train_one(prob, method, seed, hp, budget):
    pol = prob.policy
    rng = np.random.default_rng(seed)
    sc = Scaler(pol.scale)
    lo, hi = pol.ilo, pol.ihi
    u0 = project(sc.to_internal(pol.init(rng)), lo, hi)
    Jt, Jv, Jd = prob.objectives()
    t0 = time.perf_counter()
    traj = None
    if method in FD_METHODS:
        est = method.replace("_nocrn", "")
        h = replace(hp, est=est, crn=not method.endswith("_nocrn"), budget=budget, val_every=2)
        out = run_fd(Jt, Jv, u0, lo, hi, h, seed + 1000, Jdiag=Jd)
        u = out["u_best"]
        curve = [(c[1], c[2], c[3]) for c in out["curve"]]
        traj = np.array([c[4] for c in out["curve"]])
        iters = out["iters"]
    else:
        if method == "grid":
            out = grid_search(Jt, lo, hi, budget, rng)
        elif method == "random":
            out = random_search(Jt, lo, hi, budget, rng)
        else:
            out = nelder_mead(Jt, u0, lo, hi, budget, rng)
        if out is None:
            return None
        u, iters = out["u_best"], len(out["trace"])
        curve = val_curve(prob, [(0, u0)] + out["trace"], Jv, Jd)
    rt = time.perf_counter() - t0
    return {"method": method, "seed": seed, "u0": u0, "u_final_internal": u, "theta": sc.to_phys(u),
            "val_final": Jv(u, Jv.fixed_paths), "train_final": Jd(u, Jd.fixed_paths),
            "rollouts": Jt.rollouts, "iters": iters, "runtime": rt, "curve": np.array(curve, float), "traj": traj,
            "hp": hp.asdict(), "at_bound": bool(np.any(np.isclose(u, lo, atol=1e-6) | np.isclose(u, hi, atol=1e-6)))}


def _task(spec):
    cfg, domain, series, policy, method, seed, hp, budget, n_scale = spec
    prob = get_problem(cfg, domain, series, policy, n_scale)
    return train_one(prob, method, seed, hp, budget)


def tune(cfg, domain, series, policy, mode, pool):
    """Small validation-only grid on (eps, a) for the central estimator; reused for all estimators."""
    m = MODES[mode]
    path = os.path.join(OUT, "models", f"hparams__{domain}__{series}__{policy}__{mode}.json")
    if os.path.exists(path):
        return HParams(**json.load(open(path))["hp"])
    grid = list(itertools.product(m["eps"], m["a"]))
    specs = [(cfg, domain, series, policy, "central", 500 + s, HParams(eps=e, a=a), m["budget"] // 2, m["n_scale"])
             for (e, a) in grid for s in range(m["tune_seeds"])]
    res = list(pool.map(_task, specs))
    k = m["tune_seeds"]
    scores = [float(np.mean([r["val_final"] for r in res[i * k:(i + 1) * k]])) for i in range(len(grid))]
    e, a = grid[int(np.argmax(scores))]
    hp = HParams(eps=e, a=a)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump({"hp": hp.asdict(), "grid": [list(g) for g in grid], "val_scores": scores}, open(path, "w"), indent=1)
    return hp


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["trading", "inventory", "all"], default="all")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--quick", action="store_true")
    g.add_argument("--full", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args(argv)
    mode = "quick" if a.quick else "full"
    m = MODES[mode]
    cfg = load_config()
    workers = a.workers or cfg.get("workers", 4)
    for d in ("runs", "logs", "models"):
        os.makedirs(os.path.join(OUT, d), exist_ok=True)
    logf = open(os.path.join(OUT, "logs", f"run_{mode}.log"), "a")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n"); logf.flush()

    domains = ["trading", "inventory"] if a.domain == "all" else [a.domain]
    jobs = [(d, s, p) for d in domains for s in list_series(cfg, d, log) for p in POLICIES[d]]
    if not jobs:
        log("nothing to run"); return 1
    d0, s0, p0 = jobs[0]
    pr = get_problem(cfg, d0, s0, p0, m["n_scale"])
    Jt = pr.objective(); rng = np.random.default_rng(0)
    t = time.perf_counter(); Jt(np.zeros(pr.policy.n), Jt.draw(rng)); te = time.perf_counter() - t
    n_tasks = (len(FD_METHODS) + len(BASELINES)) * m["seeds"] * len(jobs)
    est = (n_tasks * m["budget"] / Jt.n_paths * te * 1.4 + len(jobs) * len(m["eps"]) * len(m["a"]) * m["tune_seeds"]
           * m["budget"] / 2 / Jt.n_paths * te) / workers
    log(f"[{mode}] {len(jobs)} problem(s), {n_tasks} runs, {workers} workers; estimated runtime ~{est/60:.1f} min")
    t_all = time.time()
    with ProcessPoolExecutor(workers) as pool:
        for domain, series, policy in jobs:
            hp = tune(cfg, domain, series, policy, mode, pool)
            log(f"{domain}/{series}/{policy}: tuned eps={hp.eps} a={hp.a}")
            specs = [(cfg, domain, series, policy, meth, sd, hp, m["budget"], m["n_scale"])
                     for meth in FD_METHODS + BASELINES for sd in range(m["seeds"])]
            recs = [r for r in pool.map(_task, specs) if r is not None]
            pickle.dump({"records": recs, "mode": mode, "hp": hp.asdict(), "budget": m["budget"]},
                        open(os.path.join(OUT, "runs", f"{domain}__{series}__{policy}__{mode}.pkl"), "wb"))
            models = {}
            for r in recs:
                models.setdefault(r["method"], {})[r["seed"]] = r["theta"].tolist()
            json.dump(models, open(os.path.join(OUT, "models", f"{domain}__{series}__{policy}__{mode}.json"), "w"), indent=1)
            for meth in FD_METHODS + BASELINES:
                v = [r["val_final"] for r in recs if r["method"] == meth]
                if v:
                    log(f"  {meth:15s} val J = {np.mean(v):8.3f} +/- {np.std(v):6.3f}  (n={len(v)})")
    log(f"done in {(time.time()-t_all)/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
