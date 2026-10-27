"""Train policy-gradient agents. Examples:
  python run_pg.py --quick                                  # smoke test, all algorithms, 1 seed, tiny budget
  python run_pg.py --full --objective neutral --algorithm ppo
  python run_pg.py --full --jobs 12                         # everything: 6 variants x 3 objectives x 10 seeds
  python run_pg.py --grid --objective neutral               # small validation-only learning-rate grid
Budgets are counted in rollouts (delivery days), identical across algorithms.
Runtime guide (16 CPUs): --quick ~5-10 min; --full for one objective ~1-2 h; all objectives several hours."""
import argparse
import itertools
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import data
from pg.actor_critic import A2C
from pg.dpg import DPG
from pg.fdpg import FDPG
from pg.ppo import PPO
from pg.reinforce import Reinforce
from pg.trpo import TRPO

ALGOS = {
    "fdpg": (FDPG, {"fd": "central", "N": 32, "lr": 0.002, "eps": 0.02}),
    "fdpg_spsa": (FDPG, {"fd": "spsa", "N": 32, "n_dir": 4, "lr": 0.002, "eps": 0.02}),
    "reinforce": (Reinforce, {"baseline": "loo", "B": 64, "lr": 3e-4}),
    "a2c": (A2C, {"B": 64, "lr": 3e-4, "vf_lr": 3e-3}),
    "dpg": (DPG, {"B": 16, "lr": 3e-5, "q_lr": 1e-3}),
    "trpo": (TRPO, {"B": 64, "delta": 0.003}),
    "ppo": (PPO, {"B": 64, "lr": 3e-4, "clip": 0.2, "epochs": 10, "minibatch": 64}),
}
LR_GRID = {"fdpg": [5e-4, 2e-3, 8e-3], "fdpg_spsa": [5e-4, 2e-3, 8e-3], "reinforce": [1e-4, 3e-4, 1e-3], "a2c": [1e-4, 3e-4, 1e-3],
           "dpg": [1e-5, 3e-5, 1e-4], "trpo": [0.001, 0.003, 0.01], "ppo": [1e-4, 3e-4, 1e-3]}  # trpo grid = KL delta
BUDGET = {"quick": 3000, "grid": 12000, "full": 30000}
RISK_DEFAULT = {"neutral": {}, "cvar": {"alpha": 0.95, "lam": 1.0}, "meanvar": {"kappa": 8.0},
                "neutral_cond": {}, "cvar_cond": {"alpha": 0.95, "lam": 1.0}, "meanvar_cond": {"kappa": 8.0}}
OUT = os.path.join(data.OUT, "pg")
_CTX = {}


def context(rho=None):
    if "P" not in _CTX:
        from problem import Problem
        from agents import forecast as F
        from pg.env import DayEnv
        P = Problem(rho=rho)
        fc = F.make_forecasts(P)
        env = DayEnv(P, fc)
        from pg.env import distill_theta
        _CTX.update(P=P, fc=fc, env=env, theta=distill_theta(env, P, fc), refs={})
    return _CTX


def make_cfg(algo, objective, mode, over=None):
    cfg = dict(ALGOS[algo][1])
    cfg.update(RISK_DEFAULT[objective])
    cfg["budget_days"] = BUDGET[mode]
    cfg["lam_gae"] = 0.0 if objective in ("neutral", "neutral_cond") else 0.95  # neutral: actions only affect their own hour
    if over:
        if algo == "trpo" and "lr" in over:
            over = {**{k: v for k, v in over.items() if k != "lr"}, "delta": over["lr"]}
        cfg.update(over)
    return cfg


def run_one(job):
    algo, objective, seed, mode, over, warm, tag, outroot = job
    from pg.common import run_training, references, make_risk
    C = context()
    cfg = make_cfg(algo, objective, mode, over)
    cfg["warm_start"] = warm
    key = (objective, json.dumps(RISK_DEFAULT[objective]))
    if key not in C["refs"]:
        C["refs"][key] = references(C["env"], objective, make_risk(objective, cfg), cfg)
    out = os.path.join(outroot, objective, tag or algo, f"seed_{seed}")
    res = run_training(ALGOS[algo][0], C["P"], C["fc"], objective, seed, cfg, out_dir=out, env=C["env"],
                       refs=C["refs"][key], theta_dist=C["theta"])
    f = res["final"]
    f["tag"] = tag or algo
    return f


def summarize(outroot):
    rows = []
    for obj in [o for o in os.listdir(outroot) if o in RISK_DEFAULT] if os.path.isdir(outroot) else []:
        for a in os.listdir(os.path.join(outroot, obj)):
            for s in os.listdir(os.path.join(outroot, obj, a)):
                p = os.path.join(outroot, obj, a, s, "final.json")
                if os.path.exists(p):
                    f = json.load(open(p))
                    r = {"objective": obj, "algo": a, "seed": f["seed"], "days_to_95": f["days_to_95"], "days_to_99": f["days_to_99"],
                         "total_days": f["total_days"], "wall_s": f["wall_s"], "worst_drop": f["worst_drop"], "diverged": f["diverged"],
                         "sigma_collapsed": f["sigma_collapsed"], "best_iter": f["best_iter"]}
                    for lbl in ("final", "best_val"):
                        for sp in ("val", "test"):
                            for k in ("obj", "skill", "gap_rel", "mean_eur", "cvar95", "H"):
                                r[f"{lbl}_{sp}_{k}"] = f[f"{lbl}_{sp}"][k]
                    rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outroot, "summary.csv"), index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--quick", action="store_true")
    g.add_argument("--full", action="store_true")
    ap.add_argument("--objective", default="all", choices=["all", "all_cond"] + list(RISK_DEFAULT))
    ap.add_argument("--algorithm", default="all", choices=["all"] + list(ALGOS))
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--warm", action="store_true", help="warm-start the linear mean from the distilled newsvendor policy")
    ap.add_argument("--grid", action="store_true", help="validation-only learning-rate grid (2 seeds), writes pg/grid.csv")
    ap.add_argument("--policy", default="linear", choices=["linear", "mlp"], help="mean-network type (mlp: 1 hidden tanh layer; FDPG is linear-only)")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    mode = "grid" if a.grid else ("full" if a.full else "quick")
    seeds = a.seeds or (10 if a.full else 1)
    objs = {"all": ["neutral", "cvar", "meanvar"], "all_cond": ["neutral_cond", "cvar_cond", "meanvar_cond"]}.get(a.objective, [a.objective])
    algs = list(ALGOS) if a.algorithm == "all" else [a.algorithm]
    t0 = time.time()
    jobs = []
    if a.grid:
        for o, al in itertools.product(objs, algs):
            for lr in LR_GRID[al]:
                lr = lr / 10 if a.policy != "linear" else lr  # networks need smaller steps than the linear policy
                for s in range(100, 102):  # grid seeds are disjoint from final seeds
                    over = {"lr": lr, **({"policy": a.policy} if a.policy != "linear" else {})}
                    tg = al + ("" if a.policy == "linear" else "_" + a.policy)
                    jobs.append((al, o, s, mode, over, a.warm, f"grid_{tg}_{lr}", os.path.join(a.out, "grid")))
    else:
        cfgs = {}
        bp = os.path.join(a.out, "best_cfg.json")
        if os.path.exists(bp):
            cfgs = json.load(open(bp))
        for o, al, s in itertools.product(objs, algs, range(seeds)):
            if a.policy == "linear":
                jobs.append((al, o, s, mode, cfgs.get(f"{o}|{al}") or cfgs.get(f"neutral|{al}"), a.warm, al, a.out))
            else:
                tg = f"{al}_{a.policy}"
                over = {**(cfgs.get(f"{o}|{tg}") or {}), "policy": a.policy}
                jobs.append((al, o, s, mode, over, a.warm, tg, a.out))
    print(f"{len(jobs)} runs, budget {BUDGET[mode]} rollouts each, {a.jobs} workers", flush=True)
    os.makedirs(a.out, exist_ok=True)
    if a.jobs <= 1:
        res = [run_one(j) for j in jobs]
    else:
        context()  # warm caches (forecast bundle, day arrays) before forking workers
        with ProcessPoolExecutor(a.jobs) as ex:
            res = []
            for i, r in enumerate(ex.map(run_one, jobs)):
                res.append(r)
                print(f"[{i + 1}/{len(jobs)}] {r['objective']}/{r['tag']}/seed{r['seed']}: best-val skill "
                      f"{r['best_val_val']['skill']:.3f}, test skill {r['best_val_test']['skill']:.3f}, {r['wall_s']:.0f}s", flush=True)
    if a.grid:
        rows = [{"objective": r["objective"], "algo": r["algo"], "tag": r["tag"], "seed": r["seed"],
                 "val_skill": r["final_val"]["skill"], "val_obj": r["final_val"]["obj"]} for r in res]
        G = pd.DataFrame(rows)
        G["lr"] = G.tag.str.split("_").str[-1].astype(float)
        G["algo"] = G.tag.str.replace(r"^grid_", "", regex=True).str.replace(r"_[^_]+$", "", regex=True)
        G.to_csv(os.path.join(a.out, "grid.csv"), index=False)
        best = {}
        for (o, al), d in G.groupby(["objective", "algo"]):
            m = d.groupby("lr").val_skill.mean()
            best[f"{o}|{al}"] = {"lr": float(m.idxmax())}
        bp = os.path.join(a.out, "best_cfg.json")
        old = json.load(open(bp)) if os.path.exists(bp) else {}
        old.update(best)
        json.dump(old, open(bp, "w"), indent=1)
        print(G.groupby(["objective", "algo", "lr"]).val_skill.mean().to_string())
    else:
        df = summarize(a.out)
        print(df.groupby(["objective", "algo"])[["best_val_val_skill", "best_val_test_skill", "days_to_95"]].median().to_string())
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
