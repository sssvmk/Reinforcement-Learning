"""End-to-end offline-RL inventory pipeline: logs -> candidate pool -> ground truth -> estimators -> ranking metrics.

    python run_experiment.py --quick        # tiny smoke test (minutes)
    python run_experiment.py --full         # main experiment (+ replications), --ablations for the ablation grid
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import model_error, policy_model_error, support_fraction
from candidates.base import Policy
from candidates.pool import build_pool, distill_mcts
from data import ROOT, load_config, load_series, split_bounds
from discretize import Discretizer, coverage_report
from env import DemandModel, make_params
from estimators import ce_ope_estimate, dr_estimate, wis_estimate
from fqe import fqe_evaluate
from candidates.ce import FittedMDP
from log_data import (BehaviorPolicy, generate_log, log_summary, save_log, split_log_ids, subset_log, tune_base_stock)
from ranking_metrics import family_metrics, metrics_for, tie_matrix
from true_value import summarize, truth_table

ESTIMATORS = ["fqe", "self", "ce_ope", "wis", "dr", "behavior", "random"]


class BehaviorAsPolicy(Policy):
    family = "behavior"
    stochastic = True

    def __init__(self, beh):
        super().__init__("behavior", beh.p.n_actions, {})
        self.beh = beh

    def probs(self, S, eps=None):
        return self.beh.probs(S)

    def greedy(self, S):
        return self.beh.probs(S).argmax(1)


def out_path(cfg):
    o = Path(cfg["out_dir"])
    return o if o.is_absolute() else ROOT / o


def estimate_runtime(cfg, n_cand, quick):
    f = cfg["fqe"]
    steps = n_cand * f["seeds"] * f["rounds"] * f["steps_per_round"]
    return steps * 1.5e-3 / 60.0


def run_one(cfg, sid, rep, cond="medium", n_episodes=None, quick=False, families=None, tag="main", verbose=True,
            fqe_over=None, keep_pool=False):
    t0 = time.time()
    say = print if verbose else (lambda *a, **k: None)
    seed = cfg["seed"] + 1000 * rep
    out = out_path(cfg)
    ser = load_series(cfg, sid)
    p = make_params(ser, cfg)
    lo, hi = split_bounds(len(ser), cfg)["train"]
    dm_args = (ser.sales[lo:hi], ser.wday[lo:hi], ser.snap[lo:hi])            # train period ONLY
    dm = DemandModel(*dm_args, kind=cfg["env"]["demand_model"], seed=seed)
    S_star, v_star = tune_base_stock(ser, cfg, p, dm, seed=seed + 1)
    lc = cfg["logging"]
    eps = lc["eps"]["medium" if cond == "mixture" else cond]
    levels = [0.7 * S_star, S_star, 1.3 * S_star] if cond == "mixture" else [S_star]
    beh = BehaviorPolicy(p, levels, eps)
    disc = Discretizer(p, cfg["discretize"]["n_pos_bins"], cfg["discretize"]["n_oh_bins"], cfg["discretize"]["use_event"])
    N = n_episodes or lc["n_episodes"]
    ids = np.arange(N) + 10_000 * rep
    full = generate_log(ser, cfg, p, dm, disc, beh, "train", ids, seed)
    tr_ids, fq_ids = split_log_ids(N, lc["fqe_frac"], seed, 10_000 * rep)
    assert not set(tr_ids) & set(fq_ids)
    train_log, fqe_log = subset_log(full, tr_ids), subset_log(full, fq_ids)
    val_log = generate_log(ser, cfg, p, dm, disc, beh, "val", np.arange(lc["n_val_episodes"]) + 1_000_000, seed)
    name = f"{sid}_rep{rep}_{cond}_n{N}_{tag}"
    meta = {"series": sid, "cond": cond, "eps": eps, "S_levels": levels, "n_episodes": N, "seed": seed,
            "train_log_cov": log_summary(train_log, disc, p.n_actions), "fqe_log_cov": log_summary(fqe_log, disc, p.n_actions),
            "val_log_cov": log_summary(val_log, disc, p.n_actions), "n_actions": p.n_actions, "actions": p.actions.tolist()}
    save_log({k: v for k, v in train_log.items()}, out / "logs" / (name + "_trainlog"), meta)
    save_log({k: v for k, v in fqe_log.items()}, out / "logs" / (name + "_fqelog"), meta)
    say(f"[{name}] S*={S_star:.1f} cov(train)={meta['train_log_cov']['frac_sa_visited']:.2f} "
        f"states={meta['train_log_cov']['n_states']} sa={meta['train_log_cov']['n_state_action']}")

    pool = build_pool(train_log, p, disc, dm_args, S_star, cfg, seed, beh, quick, families, verbose)
    say(f"  pool size {len(pool)}; est. FQE runtime ~{estimate_runtime(cfg, len(pool), quick):.1f} min")
    for pol in pool:
        pol.save(out / "candidates" / name)

    # --- ground truth on the TEST period (common random numbers)
    tc = cfg["truth"]
    nT, nM = (60, 4) if quick else (tc["n_episodes"], tc["n_episodes_mcts"])
    hz = 60 if quick else tc["horizon"]
    rets = truth_table(pool + [BehaviorAsPolicy(beh)], ser, cfg, p, dm, nT, nM, hz, seed + 7, tc["mode"])
    rets_soft = truth_table(pool, ser, cfg, p, dm, nT, nM, hz, seed + 7, tc["mode"], soft=True)
    T = summarize(rets)
    beh_true = T.pop("behavior")
    Ts = summarize(rets_soft)
    say(f"  truth done ({time.time() - t0:.0f}s); behavior true value {beh_true['truth']:.2f}")

    # --- estimators
    f = {**cfg["fqe"], **(fqe_over or {})}
    fk = dict(rounds=f["rounds"], steps=f["steps_per_round"], lr=f["lr"], hidden=f["hidden"], batch=f["batch"])
    rng = np.random.default_rng(seed + 11)
    sub = rng.choice(len(fqe_log["a"]), size=min(f["mcts_subsample"], len(fqe_log["a"])), replace=False)
    counts = np.zeros((disc.n_states, p.n_actions))
    np.add.at(counts, (train_log["s_discrete"], train_log["a"]), 1)
    mdp_fqe = FittedMDP(fqe_log["s_discrete"], fqe_log["a"], fqe_log["r"], fqe_log["s2_discrete"], disc.n_states, p.n_actions)
    err_df = model_error(mdp_fqe, fqe_log, p, disc, dm, seed=seed)
    rows, hists, extra = [], {}, []
    behavior_value = beh_true["truth"]
    for i, pol in enumerate(pool):
        proxy = pol
        fq_extra = np.nan
        if pol.family == "MCTS":
            Sx = np.vstack([fqe_log["s2_cont"][sub], fqe_log["s0"]])
            dist = distill_mcts(pol, Sx, p, seed)
            proxy = dist
            r_d = fqe_evaluate(dist.eval_probs, fqe_log, p.r_scale, p.gamma, f["seeds"], **fk)
            fq_extra = r_d["value"]
            res = fqe_evaluate(pol.eval_probs, fqe_log, p.r_scale, p.gamma, f["seeds"], sub_idx=sub, **fk)
        else:
            res = fqe_evaluate(pol.eval_probs, fqe_log, p.r_scale, p.gamma, f["seeds"], **fk)
        wis, ess = wis_estimate(proxy.probs, fqe_log, p.gamma, f["is_clip"])
        dr = dr_estimate(proxy.probs, res["runs"][0], fqe_log, p.gamma)
        ce = ce_ope_estimate(proxy, fqe_log, disc, p.gamma)
        ga = proxy.greedy(fqe_log["s_cont"])
        rows.append({"name": pol.name, "family": pol.family, **T[pol.name], "truth_soft": Ts[pol.name]["truth"],
                     "fqe": res["value"], "fqe_std": res["seed_std"], "fqe_lo": res["ci"][0], "fqe_hi": res["ci"][1],
                     "fqe_distilled": fq_extra, "self": pol.own_value, "ce_ope": ce, "wis": wis, "ess": ess, "dr": dr,
                     "support": support_fraction(proxy, fqe_log, counts, disc),
                     "model_tv": policy_model_error(disc(fqe_log["s_cont"]), ga, err_df, p.n_actions)})
        hists[pol.name] = res["history"]
        if verbose and (i % 8 == 0 or i == len(pool) - 1):
            say(f"  est {i + 1}/{len(pool)} ({time.time() - t0:.0f}s)")
    df = pd.DataFrame(rows)
    df["behavior"] = behavior_value + rng.normal(0, 1e-9, len(df))        # constant -> random ranking
    df["random"] = rng.random(len(df))
    # tie-aware ranking
    names = df["name"].tolist()
    tie = tie_matrix(rets, names)
    mrows = []
    for est in ESTIMATORS:
        m = metrics_for(df[est].to_numpy(), df["truth"].to_numpy(), tie)
        mrows.append({"estimator": est, "scope": "all", **m})
        for fam, g in df.groupby("family"):
            if len(g) >= 4:
                ix = g.index.to_numpy()
                m = metrics_for(g[est].to_numpy(), g["truth"].to_numpy(), tie[np.ix_(ix, ix)], ci=False)
                mrows.append({"estimator": est, "scope": fam, **m})
    mdf = pd.DataFrame(mrows)
    fam_err = pd.DataFrame([{"estimator": e, **r} for e in ["fqe", "self", "ce_ope", "wis", "dr"]
                            for r in family_metrics(df, e)])
    res_dir = out / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(res_dir / f"{name}_candidates.csv", index=False)
    mdf.to_csv(res_dir / f"{name}_metrics.csv", index=False)
    fam_err.to_csv(res_dir / f"{name}_family_error.csv", index=False)
    err_df.to_csv(res_dir / f"{name}_model_error.csv", index=False)
    (res_dir / f"{name}_fqe_history.json").write_text(json.dumps(hists), encoding="utf-8")
    (res_dir / f"{name}_meta.json").write_text(json.dumps({**meta, "behavior_true": beh_true, "S_star": S_star,
                                                          "S_star_train_value": v_star, "truth_horizon": hz,
                                                          "n_truth_episodes": nT, "n_truth_mcts": nM,
                                                          "runtime_s": time.time() - t0}, default=float), encoding="utf-8")
    say(f"  done {name} in {time.time() - t0:.0f}s")
    return {"df": df, "metrics": mdf, "name": name, "pool": pool if keep_pool else None}


def set_nested(cfg, path, value):
    d = cfg
    ks = path.split(".")
    for k in ks[:-1]:
        d = d[k]
    d[ks[-1]] = value


def ablation_grid(cfg):
    """(tag, overrides dict of dotted config paths, run_one kwargs)."""
    g = []
    for c in ["narrow", "medium", "wide", "mixture"]:
        g.append((f"cov_{c}", {}, {"cond": c}))
    for n in [150, 1500]:
        g.append((f"size_{n}", {}, {"n_episodes": n}))
    for h in [32, 128]:
        g.append((f"fqe_h{h}", {}, {"fqe_over": {"hidden": h}}))
    for r in [8, 60]:
        g.append((f"fqe_K{r}", {}, {"fqe_over": {"rounds": r}}))
    for npos, noh in [(6, 2), (24, 8)]:
        g.append((f"disc_{npos}x{noh}", {"discretize.n_pos_bins": npos, "discretize.n_oh_bins": noh}, {}))
    for L in [1, 5]:
        g.append((f"L{L}", {"env.lead_time": L}, {}))
    g.append(("gamma0.9", {"env.gamma": 0.9}, {}))
    g.append(("H45", {"env.horizon": 45}, {}))
    return g


def run_ablations(cfg, sid, quick):
    fams = ["ref", "CE", "DynaQ", "BC", "DQN", "CQL"]
    for tag, over, kw in ablation_grid(cfg):
        c = copy.deepcopy(cfg)
        c["fqe"]["seeds"] = 1
        for k, v in over.items():
            set_nested(c, k, v)
        run_one(c, sid, 0, quick=quick, families=fams, tag=tag, verbose=False, **kw)
        print("  ablation", tag, "done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--quick", action="store_true")
    g.add_argument("--full", action="store_true")
    ap.add_argument("--ablations", action="store_true")
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--series", default=None, help="restrict to 'primary' or a series id")
    a = ap.parse_args()
    cfg = load_config()
    print("seed", cfg["seed"])
    series = [cfg["series"]["primary"]] + ([] if a.quick else list(cfg["series"]["replication"]))
    if a.series == "primary":
        series = series[:1]
    reps = 1 if a.quick else (a.reps or cfg["experiment"]["repetitions"])
    if a.quick:
        cfg["logging"]["n_episodes"] = 60
        cfg["logging"]["n_val_episodes"] = 10
        cfg["fqe"].update(rounds=6, steps_per_round=40, seeds=2, mcts_subsample=60)
        cfg["pool"]["mcts_max_decisions"] = 300
    print(f"series={series} reps={reps} mode={'quick' if a.quick else 'full'}")
    for sid in series:
        for rep in range(reps):
            run_one(cfg, sid, rep, quick=a.quick)
    if a.ablations:
        run_ablations(cfg, cfg["series"]["primary"], a.quick)
    from report import make_report
    make_report(cfg)


if __name__ == "__main__":
    main()
