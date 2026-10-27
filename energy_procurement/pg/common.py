"""Training scaffold shared by all algorithms: logging, validation, checkpoints, plots, final reports."""
import copy
import json
import os
import time
import numpy as np
import pandas as pd
import torch
from pg.env import DayEnv, Risk, reference_schedule, distill_theta
from pg import policies, training_plots
from risk import cvar, hedge_ratio

torch.set_num_threads(1)


def make_risk(objective, cfg):
    return Risk(objective, lam=cfg.get("lam", 1.0), alpha=cfg.get("alpha", 0.95), kappa=cfg.get("kappa", 8.0),
                lr_eta=cfg.get("lr_eta", 0.02), ema=cfg.get("ema", 0.05), eta0=cfg.get("eta0", 1.5))


def collect(env, policy, idx, risk, sample=True):
    """Stochastic rollout; terminal risk penalty added to the last step reward."""
    with torch.no_grad():
        r = env.rollout(idx, policy.sample if sample else policy.mean)
    pen = risk.penalty(r["Ct"])
    R = r["R"].clone()
    R[:, -1] -= torch.tensor(pen, dtype=torch.float32)
    r["Radj"], r["pen"] = R, pen
    return r


def batch_stats(r, env):
    q, Lh, cap = r["q"], env.Lhat[r["idx"]], env.cap[r["idx"]]
    Ct = r["Ct"]
    return {"G_mean": float((-Ct - r["pen"]).mean()), "Ct_mean": float(Ct.mean()), "Ct_std": float(Ct.std()),
            "batch_cvar95": float(cvar(Ct, 0.95)), "batch_var": float(Ct.var()),
            "cap_share": float((q >= cap - 1e-9).mean()), "zero_share": float((q <= 1e-9).mean()),
            "H_batch": float(q.sum() / Lh.sum())}


class Algo:
    """Base class: subclasses set self.policy and implement iterate()."""
    name = "base"

    def __init__(self, env, risk, cfg, seed):
        self.env, self.risk, self.cfg, self.seed = env, risk, cfg, seed
        self.rng = np.random.default_rng(seed)
        self.train_idx = env.P.idx("train")
        self.policy = policies.GaussianPolicy(env.n_in, cfg.get("policy", "linear"), cfg.get("init_sigma", 0.03))
        self.theta0 = None
        self.avg = copy.deepcopy(self.policy)
        self.t_avg = 0

    def days_per_iter(self):
        return self.cfg.get("B", 32)

    def warm_start(self, theta):
        if self.cfg.get("policy", "linear") != "linear":
            return
        w = torch.tensor(theta, dtype=torch.float32)
        for pol in (self.policy, self.avg):
            pol.mu.weight.data.copy_(w[:-1][None])
            pol.mu.bias.data.copy_(w[-1:])

    def sample_days(self, n, replace=False):
        return self.rng.choice(self.train_idx, n, replace=replace or n > len(self.train_idx))

    def update_avg(self):
        """Polyak (EMA) average of the policy; the averaged policy is what gets evaluated."""
        self.t_avg += 1
        d = min(self.cfg.get("avg_decay", 0.95), (1 + self.t_avg) / (10 + self.t_avg))
        with torch.no_grad():
            for pa, p in zip(self.avg.parameters(), self.policy.parameters()):
                pa.mul_(d).add_((1 - d) * p)

    def mean_fn(self):
        return self.avg.mean

    def theta(self):
        return self.policy.theta()

    def snapshot(self):
        return {"p": copy.deepcopy(self.policy.state_dict()), "a": copy.deepcopy(self.avg.state_dict())}

    def restore(self, st):
        self.policy.load_state_dict(st["p"])
        self.avg.load_state_dict(st["a"])

    def iterate(self):
        raise NotImplementedError


def evaluate_split(algo, split, risk_eval, ref, tso, state=None):
    env = algo.env
    P = env.P
    idx = P.idx(split)
    if state is not None:
        cur = algo.snapshot()
        algo.restore(state)
    Ct, q, a = env.eval_mean(algo.mean_fn(), idx)
    Creal = env.last_real if env.cond is not None else Ct
    if state is not None:
        algo.restore(cur)
    obj = risk_eval.objective(Ct)
    j_ref, j_tso = ref[split]["obj"], tso[split]["obj"]
    skill = (j_tso - obj) / (j_tso - j_ref) if abs(j_tso - j_ref) > 1e-12 else np.nan
    return {"obj": float(obj), "mean_Ct": float(Creal.mean()), "cvar95": float(cvar(Creal, .95)), "std": float(Creal.std()),
            "mean_eur": float((Creal * P.Cref).mean()), "skill": float(skill),
            "gap_rel": float((obj - j_ref) / j_ref), "H": float(q.sum() / P.Lhat[idx].sum()),
            "act_by_hour": a.mean(0), "qshare_by_hour": (q / P.Lhat[idx]).mean(0), "Ct": Creal}


def references(env, objective, risk_eval, cfg, tag=""):
    P = env.P
    env.set_cond(risk_eval)
    out_ref, out_tso = {}, {}
    for split in ("val", "test"):
        idx = P.idx(split)
        qr = reference_schedule(split, objective, cfg, tag)
        if risk_eval.cond:
            Cr, Ct = env.cond_value(qr, idx), env.cond_value(P.Lhat[idx], idx)
        else:
            Cr, Ct = P.ctilde(qr, idx), P.ctilde(P.Lhat[idx], idx)
        out_ref[split] = {"obj": risk_eval.objective(Cr), "Ct": Cr}
        out_tso[split] = {"obj": risk_eval.objective(Ct), "Ct": Ct}
    return out_ref, out_tso


def run_training(algo_cls, P, fc, objective, seed, cfg, out_dir=None, quiet=True, env=None, refs=None, theta_dist=None):
    """Train one (algorithm, objective, seed). Returns dict with log DataFrame and final results."""
    t_start = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = env or DayEnv(P, fc, use_fc=cfg.get("use_fc", True))
    risk = make_risk(objective, cfg)
    risk_eval = make_risk(objective, cfg)
    env.set_cond(risk_eval)
    ref, tso = refs or references(env, objective, risk_eval, cfg)
    if theta_dist is None:
        theta_dist = distill_theta(env, P, fc)
    algo = algo_cls(env, risk, cfg, seed)
    if cfg.get("warm_start", False):
        algo.warm_start(theta_dist)
    if "budget_days" in cfg:
        cfg = {**cfg, "n_iter": int(np.ceil(cfg["budget_days"] / algo.days_per_iter()))}
        algo.cfg = cfg
    n_iter = cfg["n_iter"]
    eval_every = cfg.get("eval_every", max(1, n_iter // 40))
    plot_every = cfg.get("plot_every", max(1, n_iter // 4))
    log, hist_a, hist_q = [], [], []
    days, best = 0, (np.inf, None, -1)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    for it in range(n_iter + 1):
        if it > 0:
            t0 = time.time()
            m = algo.iterate()
            algo.update_avg()
            m["iter_time"] = time.time() - t0
            days += m.pop("days")
        else:
            m = {}
        row = {"iter": it, "train_days": days, **m}
        if it % eval_every == 0 or it == n_iter:
            v = evaluate_split(algo, "val", risk_eval, ref, tso)
            row.update({"val_obj": v["obj"], "val_Ct": v["mean_Ct"], "val_cvar95": v["cvar95"], "val_mean_eur": v["mean_eur"],
                        "val_skill": v["skill"], "val_gap_rel": v["gap_rel"], "H_val": v["H"]})
            hist_a.append(v["act_by_hour"]); hist_q.append(v["qshare_by_hour"])
            if v["obj"] < best[0] and np.isfinite(v["obj"]):
                best = (v["obj"], algo.snapshot(), it)
        row["theta_dist"] = float(np.linalg.norm(algo.theta() - theta_dist[:len(algo.theta())])) if len(algo.theta()) == len(theta_dist) else np.nan
        row["sigma"] = float(algo.policy.sigma.detach())
        row["eta"], row["m"] = risk.eta, risk.m
        row["wall"] = time.time() - t_start
        log.append(row)
        risk_state = (risk.eta, risk.m)
        if out_dir and (it % plot_every == 0 or it == n_iter):
            df = pd.DataFrame(log)
            df.to_csv(os.path.join(out_dir, "log.csv"), index=False)
            training_plots.plot_run(df, np.array(hist_a), np.array(hist_q), out_dir, f"{algo.name}/{objective}/seed_{seed}")
    df = pd.DataFrame(log)
    final = {}
    for lbl, st in (("final", None), ("best_val", best[1])):
        for split in ("val", "test"):
            e = evaluate_split(algo, split, risk_eval, ref, tso, state=st)
            final[f"{lbl}_{split}"] = {k: v for k, v in e.items() if k not in ("act_by_hour", "qshare_by_hour", "Ct")}
            final[f"{lbl}_{split}"]["act_by_hour"] = e["act_by_hour"].tolist()
    skill_curve = df.dropna(subset=["val_skill"])
    def first_reach(th):
        ok = skill_curve[skill_curve.val_skill >= th]
        return float(ok.train_days.iloc[0]) if len(ok) else np.nan
    sigma_collapse = bool(df.sigma.iloc[-1] <= 0.0101)
    vs = skill_curve.val_skill.to_numpy()
    final.update({"days_to_95": first_reach(0.95), "days_to_99": first_reach(0.99), "best_iter": best[2],
                  "total_days": int(days), "wall_s": time.time() - t_start,
                  "worst_drop": float(np.nanmax(np.maximum(-np.diff(vs), 0))) if len(vs) > 1 else 0.0,
                  "diverged": bool(not np.isfinite(vs[-1]) or vs[-1] < -1.0), "sigma_collapsed": sigma_collapse,
                  "eta_final": risk.eta, "m_final": risk.m, "algo": algo.name, "objective": objective, "seed": seed})
    if out_dir:
        json.dump(final, open(os.path.join(out_dir, "final.json"), "w"), indent=1, default=float)
        df.to_csv(os.path.join(out_dir, "log.csv"), index=False)
    return {"log": df, "final": final, "algo": algo}
