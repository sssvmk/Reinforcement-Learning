"""Estimation engine: policy arrays on logs, hyper-parameter selection (data only), one repetition of OPE for all
agents and all estimators, and the resulting per-agent records."""
from __future__ import annotations
import time
import numpy as np
from estimators import (TD, GTD2, TDC, ETD, run_many, make_orders, batch_AbC, mspbe, lstd, emphatic_stats, wis, fqe,
                        oracle_projection)

METHOD_ID = {"TD": TD, "GTD2": GTD2, "TDC": TDC, "ETD": ETD}


def policy_arrays(domain, agents, log, eps_t):
    """Per agent: rho (W,H), P at next states (W,H,nA), P at window starts (W,nA)."""
    W, H1, k = log.raw.shape
    H = H1 - 1
    nA = domain.n_actions
    flat = log.raw.reshape(-1, k)
    rho = np.zeros((len(agents), W, H))
    Pn = np.zeros((len(agents), W, H, nA))
    P0 = np.zeros((len(agents), W, nA))
    for i, ag in enumerate(agents):
        g = ag.act(flat).reshape(W, H1)
        pi_taken = (1 - eps_t) * (g[:, :-1] == log.act) + eps_t / nA
        rho[i] = pi_taken / log.bprob
        for a in range(nA):
            Pn[i, :, :, a] = (1 - eps_t) * (g[:, 1:] == a) + eps_t / nA
            P0[i, :, a] = (1 - eps_t) * (g[:, 0] == a) + eps_t / nA
    return rho, Pn, P0


def _tasks(n_agents, specs):
    agent_idx = np.repeat(np.arange(n_agents), len(specs))
    sp = specs * n_agents
    f = lambda key, dt: np.array([s[key] for s in sp], dtype=dt)
    return (agent_idx, f("method", np.int64), f("alpha", np.float64), f("kappa", np.float64), f("lam", np.float64),
            f("vform", np.int64), f("reset", np.bool_), f("clip", np.float64))


def fit(Xs, rho_all, R, specs, gamma, epochs, n_ckpt, order_seed, clipF=0.0, decay_p=0.0, decay_k0=20000.0):
    """Run every (agent, spec) task in parallel. Returns dict of arrays with first axis = agent*len(specs)+spec."""
    A = rho_all.shape[0]
    order = make_orders(Xs.shape[0], epochs, np.random.default_rng(order_seed))
    specs = [{"clip": clipF, **s} for s in specs]
    ai, me, al, ka, la, vf, rs, cl = _tasks(A, specs)
    t0 = time.time()
    snaps, wf, wt, div, nup = run_many(Xs, rho_all, R, order, ai, me, al, ka, la, vf, rs, cl,
                                       float(decay_p), float(decay_k0), int(n_ckpt), float(gamma))
    return dict(snaps=snaps, w=wf, w_tail=wt, diverged=div, n_updates=nup, agent=ai, secs=time.time() - t0)


# ------------------------------------------------------------------ hyper-parameter selection (data only)
def grid_specs(cfg_e, method):
    e = cfg_e
    out = []
    if method == "TD":
        out = [dict(method=TD, alpha=a, kappa=0.0, lam=0.0, vform=0, reset=True) for a in e["alpha_grid"]]
    elif method in ("GTD2", "TDC"):
        out = [dict(method=METHOD_ID[method], alpha=a, kappa=k, lam=0.0, vform=0, reset=True)
               for a in e["alpha_grid"] for k in e["kappa_grid"]]
    elif method == "ETD":
        out = [dict(method=ETD, alpha=a, kappa=0.0, lam=l, vform=0, reset=True)
               for a in e["alpha_grid"] for l in e["lambda_grid"]]
    return out


def tune_hparams(domain, agents, cfg, n_windows, eps_b, eps_t, seed, epochs, schedule=None, log_fn=print):
    """Choose (alpha, kappa, lambda) per method by minimizing the method's own data-based objective on the
    TUNING-log (separate validation windows): MSPBE (TD, GTD2, TDC) or emphatic PBE (ETD), normalised by its value
    at w=0 and aggregated by the median over agents. Ground truth is never used."""
    e = cfg["estimators"]
    gamma, scale = domain.gamma, domain.reward_scale
    rng = np.random.default_rng(seed)
    train = domain.sample_log("ope", n_windows, eps_b, rng)
    tune = domain.sample_log("tune", max(n_windows // 2, 40), eps_b, rng)
    Xtr, Xtu = train.phi(domain), tune.phi(domain)
    rho_tr, _, _ = policy_arrays(domain, agents, train, eps_t)
    rho_tu, _, _ = policy_arrays(domain, agents, tune, eps_t)
    Rtr = train.r * scale
    Rtu = tune.r * scale
    ABC = [batch_AbC(Xtu, rho_tu[i], Rtu, gamma, ridge=1e-6) for i in range(len(agents))]
    EABC = []
    for i in range(len(agents)):
        F = emphatic_stats(rho_tu[i], gamma, True, 0.0)
        EABC.append(batch_AbC(Xtu, rho_tu[i], Rtu, gamma, ridge=1e-6, weights=F))
    dp, dk = (schedule or (0.0, e["decay_k0"]))
    chosen, table = {}, {}
    for method in ("TD", "GTD2", "TDC", "ETD"):
        specs = grid_specs(e, method)
        res = fit(Xtr, rho_tr, Rtr, specs, gamma, epochs, 4, seed + 11, decay_p=dp, decay_k0=dk)
        n = len(specs)
        ratios = np.full((len(agents), n), np.inf)
        for kk in range(len(agents) * n):
            i, s = divmod(kk, n)
            if res["diverged"][kk] or not np.all(np.isfinite(res["w_tail"][kk])):
                continue
            A, b, C = (EABC if method == "ETD" else ABC)[i]
            base = mspbe(np.zeros(len(b)), A, b, C)
            ratios[i, s] = mspbe(res["w_tail"][kk], A, b, C) / max(base, 1e-30)
        score = np.nanmedian(np.where(np.isfinite(ratios), ratios, 1e6), axis=0)
        best = int(np.argmin(score))
        chosen[method] = specs[best]
        table[method] = [(dict(alpha=s["alpha"], kappa=s["kappa"], lam=s["lam"]), float(sc))
                         for s, sc in zip(specs, score)]
        log_fn(f"  tuned {method}: alpha={specs[best]['alpha']} kappa={specs[best]['kappa']} "
               f"lam={specs[best]['lam']} score={score[best]:.4g}")
    return chosen, table


# ------------------------------------------------------------------ one repetition
def diagnostics(rho, gamma, lam):
    A = rho.shape[0]
    rows = []
    for i in range(A):
        r = rho[i]
        F = emphatic_stats(r, gamma, True, 0.0)
        M = lam + (1 - lam) * F
        rows.append(dict(rho_max=float(r.max()), rho_mean=float(r.mean()),
                         ess_rho=float(r.sum() ** 2 / (r ** 2).sum()),
                         gamma2_Erho2=float(gamma ** 2 * (r ** 2).mean()),
                         F_mean=float(F.mean()), F_var=float(F.var()), F_max=float(F.max()), M_var=float(M.var())))
    return rows


def run_rep(domain, agents, chosen, cond, cfg, seed, truth_cache=None):
    """One repetition: fresh OPE-log (and fresh test windows under protocol B), all estimators, all agents."""
    e = cfg["estimators"]
    gamma, scale, H = domain.gamma, domain.reward_scale, domain.H
    eps_b, eps_t = cond["eps_b"], cond["eps_t"]
    dense = cond.get("dense", False)
    rng = np.random.default_rng(seed)
    log = domain.sample_log("ope", cond["n_windows"], eps_b, rng)
    protocol = cond.get("protocol", "A")
    if protocol == "A":
        evlog, kind = log, "ope"
    else:
        evlog, kind = domain.sample_log("test", cond["n_windows"], eps_b, rng), "test"
    S = e["n_state_samples"]
    sidx = np.stack([rng.integers(0, evlog.W, S), rng.integers(0, getattr(domain, "state_step_max", evlog.H), S)], axis=1)
    Jw, vs = domain.truth(agents, evlog, eps_t, kind, sidx)
    Jtrue = Jw.mean(axis=1)
    Xs = log.phi(domain, dense)
    Xev = evlog.phi(domain, dense)
    x0 = Xev[:, 0].mean(axis=0)
    Xst = Xev[sidx[:, 0], sidx[:, 1]]
    rho_all, Pn, P0 = policy_arrays(domain, agents, log, eps_t)
    _, _, P0ev = policy_arrays(domain, agents, evlog, eps_t) if protocol == "B" else (None, None, P0)
    R = log.r * scale
    A = len(agents)
    dp, dk = cond.get("schedule", (0.0, e["decay_k0"]))
    epochs = cond.get("epochs", e["epochs"])
    methods = cond.get("methods", ["TD", "GTD2", "TDC", "ETD", "ETD_noreset", "ETD_clip"])
    kappa_override = cond.get("kappa")
    lam_override = cond.get("lam")
    vform = cond.get("vform", 0)
    specs, names = [], []
    for m in methods:
        base = chosen["ETD" if m.startswith("ETD") else m.split("@")[0]]
        s = dict(base)
        if m == "ETD_noreset":
            s["reset"] = False
        if m == "ETD_clip":
            s["clip"] = cond.get("etd_clip", 50.0)
        if kappa_override is not None and s["method"] in (GTD2, TDC):
            s["kappa"] = kappa_override
        if lam_override is not None and s["method"] == ETD:
            s["lam"] = lam_override
        if s["method"] in (GTD2, TDC):
            s["vform"] = vform
        specs.append(s)
        names.append(m)
    res = fit(Xs, rho_all, R, specs, gamma, epochs, e["n_ckpt"], seed + 5, decay_p=dp, decay_k0=dk,
              clipF=cond.get("clipF", 0.0))
    nS = len(specs)
    # batch quantities from the OPE-log for the MSPBE curves
    ABC, EABC = [], []
    for i in range(A):
        ABC.append(batch_AbC(Xs, rho_all[i], R, gamma))
        F = emphatic_stats(rho_all[i], gamma, True, 0.0)
        EABC.append(batch_AbC(Xs, rho_all[i], R, gamma, weights=F))
    diag = diagnostics(rho_all, gamma, specs[names.index("ETD")]["lam"] if "ETD" in names else 0.0)
    rows, curves = [], {}
    for kk in range(A * nS):
        i, s = divmod(kk, nS)
        m = names[s]
        ag = agents[i]
        dv = bool(res["diverged"][kk])
        row = dict(agent=ag.name, family=ag.family, method=m, J_true=float(Jtrue[i]), diverged=dv,
                   alpha=specs[s]["alpha"], kappa=specs[s]["kappa"], lam=specs[s]["lam"])
        if not dv:
            for tag, w in (("final", res["w"][kk]), ("tail", res["w_tail"][kk])):
                row[f"J_{tag}"] = float(w @ x0) / scale
                row[f"msve_{tag}"] = float(np.mean((Xst @ w / scale - vs[i]) ** 2))
            Ab, bb, Cb = ABC[i]
            row["mspbe_tail"] = mspbe(res["w_tail"][kk], Ab, bb, Cb) / scale ** 2
            Ae, be, Ce = EABC[i]
            row["epbe_tail"] = mspbe(res["w_tail"][kk], Ae, be, Ce) / scale ** 2
            sn = res["snaps"][kk]
            ok = np.isfinite(sn).all(axis=1)
            cv = dict(J=np.where(ok, sn @ x0 / scale, np.nan),
                      msve=np.array([np.mean((Xst @ sn[c] / scale - vs[i]) ** 2) if ok[c] else np.nan
                                     for c in range(len(sn))]),
                      mspbe=np.array([mspbe(sn[c], Ab, bb, Cb) / scale ** 2 if ok[c] else np.nan
                                      for c in range(len(sn))]))
            curves[(ag.name, m)] = cv
        else:
            row.update(J_final=np.nan, J_tail=np.nan, msve_final=np.nan, msve_tail=np.nan, mspbe_tail=np.nan,
                       epbe_tail=np.nan)
        row["n_updates"] = int(res["n_updates"][kk])
        rows.append(row)
    # ---- baselines
    w_or = None
    for i, ag in enumerate(agents):
        base = dict(agent=ag.name, family=ag.family, J_true=float(Jtrue[i]), diverged=False, alpha=np.nan,
                    kappa=np.nan, lam=np.nan, n_updates=0)
        wl = lstd(Xs, rho_all[i], R, gamma, e["lstd_ridge"])
        rows.append({**base, "method": "LSTD", "J_final": float(wl @ x0) / scale,
                     "msve_final": float(np.mean((Xst @ wl / scale - vs[i]) ** 2))})
        Jf, _ = fqe(Xs, log.act, R, Pn[i], P0ev[i] if protocol == "B" else P0[i], gamma, e["fqe_rounds"],
                    e["fqe_ridge"], domain.n_actions, X0=Xev[:, 0])
        rows.append({**base, "method": "FQE", "J_final": Jf / scale, "msve_final": np.nan})
        Jwis, ess = wis(log.r, np.log(rho_all[i]), gamma, e["wis_clip"])
        rows.append({**base, "method": "WIS", "J_final": Jwis, "msve_final": np.nan, "ess": ess})
        wo = oracle_projection(Xst, vs[i])
        rows.append({**base, "method": "OracleProj", "J_final": float(wo @ x0), "msve_final": float(np.mean((Xst @ wo - vs[i]) ** 2))})
        bv = domain.behavior_value(evlog, eps_b, kind)
        rows.append({**base, "method": "BehaviorValue", "J_final": bv, "msve_final": np.nan})
        rows.append({**base, "method": "Random", "J_final": float(rng.random()), "msve_final": np.nan})
    return dict(rows=rows, curves=curves, diag=diag, agents=[a.name for a in agents], Jw=Jw, Jtrue=Jtrue,
                secs_fit=res["secs"], start=evlog.start, n_updates_per_run=int(res["n_updates"][0]))
