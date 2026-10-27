"""TRPO: natural-gradient step by conjugate gradient inside a KL trust region with a backtracking line search, GAE advantages and a
learned linear value baseline. Budget = number of training episodes (window rollouts) including the pilot batch.

Variants (same advantages, seeds and budgets):
  'trpo'      CG natural gradient + KL/surrogate-checked backtracking line search
  'vpg'       vanilla policy gradient with Adam (no trust region)
  'ng_fixed'  natural gradient with the fixed step beta_max = sqrt(2 delta / s^T F s), NO KL check and NO line search
"""
import time
from collections import deque
import numpy as np
from reinforce.optimizer import Adam
from .kl_fisher import family_of
from .conjugate_gradient import conjugate_gradient, mean_diag_fisher
from .gae import gae
from .value import ValueFn, returns_to_go


def policy_gradient(fam, theta, phi, act, A):
    """Gradient of the surrogate at theta_old: mean over samples of score * A (identical to the ordinary policy gradient)."""
    return (fam.score(theta, phi, act) * A[:, None]).mean(axis=0)


def surrogate(fam, theta_old, theta, phi, act, A, logp_old=None):
    lo = fam.logp(theta_old, phi, act) if logp_old is None else logp_old
    return float(np.mean(np.exp(fam.logp(theta, phi, act) - lo) * A))


def natural_direction(fvp, g, P, damp_frac=0.01, cg_iters=20, tol=1e-10, exact=False, diag_mean=None):
    """s = (F + d I)^{-1} g with d = damp_frac * mean diag(F). Returns s, d, residual history, s^T F s."""
    d = damp_frac * (mean_diag_fisher(fvp, P) if diag_mean is None else diag_mean)
    d = max(d, 1e-12)
    if exact or cg_iters is None:
        Fm = np.column_stack([fvp(e) for e in np.eye(P)])
        s = np.linalg.solve(0.5 * (Fm + Fm.T) + d * np.eye(P), g)
        hist = [0.0]
    else:
        s, hist = conjugate_gradient(fvp, g, cg_iters, tol, d)
    return s, d, hist, float(s @ fvp(s))


def line_search(fam, theta, phi, s, sFs, delta, surr, project, L0=0.0, max_bt=10, check=True, kl_tol=1e-9):
    """Backtracking line search theta = theta_old + 0.5^j beta_max s, j = 0..max_bt. Accept the first candidate with exact mean KL <= delta
    and surrogate improvement L(theta) > L0. Returns (theta_new or None, info)."""
    beta = np.sqrt(2.0 * delta / max(sFs, 1e-12))
    for j in range(max_bt + 1):
        cand = project(theta + (0.5 ** j) * beta * s)
        kl = fam.kl(theta, cand, phi)
        L = surr(cand)
        if not check or (kl <= delta + kl_tol and L > L0):
            return cand, {"kl": kl, "L": L, "nbt": j, "beta": beta, "accepted": True}
    return None, {"kl": 0.0, "L": L0, "nbt": max_bt + 1, "beta": beta, "accepted": False}


def val_J(pr, pol, theta, vstarts, mode, seed=12345):
    rng = np.random.default_rng(seed)
    return float(pr.run(pol, theta, vstarts, rng, mode)["total"].mean() / pr.norm)


def trpo(pr, pol, theta0, rng, episodes, variant="trpo", delta=0.01, lam=0.97, B=32, gamma=0.99, damp=0.01, cg_iters=20, std=True,
         lr=0.03, clip=5.0, max_bt=10, n_eval=40, diag=True, diag_mult=5, exact_every=25, keep_theta=False, hist_batches=4, ent_coef=0.0):
    """Returns dict(theta_best, theta_final, log (validation checkpoints), it_log (per iteration), best_val, secs, episodes_used).
    ent_coef > 0 adds ent_coef * mean per-state entropy to the surrogate (gradient, line-search objective and acceptance test)."""
    fam = family_of(pol)
    H = pr.H
    stride = 5 if pr.domain == "inventory" else 15
    vst = pr.grid("val", stride)
    theta = pol.project(theta0.copy())
    vf = ValueFn(H)
    iters = max(1, (episodes - B) // B)
    opt = Adam(len(theta), lr, clip, "cosine", iters) if variant == "vpg" else None
    cs = pr.sample("train", diag_mult * B, np.random.default_rng(int(rng.integers(2 ** 31)))) if diag else None      # fixed common-path set
    crng_seed = 4242
    ret = lambda th: float(pr.run(pol, th, cs, np.random.default_rng(crng_seed), "sample")["total"].mean() / pr.norm)

    used = 0
    t_start = time.time()
    # pilot batch: only to fit the value function (counted in the budget)
    st0 = pr.sample("train", B, rng)
    o = pr.run(pol, theta, st0, rng, "sample", record=True)
    used += B
    G0 = returns_to_go(o["rew"] / pr.norm, gamma)
    vf.fit([(o["phi"], G0)])
    batches = deque([(o["phi"], G0)], maxlen=hist_batches)

    every = max(1, iters // n_eval)
    log, it_log = [], []
    best, best_theta = -np.inf, theta.copy()
    ret_prev = ret(theta) if diag else np.nan
    cos_exact = np.nan
    for it in range(iters + 1):
        if it % every == 0 or it == iters:
            vd, vs = val_J(pr, pol, theta, vst, "det"), val_J(pr, pol, theta, vst, "sample")
            if not np.isfinite(vd):
                vd = -1e9
            log.append({"it": it, "episodes": used, "val_det": vd, "val_stoch": vs, "sigma": fam.sigma(theta), "theta": theta.tolist() if keep_theta else None})
            if vd > best:
                best, best_theta = vd, theta.copy()
        if it == iters:
            break
        tc0 = time.time()
        starts = pr.sample("train", B, rng)
        o = pr.run(pol, theta, starts, rng, "sample", record=True)
        used += B
        t_collect = time.time() - tc0
        tu0 = time.time()
        rew = o["rew"] / pr.norm
        phi3, act2 = o["phi"], o["act"]
        G = returns_to_go(rew, gamma)
        V = vf.predict(phi3)                                  # fitted BEFORE this batch was seen
        A, _ = gae(rew, V, vf.predict_terminal(phi3), gamma, lam)
        r2 = vf.r2(phi3, G)
        phi, act, Af = phi3.reshape(-1, phi3.shape[2]), act2.reshape(-1), A.reshape(-1)
        if std:
            Af = (Af - Af.mean()) / (Af.std() + 1e-8)
        g = policy_gradient(fam, theta, phi, act, Af)
        if ent_coef > 0:
            g = g + ent_coef * fam.ent_grad(theta, phi)
        gn = float(np.linalg.norm(g))
        logp_old = fam.logp(theta, phi, act)
        bonus = (lambda th: ent_coef * fam.ent_value(th, phi)) if ent_coef > 0 else (lambda th: 0.0)
        L0 = float(np.mean(Af)) + bonus(theta)
        info = {"kl": 0.0, "L": L0, "nbt": 0, "accepted": False, "beta": np.nan}
        hist = [np.nan]
        theta_new = theta
        if not np.all(np.isfinite(g)) or gn < 1e-14:
            pass
        elif variant == "vpg":
            theta_new, _ = opt.step(theta, g)
            theta_new = pol.project(theta_new)
            info = {"kl": fam.kl(theta, theta_new, phi), "L": surrogate(fam, theta, theta_new, phi, act, Af, logp_old) + bonus(theta_new), "nbt": 0, "accepted": True, "beta": np.nan}
        else:
            fvp = lambda v: fam.fvp(theta, phi, v)
            dm = float(fam.diag(theta, phi).mean())
            s, d, hist, sFs = natural_direction(fvp, g, fam.P, damp, cg_iters, exact=(cg_iters is None), diag_mean=dm)
            if it % exact_every == 0 and fam.P <= 130:
                s_ex = natural_direction(fvp, g, fam.P, damp, None, exact=True, diag_mean=dm)[0]
                cos_exact = float(s @ s_ex / (np.linalg.norm(s) * np.linalg.norm(s_ex) + 1e-300))
            surr = lambda c: surrogate(fam, theta, c, phi, act, Af, logp_old) + bonus(c)
            cand, info = line_search(fam, theta, phi, s, sFs, delta, surr, pol.project, L0, max_bt, check=(variant == "trpo"))
            if cand is not None:
                theta_new = cand
                if variant == "ng_fixed":
                    info["kl"] = fam.kl(theta, theta_new, phi)
        t_update = time.time() - tu0
        theta = theta_new
        batches.append((phi3, G))
        vf.fit(list(batches))                                 # refit on the current batch plus the last 3
        if not np.all(np.isfinite(theta)):
            theta = best_theta.copy()
            it_log.append({"it": it, "episodes": used, "diverged": True})
            break
        ret_after = ret(theta) if diag else np.nan
        it_log.append({"it": it, "episodes": used, "surr_improve": info["L"] - L0, "kl": info["kl"], "nbt": info["nbt"], "accepted": bool(info["accepted"]),
                       "cg_res": hist[-1] / (hist[0] + 1e-300) if len(hist) > 1 else np.nan, "cg_iters_used": len(hist) - 1, "gnorm": gn, "vf_r2": r2,
                       "sigma": fam.sigma(theta), "entropy": fam.entropy(theta, phi), "ret_before": ret_prev, "ret_after": ret_after, "cos_exact": cos_exact,
                       "t_collect": t_collect, "t_update": t_update})
        cos_exact = np.nan
        ret_prev = ret_after
    return {"theta_best": best_theta, "theta_final": theta, "log": log, "it_log": it_log, "best_val": best, "secs": time.time() - t_start, "episodes_used": used}
