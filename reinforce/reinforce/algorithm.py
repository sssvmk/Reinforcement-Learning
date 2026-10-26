"""REINFORCE (Monte Carlo policy gradient) with baselines. Budget = number of training episodes (windows)."""
import numpy as np
from .baselines import rewards_to_go, advantages, BaselineState
from .optimizer import Adam


def grad_estimate(pr, pol, theta, starts, rng, baseline="loo_time", gamma=1.0, ent_coef=0.0, bstate=None, discount_weight=False):
    """Gradient of J (in normalised reward units): (1/B) sum_i sum_t score_it * (G_it - b_it) [+ entropy bonus].
    The gamma^t weighting of the exact discounted gradient is omitted by default (common practice; flag `discount_weight` restores it)."""
    o = pr.run(pol, theta, starts, rng, "sample", record=True)
    rew = o["rew"] / pr.norm
    G = rewards_to_go(rew, gamma)
    A = advantages(baseline, G, o["phi"], bstate, pr.H) if bstate is not None or baseline != "linear" else G
    w = gamma ** np.arange(pr.H)[None, :, None] if discount_weight else 1.0
    g = (o["score"] * A[:, :, None] * w).sum(axis=1).mean(axis=0)
    if ent_coef:
        g = g + ent_coef * o["ent"].mean(axis=0)
    return g, o, G


def val_J(pr, pol, theta, vstarts, mode, seed=12345):
    rng = np.random.default_rng(seed)
    return float(pr.run(pol, theta, vstarts, rng, mode)["total"].mean() / pr.norm)


def reinforce(pr, pol, theta0, rng, episodes, lr=0.05, B=32, ent_coef=0.0, baseline="loo_time", gamma=0.99, clip=5.0, schedule="cosine",
              n_eval=40, val_stride=None, discount_weight=False, keep_theta=True):
    """Train; returns dict(theta_best, log, theta_final). log rows: it, episodes, val_det, val_stoch, train_J, sigma, gnorm, theta."""
    stride = val_stride or (5 if pr.domain == "inventory" else 15)
    vst = pr.grid("val", stride)
    iters = max(1, episodes // B)
    opt = Adam(len(theta0), lr, clip, schedule, iters)
    theta = pol.project(theta0.copy())
    bst = BaselineState() if baseline == "linear" else None
    every = max(1, iters // n_eval)
    log, best, best_theta = [], -np.inf, theta.copy()
    last_tj = last_gn = np.nan
    for it in range(iters + 1):
        if it % every == 0 or it == iters:
            vd, vs = val_J(pr, pol, theta, vst, "det"), val_J(pr, pol, theta, vst, "sample")
            if not np.isfinite(vd):
                vd = -1e9
            log.append({"it": it, "episodes": it * B, "val_det": vd, "val_stoch": vs,             "train_J": last_tj, "sigma": pol.entropy_proxy(theta), "gnorm": last_gn, "theta": theta.tolist() if keep_theta else None})
            if vd > best:
                best, best_theta = vd, theta.copy()
        if it == iters:
            break
        starts = pr.sample("train", B, rng)
        g, o, G = grad_estimate(pr, pol, theta, starts, rng, baseline, gamma, ent_coef, bst, discount_weight)
        if baseline == "linear":
            bst.fit(o["phi"], G, pr.H)
        theta, gn = opt.step(theta, g)
        theta = pol.project(theta)
        last_tj, last_gn = float(o["total"].mean() / pr.norm), gn
        if not np.all(np.isfinite(theta)):
            theta = best_theta.copy()
            log.append({"it": it + 1, "episodes": (it + 1) * B, "val_det": -1e9, "val_stoch": -1e9, "train_J": np.nan, "sigma": np.nan, "gnorm": np.nan, "theta": None})
            break
    return {"theta_best": best_theta, "theta_final": theta, "log": log, "best_val": best}
