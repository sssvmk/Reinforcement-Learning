"""Equal-budget derivative-free / finite-difference comparison methods on the mean-policy parameters (budget = rollouts).
The evaluated policy is the 'mean' mode of the stochastic family (Gaussian mean; for R-T1 the continuous position p = FDPG's F-T1 form)."""
import numpy as np
from scipy.optimize import minimize
from .optimizer import Adam


class Tracker:
    """Counts rollouts and records validation J of the current iterate at regular rollout intervals."""

    def __init__(self, pr, pol, full_theta, budget, n_eval=40, stride=None):
        self.pr, self.pol, self.full, self.budget = pr, pol, full_theta.copy(), budget
        self.d = pol.mean_dim
        self.vst = pr.grid("val", stride or (5 if pr.domain == "inventory" else 15))
        self.every, self.used, self.next = budget / n_eval, 0, 0.0
        self.log, self.best, self.best_theta = [], -np.inf, None

    def th(self, x):
        t = self.full.copy()
        t[:self.d] = x
        return t

    def f(self, x, starts):
        """Train objective on given windows, counts rollouts."""
        self.used += len(starts)
        return self.pr.J(self.pol, self.th(x), starts, None, "mean")

    def checkpoint(self, x, force=False):
        if self.used >= self.next or force:
            v = self.pr.J(self.pol, self.th(x), self.vst, None, "mean")
            if not np.isfinite(v):
                v = -1e9
            self.log.append({"episodes": int(self.used), "val_det": v, "theta": self.th(x).tolist()})
            if v > self.best:
                self.best, self.best_theta = v, self.th(x)
            self.next = self.used + self.every

    def done(self):
        return self.used >= self.budget

    def result(self):
        return {"theta_best": self.best_theta, "log": self.log, "best_val": self.best}


def _clip(pol, x):
    return np.clip(x, pol.lo, pol.hi)


def random_search(pr, pol, theta0, rng, budget, B=32, step=0.3, decay=0.999):
    tr = Tracker(pr, pol, theta0, budget)
    x, s = _clip(pol, theta0[:tr.d].copy()), step
    tr.checkpoint(x, True)
    while not tr.done():
        starts = pr.sample("train", B, rng)
        cand = _clip(pol, x + s * rng.standard_normal(tr.d))
        if tr.f(cand, starts) > tr.f(x, starts):
            x = cand
        s *= decay
        tr.checkpoint(x)
    tr.checkpoint(x, True)
    return tr.result()


def nelder_mead(pr, pol, theta0, rng, budget, B=32, **_):
    tr = Tracker(pr, pol, theta0, budget)
    starts = pr.sample("train", B, rng)          # fixed windows: Nelder-Mead needs a deterministic objective
    x0 = _clip(pol, theta0[:tr.d].copy())

    class Stop(Exception):
        pass

    last = {"x": x0}

    def obj(x):
        if tr.done():
            raise Stop
        x = _clip(pol, x)
        v = tr.f(x, starts)
        last["x"] = x
        tr.checkpoint(x)
        return -v if np.isfinite(v) else 1e9

    tr.checkpoint(x0, True)
    try:
        minimize(obj, x0, method="Nelder-Mead", options={"maxfev": budget // B + 10, "xatol": 1e-4, "fatol": 1e-6, "initial_simplex": None})
    except Stop:
        pass
    tr.checkpoint(last["x"], True)
    return tr.result()


def spsa(pr, pol, theta0, rng, budget, B=32, a=0.1, c=0.1, A=20, crn=True):
    tr = Tracker(pr, pol, theta0, budget)
    x = _clip(pol, theta0[:tr.d].copy())
    tr.checkpoint(x, True)
    k = 0
    while not tr.done():
        k += 1
        ak, ck = a / (k + A) ** 0.602, c / k ** 0.101
        delta = rng.choice([-1.0, 1.0], tr.d)
        sp = pr.sample("train", B, rng)
        sm = sp if crn else pr.sample("train", B, rng)
        g = (tr.f(_clip(pol, x + ck * delta), sp) - tr.f(_clip(pol, x - ck * delta), sm)) / (2 * ck) / delta
        x = _clip(pol, x + ak * g * 1.0)
        tr.checkpoint(x)
    tr.checkpoint(x, True)
    return tr.result()


def central_fd(pr, pol, theta0, rng, budget, B=32, eps=0.05, lr=0.05, clip=5.0):
    tr = Tracker(pr, pol, theta0, budget)
    x = _clip(pol, theta0[:tr.d].copy())
    opt = Adam(tr.d, lr, clip, "const")
    tr.checkpoint(x, True)
    while not tr.done():
        starts = pr.sample("train", B, rng)       # common paths for all 2d evaluations
        g = np.zeros(tr.d)
        for i in range(tr.d):
            e = np.zeros(tr.d)
            e[i] = eps
            g[i] = (tr.f(x + e, starts) - tr.f(x - e, starts)) / (2 * eps)
        x, _ = opt.step(x, g)
        x = _clip(pol, x)
        tr.checkpoint(x)
    tr.checkpoint(x, True)
    return tr.result()


METHODS = {"random": random_search, "neldermead": nelder_mead, "spsa": spsa, "central": central_fd}
GRIDS = {"random": [{"step": s} for s in (0.1, 0.3, 1.0)], "neldermead": [{}],
         "spsa": [{"a": a, "c": c} for a in (0.03, 0.1, 0.3) for c in (0.05, 0.2)],
         "central": [{"eps": e, "lr": l} for e in (0.02, 0.1) for l in (0.02, 0.1)]}
