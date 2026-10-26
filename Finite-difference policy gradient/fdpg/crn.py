"""Objective wrapper with rollout accounting, and common-random-number (CRN) path provider."""
import numpy as np


class PathSampler:
    """Uniform window starts in [lo, hi - H] (window uses indices start..start+H, all < hi)."""

    def __init__(self, lo, hi, H, valid=None):
        self.lo, self.hi, self.H = int(lo), int(hi), int(H)
        assert self.hi - self.lo > self.H, "split too short for one window"
        s = np.arange(self.lo, self.hi - self.H)
        self.allowed = s if valid is None else s[np.asarray(valid)[s]]
        assert len(self.allowed), "no valid window in split"

    def sample(self, rng, n):
        return self.allowed[rng.integers(0, len(self.allowed), size=n)]

    def grid(self, stride):
        return self.allowed[::stride]


class Objective:
    """J(u, paths): mean (optionally minus lam*std) of per-path returns, in internal coordinates u.

    rollout_fn(theta_phys, starts) -> per-path returns. `rollouts` counts every window rollout run.
    """

    def __init__(self, rollout_fn, sampler, n_paths, to_phys=None, norm=1.0, lam=0.0, fixed_paths=None, record=False):
        self.rollout_fn, self.sampler, self.n_paths = rollout_fn, sampler, n_paths
        self.to_phys = to_phys or (lambda u: u)
        self.norm, self.lam = float(norm), float(lam)
        self.fixed_paths = fixed_paths
        self.rollouts = 0
        self.record = record
        self.log = []
        self.max_touch = -1

    def draw(self, rng):
        if self.fixed_paths is not None:
            return self.fixed_paths
        return self.sampler.sample(rng, self.n_paths)

    def per_path(self, u, paths):
        paths = np.asarray(paths)
        self.rollouts += len(paths)
        if self.record:
            self.log.append(paths.copy())
        if len(paths):
            self.max_touch = max(self.max_touch, int(paths.max()) + self.sampler.H)
        return np.asarray(self.rollout_fn(self.to_phys(u), paths), dtype=float)

    def __call__(self, u, paths):
        r = self.per_path(u, paths)
        v = r.mean() - (self.lam * r.std() if self.lam else 0.0)
        return v / self.norm


class PathProvider:
    """One per gradient estimate. crn=True: every evaluation (base and perturbed) uses the identical paths."""

    def __init__(self, J, rng, crn=True):
        self.J, self.rng, self.crn = J, rng, crn
        self.paths = J.draw(rng) if crn else None

    def eval(self, u):
        paths = self.paths if self.crn else self.J.draw(self.rng)
        return self.J(u, paths)
