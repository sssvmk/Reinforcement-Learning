"""Finite-difference policy gradient (FDPG) on a linear policy.
Perturbs theta, evaluates each perturbed deterministic policy on the SAME days (common random numbers)."""
import numpy as np
import torch
from pg.common import Algo, batch_stats


class FDPG(Algo):
    name = "fdpg"

    def __init__(self, env, risk, cfg, seed):
        cfg = {**cfg, "policy": "linear"}
        super().__init__(env, risk, cfg, seed)
        self.N = cfg.get("N", 128)
        self.method = cfg.get("fd", "central")  # central | spsa
        self.eps = cfg.get("eps", 0.02)
        self.lr = cfg.get("lr", 3e-3)
        self.n_dir = cfg.get("n_dir", 4)  # directions per iteration (SPSA)
        self.clip = cfg.get("clip_grad", 1.0)
        self.m1 = self.m2 = None
        self.t = 0

    def days_per_iter(self):
        d = len(self.theta())
        return self.N * ((2 * d if self.method == "central" else 2 * self.n_dir) + 1)

    def _set(self, th):
        w = torch.tensor(th, dtype=torch.float32)
        self.policy.mu.weight.data.copy_(w[:-1][None])
        self.policy.mu.bias.data.copy_(w[-1:])

    def _J(self, th, idx):
        self._set(th)
        with torch.no_grad():
            r = self.env.rollout(idx, self.policy.mean)
        pen = self.risk.penalty(r["Ct"])
        return float((r["Ct"] + pen).mean()), r

    def iterate(self):
        idx = self.sample_days(self.N)
        th = self.theta().astype(np.float64)
        d = len(th)
        g = np.zeros(d)
        n_eval = 0
        if self.method == "central":
            for j in range(d):
                e = np.zeros(d); e[j] = self.eps
                jp, _ = self._J(th + e, idx); jm, _ = self._J(th - e, idx)
                g[j] = (jp - jm) / (2 * self.eps)
                n_eval += 2
        else:
            for _ in range(self.n_dir):
                delta = self.rng.choice([-1.0, 1.0], d)
                jp, _ = self._J(th + self.eps * delta, idx); jm, _ = self._J(th - self.eps * delta, idx)
                g += (jp - jm) / (2 * self.eps) * delta
                n_eval += 2
            g /= self.n_dir
        self._set(th)
        j0, r0 = self._J(th, idx)
        gn = float(np.linalg.norm(g))
        if gn > self.clip:
            g = g * self.clip / gn
        # Adam step on the descent direction
        self.t += 1
        if self.m1 is None:
            self.m1, self.m2 = np.zeros(d), np.zeros(d)
        self.m1 = 0.9 * self.m1 + 0.1 * g
        self.m2 = 0.999 * self.m2 + 0.001 * g ** 2
        step = self.lr * (self.m1 / (1 - 0.9 ** self.t)) / (np.sqrt(self.m2 / (1 - 0.999 ** self.t)) + 1e-8)
        new = th - step
        self._set(new)
        self.risk.update(r0["Ct"])
        m = {"G_mean": -j0, "Ct_mean": float(r0["Ct"].mean()), "Ct_std": float(r0["Ct"].std()),
             "batch_cvar95": float(np.sort(r0["Ct"])[int(0.95 * len(r0["Ct"])):].mean()),
             "days": self.N * (n_eval + 1), "grad_norm": gn, "update_norm": float(np.linalg.norm(new - th)),
             "n_eval": n_eval + 1}
        return m
