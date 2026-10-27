"""Deterministic policy gradient (DDPG-style actor-critic): Q(s,a) critic, replay, target networks, optional twin critics.
The exploration noise is Gaussian with the policy's sigma (fixed here); the actor mean is trained through Q."""
import copy
import numpy as np
import torch
from pg import policies
from pg.common import Algo, collect, batch_stats


class DPG(Algo):
    name = "dpg"

    def __init__(self, env, risk, cfg, seed):
        super().__init__(env, risk, cfg, seed)
        self.policy.log_std.requires_grad_(False)
        self.twin = cfg.get("twin", False)
        self.q1 = policies.QNet(env.n_in)
        self.q2 = policies.QNet(env.n_in) if self.twin else None
        self.q1t = copy.deepcopy(self.q1)
        self.q2t = copy.deepcopy(self.q2) if self.twin else None
        self.aopt = torch.optim.Adam(self.policy.mu.parameters(), lr=cfg.get("lr", 1e-3))
        qp = list(self.q1.parameters()) + (list(self.q2.parameters()) if self.twin else [])
        self.qopt = torch.optim.Adam(qp, lr=cfg.get("q_lr", 3e-3))
        self.tau = cfg.get("polyak", 0.01)
        self.B = cfg.get("B", 16)
        self.buf = None
        self.cap_size = cfg.get("replay", 40000)
        self.upd = cfg.get("updates", 24)
        self.mb = cfg.get("minibatch", 128)
        self.warm = cfg.get("warmup_iters", 5)
        self.it = 0
        self.gamma = 1.0

    def _add(self, S, A, R, S2, done):
        new = [x.reshape(-1, *x.shape[2:]) for x in (S, A, R, S2, done)]
        if self.buf is None:
            self.buf = new
        else:
            self.buf = [torch.cat([b, n])[-self.cap_size:] for b, n in zip(self.buf, new)]

    def _target(self, S2, R, done):
        with torch.no_grad():
            a2 = self.policy.mean(S2)
            if self.cfg.get("target_noise", 0.0) > 0:
                a2 = a2 + (torch.randn_like(a2) * self.cfg["target_noise"]).clamp(-0.05, 0.05)
            q2 = self.q1t(S2, a2)
            if self.twin:
                q2 = torch.min(q2, self.q2t(S2, a2))
            return R + self.gamma * (1 - done) * q2

    def iterate(self):
        self.it += 1
        idx = self.sample_days(self.B)
        r = collect(self.env, self.policy, idx, self.risk)
        S, A, R = r["S"], r["A"], r["Radj"] * 24.0  # hourly rewards ~O(1) for the critic
        S2 = torch.cat([S[:, 1:], S[:, -1:]], 1)
        done = torch.zeros_like(R); done[:, -1] = 1.0
        self._add(S, A, R, S2, done)
        qls, als, qmean = [], [], []
        if self.it > self.warm:
            n = len(self.buf[0])
            for u in range(self.upd):
                b = torch.tensor(self.rng.integers(0, n, self.mb))
                s, a, rr, s2, d = (x[b] for x in self.buf)
                y = self._target(s2, rr, d)
                ql = ((self.q1(s, a) - y) ** 2).mean()
                if self.twin:
                    ql = ql + ((self.q2(s, a) - y) ** 2).mean()
                self.qopt.zero_grad(); ql.backward(); self.qopt.step()
                if (not self.twin) or u % 2 == 0:
                    al = -self.q1(s, self.policy.mean(s)).mean()
                    self.aopt.zero_grad(); al.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.mu.parameters(), self.cfg.get("clip_grad", 1.0))
                    self.aopt.step()
                    als.append(float(al))
                    with torch.no_grad():
                        pairs = [(self.q1, self.q1t)] + ([(self.q2, self.q2t)] if self.twin else [])
                        for net, tgt in pairs:
                            for p, pt in zip(net.parameters(), tgt.parameters()):
                                pt.data.mul_(1 - self.tau).add_(self.tau * p.data)
                qls.append(float(ql))
        self.risk.update(r["Ct"])
        m = batch_stats(r, self.env)
        m.update({"days": self.B, "critic_loss": float(np.mean(qls)) if qls else np.nan,
                  "actor_loss": float(np.mean(als)) if als else np.nan, "buffer": len(self.buf[0])})
        return m
