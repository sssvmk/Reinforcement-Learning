"""REINFORCE: score-function gradient with rewards-to-go; baselines none / loo / value."""
import numpy as np
import torch
from pg import policies
from pg.common import Algo, collect, batch_stats


class Reinforce(Algo):
    name = "reinforce"

    def __init__(self, env, risk, cfg, seed):
        super().__init__(env, risk, cfg, seed)
        self.baseline = cfg.get("baseline", "loo")
        self.params = list(self.policy.parameters())
        self.opt = torch.optim.Adam(self.params, lr=cfg.get("lr", 3e-3))
        self.B = cfg.get("B", 32)
        self.value = policies.ValueNet(env.n_in) if self.baseline == "value" else None
        if self.value is not None:
            self.vopt = torch.optim.Adam(self.value.parameters(), lr=cfg.get("vf_lr", 3e-3))

    def iterate(self):
        idx = self.sample_days(self.B)
        r = collect(self.env, self.policy, idx, self.risk)
        S, A, R = r["S"], r["A"], r["Radj"]
        G = torch.zeros_like(R)
        gam, run = self.cfg.get("gamma", 1.0), torch.zeros(self.B)
        for t in reversed(range(R.shape[1])):
            run = R[:, t] + gam * run
            G[:, t] = run  # (discounted) rewards-to-go
        if self.baseline == "loo":
            b = (G.sum(0, keepdim=True) - G) / max(self.B - 1, 1)
        elif self.baseline == "value":
            with torch.no_grad():
                b = self.value(S)
        else:
            b = torch.zeros_like(G)
        adv = G - b
        adv_var = float(adv.var())
        if self.cfg.get("std_adv", True):
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        old = torch.cat([p.detach().reshape(-1) for p in self.params])

        def loss_fn(sl):
            return -(self.policy.log_prob(S[sl], A[sl]) * adv[sl]).mean()

        g, snr, gvar = policies.chunked_grad(self.params, loss_fn, self.B, 4)
        gn = float(g.norm())
        clip = self.cfg.get("clip_grad", 1.0)
        if gn > clip:
            g = g * clip / gn
        i = 0
        for p in self.params:
            p.grad = g[i:i + p.numel()].view_as(p).clone()
            i += p.numel()
        self.opt.step()
        vloss = float("nan")
        if self.value is not None:
            for _ in range(self.cfg.get("vf_epochs", 5)):
                vl = ((self.value(S) - G) ** 2).mean()
                self.vopt.zero_grad(); vl.backward(); self.vopt.step()
            vloss = float(vl)
        self.risk.update(r["Ct"])
        new = torch.cat([p.detach().reshape(-1) for p in self.params])
        m = batch_stats(r, self.env)
        m.update({"days": self.B, "grad_norm": gn, "update_norm": float((new - old).norm()), "grad_snr": snr,
                  "grad_var": gvar, "adv_var": adv_var, "value_loss": vloss})
        return m
