"""Advantage actor-critic (A2C) with GAE, learned critic, entropy bonus, gradient clipping."""
import torch
from pg import policies
from pg.common import Algo, collect, batch_stats


class A2C(Algo):
    name = "a2c"

    def __init__(self, env, risk, cfg, seed):
        super().__init__(env, risk, cfg, seed)
        self.params = list(self.policy.parameters())
        self.value = policies.ValueNet(env.n_in, kind=cfg.get("critic", "mlp"))
        self.opt = torch.optim.Adam(self.params, lr=cfg.get("lr", 3e-3))
        self.vopt = torch.optim.Adam(self.value.parameters(), lr=cfg.get("vf_lr", 3e-3))
        self.B = cfg.get("B", 32)

    def iterate(self):
        cfg = self.cfg
        idx = self.sample_days(self.B)
        r = collect(self.env, self.policy, idx, self.risk)
        S, A, R = r["S"], r["A"], r["Radj"]
        with torch.no_grad():
            v = self.value(S)
            vfull = torch.cat([v, torch.zeros(self.B, 1)], 1)
        adv, ret = policies.gae(R, vfull, cfg.get("gamma", 1.0), cfg.get("lam_gae", 0.95))
        td = float((R + vfull[:, 1:] - v).abs().mean())
        evar = policies.explained_variance(v.flatten(), ret.flatten())
        advn = (adv - adv.mean()) / (adv.std() + 1e-8) if cfg.get("std_adv", True) else adv
        old = torch.cat([p.detach().reshape(-1) for p in self.params])

        def loss_fn(sl):
            return -(self.policy.log_prob(S[sl], A[sl]) * advn[sl]).mean() - cfg.get("ent_coef", 0.0) * self.policy.entropy()

        g, snr, gvar = policies.chunked_grad(self.params, loss_fn, self.B, 4)
        gn = float(g.norm())
        clip = cfg.get("clip_grad", 1.0)
        if gn > clip:
            g = g * clip / gn
        i = 0
        for p in self.params:
            p.grad = g[i:i + p.numel()].view_as(p).clone(); i += p.numel()
        self.opt.step()
        for _ in range(cfg.get("vf_epochs", 5)):
            vl = ((self.value(S) - ret) ** 2).mean()
            self.vopt.zero_grad(); vl.backward()
            torch.nn.utils.clip_grad_norm_(self.value.parameters(), 1.0)
            self.vopt.step()
        self.risk.update(r["Ct"])
        new = torch.cat([p.detach().reshape(-1) for p in self.params])
        m = batch_stats(r, self.env)
        m.update({"days": self.B, "grad_norm": gn, "update_norm": float((new - old).norm()), "grad_snr": snr,
                  "grad_var": gvar, "value_loss": float(vl), "expl_var": evar, "td_error": td})
        return m
