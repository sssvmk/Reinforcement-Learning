"""TRPO: natural gradient via conjugate gradient on Fisher-vector products, KL-constrained line search."""
import numpy as np
import torch
from pg import policies
from pg.common import Algo, collect, batch_stats


def conjugate_gradient(Avp, b, iters=10, tol=1e-10):
    x = torch.zeros_like(b)
    r, p = b.clone(), b.clone()
    rr = r @ r
    for _ in range(iters):
        Ap = Avp(p)
        a = rr / (p @ Ap + 1e-12)
        x += a * p
        r -= a * Ap
        rr_new = r @ r
        if rr_new < tol:
            break
        p = r + rr_new / rr * p
        rr = rr_new
    return x


class TRPO(Algo):
    name = "trpo"

    def __init__(self, env, risk, cfg, seed):
        super().__init__(env, risk, cfg, seed)
        self.params = list(self.policy.parameters())
        self.value = policies.ValueNet(env.n_in, kind=cfg.get("critic", "mlp"))
        self.vopt = torch.optim.Adam(self.value.parameters(), lr=cfg.get("vf_lr", 3e-3))
        self.B = cfg.get("B", 32)
        self.delta = cfg.get("delta", 0.01)
        self.damping = cfg.get("damping", 0.1)

    def iterate(self):
        cfg = self.cfg
        idx = self.sample_days(self.B)
        r = collect(self.env, self.policy, idx, self.risk)
        S, A, R = r["S"].reshape(-1, self.env.n_in), r["A"].reshape(-1), r["Radj"]
        Sb = r["S"]
        with torch.no_grad():
            v = self.value(Sb)
            vfull = torch.cat([v, torch.zeros(self.B, 1)], 1)
        adv, ret = policies.gae(R, vfull, cfg.get("gamma", 1.0), cfg.get("lam_gae", 0.95))
        advf = adv.reshape(-1)
        advf = (advf - advf.mean()) / (advf.std() + 1e-8)
        with torch.no_grad():
            mu_old, sg_old = self.policy.mean(S), self.policy.sigma.clone()
            lp_old = self.policy.log_prob(S, A)

        def surrogate():
            return (torch.exp(self.policy.log_prob(S, A) - lp_old) * advf).mean()

        g = policies.flat_grad(surrogate(), self.params)  # ascent direction
        gn = float(g.norm())

        def fvp(vec):
            kl = policies.kl_gauss(mu_old, sg_old, self.policy.mean(S), self.policy.sigma)
            gk = policies.flat_grad(kl, self.params, retain=True, create=True)
            return policies.flat_grad(gk @ vec, self.params, retain=True) + self.damping * vec

        step_dir = conjugate_gradient(fvp, g, cfg.get("cg_iters", 10))
        shs = 0.5 * float(step_dir @ fvp(step_dir))
        beta = np.sqrt(self.delta / max(shs, 1e-12))
        full = step_dir * beta
        old = policies.get_flat(self.params)
        s0 = float(surrogate())
        accepted, backtracks, kl_after = False, 0, 0.0
        for k in range(cfg.get("max_backtrack", 10)):
            policies.set_flat(self.params, old + full * (0.5 ** k))
            with torch.no_grad():
                kl_after = float(policies.kl_gauss(mu_old, sg_old, self.policy.mean(S), self.policy.sigma))
                imp = float(surrogate()) - s0
            if kl_after <= 1.5 * self.delta and imp > 0:
                accepted, backtracks = True, k
                break
            backtracks = k + 1
        if not accepted:
            policies.set_flat(self.params, old)
            kl_after = 0.0
        for _ in range(cfg.get("vf_epochs", 5)):
            vl = ((self.value(Sb) - ret) ** 2).mean()
            self.vopt.zero_grad(); vl.backward()
            torch.nn.utils.clip_grad_norm_(self.value.parameters(), 1.0)
            self.vopt.step()
        self.risk.update(r["Ct"])
        new = policies.get_flat(self.params)
        m = batch_stats(r, self.env)
        m.update({"days": self.B, "grad_norm": gn, "update_norm": float((new - old).norm()), "kl": kl_after,
                  "kl_target": self.delta, "backtracks": backtracks, "accepted": float(accepted), "value_loss": float(vl),
                  "surr_gain": float(imp) if accepted else 0.0, "cg_shs": shs})
        return m
