"""PPO with clipped surrogate, GAE, multiple epochs of minibatches, approximate-KL early stopping."""
import numpy as np
import torch
from pg import policies
from pg.common import Algo, collect, batch_stats


class PPO(Algo):
    name = "ppo"

    def __init__(self, env, risk, cfg, seed):
        super().__init__(env, risk, cfg, seed)
        self.params = list(self.policy.parameters())
        self.value = policies.ValueNet(env.n_in, kind=cfg.get("critic", "mlp"))
        self.opt = torch.optim.Adam(list(self.policy.parameters()) + list(self.value.parameters()), lr=cfg.get("lr", 3e-3))
        self.B = cfg.get("B", 32)

    def iterate(self):
        cfg = self.cfg
        idx = self.sample_days(self.B)
        r = collect(self.env, self.policy, idx, self.risk)
        F = self.env.n_in
        Sb, R = r["S"], r["Radj"]
        S, A = Sb.reshape(-1, F), r["A"].reshape(-1)
        with torch.no_grad():
            v = self.value(Sb)
            vfull = torch.cat([v, torch.zeros(self.B, 1)], 1)
            lp_old = self.policy.log_prob(S, A)
        adv, ret = policies.gae(R, vfull, cfg.get("gamma", 1.0), cfg.get("lam_gae", 0.95))
        advf, retf = adv.reshape(-1), ret.reshape(-1)
        advf = (advf - advf.mean()) / (advf.std() + 1e-8)
        clip, mb = cfg.get("clip", 0.2), cfg.get("minibatch", 64)
        n = len(A)
        old = policies.get_flat(self.params)
        kls, cfr, rmax, gnorms = [], [], [], []
        stopped = False
        for ep in range(cfg.get("epochs", 10)):
            perm = torch.tensor(self.rng.permutation(n))
            ep_kl = []
            for s in range(0, n, mb):
                b = perm[s:s + mb]
                lp = self.policy.log_prob(S[b], A[b])
                ratio = torch.exp(lp - lp_old[b])
                surr = torch.min(ratio * advf[b], torch.clamp(ratio, 1 - clip, 1 + clip) * advf[b]).mean()
                vl = ((self.value(S[b]) - retf[b]) ** 2).mean()
                loss = -surr + cfg.get("vf_coef", 0.5) * vl - cfg.get("ent_coef", 0.0) * self.policy.entropy()
                self.opt.zero_grad(); loss.backward()
                gnorms.append(float(torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.get("clip_grad", 1.0))))
                self.opt.step()
                with torch.no_grad():
                    ep_kl.append(float((lp_old[b] - lp).mean()))
                    cfr.append(float(((ratio - 1).abs() > clip).float().mean()))
                    rmax.append(float(ratio.max()))
            kls.append(np.mean(ep_kl))
            if kls[-1] > cfg.get("target_kl", 0.03):
                stopped = True
                break
        self.risk.update(r["Ct"])
        new = policies.get_flat(self.params)
        m = batch_stats(r, self.env)
        m.update({"days": self.B, "grad_norm": float(np.mean(gnorms)), "update_norm": float((new - old).norm()),
                  "kl": float(kls[-1]), "clip_frac": float(np.mean(cfr)), "ratio_max": float(np.max(rmax)),
                  "epochs_run": len(kls), "early_stop": float(stopped), "value_loss": float(vl)})
        return m
