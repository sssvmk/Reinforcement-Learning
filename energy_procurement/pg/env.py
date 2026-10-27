"""Episode environment (one delivery day = 24 sequential hourly decisions), risk objectives, references."""
import os
import numpy as np
import torch
import costs
import data
from risk import cvar

FC_LEVELS = (0.1, 0.5, 0.9)


class Risk:
    """Episode objective: 'neutral', 'cvar' (learned eta, Rockafellar-Uryasev) or 'meanvar' (running mean m)."""

    def __init__(self, kind="neutral", lam=1.0, alpha=0.95, kappa=8.0, lr_eta=0.02, ema=0.05, eta0=1.5, m0=1.0):
        # "*_cond": conditional per-day risk computed over scenarios inside the environment (the quantity the
        # classical agents minimize); the rollout reward already contains it, so no cross-day penalty applies.
        self.cond = kind.endswith("_cond")
        self.base = kind[:-5] if self.cond else kind
        self.kind, self.lam, self.alpha, self.kappa = kind, lam, alpha, kappa
        self.lr_eta, self.ema = lr_eta, ema
        self.eta, self.m = eta0, m0

    def penalty(self, Ct):
        """Risk term subtracted from the return on the last step (>= 0)."""
        Ct = np.asarray(Ct)
        if self.cond:
            return np.zeros_like(Ct)
        if self.kind == "cvar":
            return self.lam * np.maximum(Ct - self.eta, 0.0) / (1 - self.alpha)
        if self.kind == "meanvar":
            return self.kappa * (Ct - self.m) ** 2
        return np.zeros_like(Ct)

    def update(self, Ct):
        Ct = np.asarray(Ct)
        if self.cond:
            return
        if self.kind == "cvar":
            g = 1.0 - np.mean(Ct > self.eta) / (1 - self.alpha)  # d/d eta of eta + E(C-eta)+/(1-alpha)
            self.eta = self.eta - self.lr_eta * g
        elif self.kind == "meanvar":
            self.m = (1 - self.ema) * self.m + self.ema * Ct.mean()

    def objective(self, Ct):
        """Scalar batch objective being minimized (what evaluation reports)."""
        Ct = np.asarray(Ct)
        if self.cond:
            return Ct.mean()
        if self.kind == "cvar":
            return Ct.mean() + self.lam * cvar(Ct, self.alpha)
        if self.kind == "meanvar":
            return Ct.mean() + self.kappa * Ct.var()
        return Ct.mean()


class DayEnv:
    def __init__(self, P, fc=None, use_fc=True, burn=True):
        self.P, self.use_fc = P, use_fc and fc is not None
        X = P.X.copy()
        tr = P.idx("train")
        feats = [X]
        if self.use_fc:
            lv = list(__import__("agents.forecast", fromlist=["LEVELS"]).LEVELS)
            cols = [fc["eL_q"][..., lv.index(q)] for q in FC_LEVELS] + [fc["g_mean"]]
            gm = fc["g_mean"]
            # regime features: forecast gap <= 0 means the day-ahead purchase is not worth making (newsvendor Cu <= 0)
            cols += [(gm <= 0).astype(float), gm / np.maximum(P.pDA, 1.0)]
            feats.append(np.stack(cols, -1))
        X = np.concatenate(feats, -1)
        mu = np.nanmean(X[tr].reshape(-1, X.shape[-1]), 0)
        sd = np.nanstd(X[tr].reshape(-1, X.shape[-1]), 0) + 1e-9
        Z = np.nan_to_num((X - mu) / sd)
        self.static = torch.tensor(Z, dtype=torch.float32)
        self.n_static = Z.shape[-1]
        self.n_in = self.n_static + 1  # + committed ratio
        self.Lhat, self.L, self.pDA, self.pA, self.cap = P.Lhat, P.L, P.pDA, P.pA, P.cap
        self.step_scale = P.Cref  # r_h = -cost_h / Cref so the day's return is -Ctilde
        self.rng = np.random.default_rng(0)
        self.fc, self.cond, self._scen = fc, None, None
        self.last_real = None

    # ---- conditional (scenario-based) per-day risk ----
    def set_cond(self, risk, S_train=64, S_eval=300):
        """Switch the episode reward to the conditional per-day risk of `risk` (a Risk with .cond), or off (None)."""
        self.cond = risk if (risk is not None and risk.cond) else None
        if self.cond is not None and self._scen is None:
            from agents import scenarios
            sc = scenarios.Scenarios(self.P, "gbm", self.fc)
            rg = np.random.default_rng(123)
            n = len(sc.rL)
            self._scen = {"sc": sc, "train": rg.choice(n, S_train, replace=False), "eval": rg.choice(n, S_eval, replace=False)}

    def _cond(self, Q, idx, ev):
        """Scenario daily costs for quantities Q (B,24): reward matrix R (B,24) and conditional risk value J (B,)."""
        z, r = self._scen, self.cond
        sc = z["sc"]
        sel = z["eval" if ev else "train"]
        Ls = self.Lhat[idx][:, None, :] * (1 + sc.cL[idx][:, None, :] + sc.rL[sel][None])
        pAs = np.maximum(self.pDA[idx][:, None, :] + sc.cg[idx][:, None, :] + sc.rg[sel][None], 0.0)
        Cs = costs.hourly_cost(Q[:, None, :], Ls, self.pDA[idx][:, None, :], pAs, self.P.rho, self.P.pi)  # (B,S,24)
        day = Cs.sum(2) / self.P.Cref
        R = -Cs.mean(1) / self.step_scale
        if r.base == "cvar":
            eta = np.quantile(day, r.alpha, axis=1, method="inverted_cdf")
            term = r.lam * (eta + np.maximum(day - eta[:, None], 0.0).mean(1) / (1 - r.alpha))
        elif r.base == "meanvar":
            term = r.kappa * day.var(1)
        else:
            term = np.zeros(len(idx))
        R[:, -1] -= term
        return torch.tensor(R, dtype=torch.float32), day.mean(1) + term

    def cond_value(self, Q, idx):
        """Conditional per-day risk of fixed schedules Q on the evaluation scenarios (for references)."""
        return self._cond(np.asarray(Q), np.asarray(idx), True)[1]

    def rollout(self, idx, act_fn, noise_fn=None, ev=False):
        """Run 24 sequential decisions for days idx. act_fn(states (B,F)) -> actions (B,) tensor.
        Returns dict with states (B,24,F), actions (B,24), rewards (B,24) = -cost_h/(Cref/24), q, Ct."""
        idx = np.asarray(idx)
        B = len(idx)
        T = 24
        F = self.n_in
        S = torch.zeros(B, T, F)
        A = torch.zeros(B, T)
        Q = np.zeros((B, T))
        cum_q = np.zeros(B)
        cum_l = np.zeros(B)
        for h in range(T):
            cr = np.where(cum_l > 0, cum_q / np.maximum(cum_l, 1e-9) - 1.0, 0.0) * 10.0
            s = torch.cat([self.static[idx, h], torch.tensor(cr, dtype=torch.float32)[:, None]], -1)
            a = act_fn(s)
            q = np.clip(self.Lhat[idx, h] * (1.0 + a.detach().numpy().astype(np.float64)), 0.0, self.cap[idx, h])
            S[:, h], A[:, h], Q[:, h] = s, a, q
            cum_q += q
            cum_l += self.Lhat[idx, h]
        cost_h = costs.hourly_cost(Q, self.L[idx], self.pDA[idx], self.pA[idx], self.P.rho, self.P.pi)
        R = torch.tensor(-cost_h / self.step_scale, dtype=torch.float32)
        Ct = cost_h.sum(1) / self.P.Cref
        self.last_real = Ct
        if self.cond is not None:
            R, J = self._cond(Q, idx, ev)
            return {"S": S, "A": A, "R": R, "q": Q, "Ct": J, "Ct_real": Ct, "idx": idx}
        return {"S": S, "A": A, "R": R, "q": Q, "Ct": Ct, "Ct_real": Ct, "idx": idx}

    def eval_mean(self, mean_fn, idx):
        """Deterministic policy evaluation (Gaussian mean); with a conditional objective returns its per-day value."""
        with torch.no_grad():
            r = self.rollout(idx, mean_fn, ev=True)
        return r["Ct"], r["q"], r["A"].numpy()


# ---------------- references and distillation ----------------
def reference_schedule(split, objective, params, tag=""):
    """Classical agent for the same objective on the same days: dict q (n,24) from saved schedules."""
    z = np.load(os.path.join(data.OUT, "agents", f"schedules{tag}.npz"))
    objective = objective[:-5] if objective.endswith("_cond") else objective
    if objective == "neutral":
        return z[f"{split}|nvB|gbm"]
    if objective == "cvar":
        return z[f"{split}|cvar|gbm|{params['alpha']}|{params['lam']}"]
    return z[f"{split}|mv|gbm|{params['kappa']}"]


def distill_theta(env, P, fc, n_epochs=1, ridge=1e-3):
    """Least-squares fit of the linear mean policy (static features + committed ratio weight 0) to the
    newsvendor-B (gbm scenarios) action a* = q/Lhat - 1 on training days."""
    from agents import scenarios, newsvendor as nv
    tr = P.idx("train")
    sc = scenarios.Scenarios(P, "gbm", fc)
    q = nv.newsvendor_B(P, sc, tr)
    a = q / P.Lhat[tr] - 1.0
    Z = env.static[tr].numpy().reshape(-1, env.n_static)
    Z1 = np.c_[Z, np.zeros(len(Z)), np.ones(len(Z))]  # weight on the committed ratio kept at 0
    keep = np.r_[np.arange(env.n_static), env.n_static + 1]
    A_ = Z1[:, keep]
    w = np.linalg.solve(A_.T @ A_ + ridge * np.eye(A_.shape[1]), A_.T @ a.reshape(-1))
    theta = np.zeros(env.n_in + 1)
    theta[:env.n_static] = w[:-1]
    theta[-1] = w[-1]
    return theta  # layout matches nn.Linear: weights (n_in) then bias
