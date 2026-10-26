"""Certainty-equivalence (fitted tabular MDP), value/policy iteration, model-based OPE. NumPy only."""
import numpy as np

from candidates.base import Policy


class FittedMDP:
    def __init__(self, s, a, r, s2, nS, nA, smooth=0.0):
        self.nS, self.nA, self.smooth = nS, nA, smooth
        self.n = np.zeros((nS, nA))
        np.add.at(self.n, (s, a), 1)
        rs = np.zeros((nS, nA))
        np.add.at(rs, (s, a), r)
        cnt = np.zeros((nS, nA, nS))
        np.add.at(cnt, (s, a, s2), 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            self.R = np.where(self.n > 0, rs / np.maximum(self.n, 1), 0.0)
        self.r_mean = float(np.mean(r))
        self.cnt = cnt
        den = self.n[..., None] + smooth
        self.P = np.where(den > 0, (cnt + smooth / nS) / np.maximum(den, 1e-12), 0.0)
        # (s,a) never observed with smooth == 0 has P == 0: handled by masking
        self.has_data = self.n.sum(1) > 0
        self.freq_action = self.n.argmax(1)
        self.global_action = int(self.n.sum(0).argmax())


def _allowed(mdp, rule, n_min):
    thr = max(1, n_min) if rule == "mask" else 1
    allowed = mdp.n >= thr
    none = ~allowed.any(1)
    fb = none & mdp.has_data
    allowed[fb, mdp.freq_action[fb]] = True      # fall back to the most frequent logged action
    return allowed


def _qeff(mdp, rule, lam):
    R = mdp.R.copy()
    if rule == "pess":
        R = R - lam / np.sqrt(np.maximum(mdp.n, 1))
    return R


def value_iteration(mdp, gamma, rule="mask", n_min=1, lam=0.0, tol=1e-10, max_iter=20000):
    allowed = _allowed(mdp, rule, n_min)
    R = _qeff(mdp, rule, lam)
    V = np.zeros(mdp.nS)
    for _ in range(max_iter):
        Q = R + gamma * (mdp.P @ V)
        Qm = np.where(allowed, Q, -np.inf)
        Vn = np.where(mdp.has_data, Qm.max(1), 0.0)
        if np.max(np.abs(Vn - V)) < tol:
            V = Vn
            break
        V = Vn
    Q = R + gamma * (mdp.P @ V)
    Qm = np.where(allowed, Q, -np.inf)
    pi = np.where(mdp.has_data, Qm.argmax(1), mdp.global_action)
    return pi, V, Q


def policy_iteration(mdp, gamma, rule="mask", n_min=1, lam=0.0, max_iter=200):
    allowed = _allowed(mdp, rule, n_min)
    R = _qeff(mdp, rule, lam)
    pi = np.where(mdp.has_data, np.where(allowed, 0, -1).argmax(1), mdp.global_action)
    idx = np.arange(mdp.nS)
    for _ in range(max_iter):
        Ppi = mdp.P[idx, pi]
        Rpi = R[idx, pi]
        Ppi = np.where(mdp.has_data[:, None], Ppi, 0.0)
        Rpi = np.where(mdp.has_data, Rpi, 0.0)
        V = np.linalg.solve(np.eye(mdp.nS) - gamma * Ppi, Rpi)
        Q = R + gamma * (mdp.P @ V)
        Qm = np.where(allowed, Q, -np.inf)
        new = np.where(mdp.has_data, Qm.argmax(1), mdp.global_action)
        # keep current action when it is already (numerically) optimal
        keep = Qm[idx, pi] >= Qm.max(1) - 1e-12
        new = np.where(keep & mdp.has_data, pi, new)
        if np.array_equal(new, pi):
            break
        pi = new
    return pi, V, Q


def ope_value(mdp, pi_probs, gamma, mu0):
    """Policy evaluation in the fitted MDP ("model-based OPE").
    pi_probs: (nS, nA). Unvisited (s,a): use the nearest visited action of the same state; if the state was never
    seen, self-loop with the mean logged reward (documented fallback)."""
    nS, nA = mdp.nS, mdp.nA
    P, R = mdp.P.copy(), mdp.R.copy()
    seen = mdp.n > 0
    for s in np.where(~seen.all(1))[0]:
        vis = np.where(seen[s])[0]
        for a in np.where(~seen[s])[0]:
            if len(vis):
                b = vis[np.abs(vis - a).argmin()]
                P[s, a], R[s, a] = P[s, b], R[s, b]
            else:
                P[s, a] = 0.0
                P[s, a, s] = 1.0
                R[s, a] = mdp.r_mean
    Ppi = np.einsum("sa,san->sn", pi_probs, P)
    Rpi = (pi_probs * R).sum(1)
    V = np.linalg.solve(np.eye(nS) - gamma * Ppi, Rpi)
    return float(mu0 @ V), V


def start_distribution(s0_disc, nS):
    mu = np.bincount(s0_disc, minlength=nS).astype(float)
    return mu / mu.sum()


class CEPolicy(Policy):
    family = "CE"

    def __init__(self, name, disc, pi, V, config, mu0):
        super().__init__(name, disc.p.n_actions, config)
        self.disc, self.pi = disc, pi
        self.own_value = float(mu0 @ V)

    def greedy(self, S):
        return self.pi[self.disc(S)]


def fit_ce_policy(name, mdp, disc, gamma, mu0, rule="mask", n_min=1, lam=0.0, smooth=0.0):
    pi, V, _ = value_iteration(mdp, gamma, rule, n_min, lam)
    cfg = {"rule": rule, "n_min": n_min, "lam": lam, "smooth": smooth}
    return CEPolicy(name, disc, pi, V, cfg, mu0)
