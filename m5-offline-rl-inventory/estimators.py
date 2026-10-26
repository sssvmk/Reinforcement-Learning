"""Competing OPE estimators: FQE (main, in fqe.py), self-estimate, CE model-based OPE, WIS, DR, trivial baselines."""
import numpy as np

from candidates.ce import FittedMDP, ope_value, start_distribution
from fqe import q_values


def ce_ope_estimate(policy, fqe_log, disc, gamma, smooth=0.0, S_disc_probs=None):
    """Policy evaluation in a tabular MDP fitted on the FQE-log (not the train-log).
    Policy over discrete states = average of the candidate's eval probabilities over logged states in that bin."""
    nS, nA = disc.n_states, policy.n_actions
    mdp = FittedMDP(fqe_log["s_discrete"], fqe_log["a"], fqe_log["r"], fqe_log["s2_discrete"], nS, nA, smooth)
    P = S_disc_probs if S_disc_probs is not None else policy.eval_probs(fqe_log["s_cont"])
    acc = np.zeros((nS, nA))
    np.add.at(acc, fqe_log["s_discrete"], P)
    cnt = np.bincount(fqe_log["s_discrete"], minlength=nS)[:, None]
    pi = np.where(cnt > 0, acc / np.maximum(cnt, 1), 1.0 / nA)
    mu0 = start_distribution(disc(fqe_log["s0"]), nS)
    v, _ = ope_value(mdp, pi, gamma, mu0)
    return v


def trajectories(log):
    """Reshape the flat log to (n_ep, T) arrays (episodes have equal length)."""
    ids = log["episode_id"]
    order = np.lexsort((log["t"], ids))
    n_ep = len(np.unique(ids))
    T = len(ids) // n_ep
    sh = lambda x: x[order].reshape(n_ep, T, *x.shape[1:])
    return {k: sh(log[k]) for k in ["s_cont", "a", "r", "s2_cont", "behavior_prob"]}


def wis_estimate(pi_probs_fn, log, gamma, clip=100.0):
    """Weighted importance sampling over whole trajectories with clipped weights, on the eps-soft candidate.
    Estimates the H-step truncated discounted return. Returns (value, effective sample size)."""
    tr = trajectories(log)
    n_ep, T = tr["a"].shape
    P = pi_probs_fn(tr["s_cont"].reshape(-1, tr["s_cont"].shape[-1])).reshape(n_ep, T, -1)
    pa = np.take_along_axis(P, tr["a"][..., None], 2)[..., 0]
    logw = np.log(np.maximum(pa, 1e-12)).sum(1) - np.log(tr["behavior_prob"]).sum(1)
    w = np.minimum(np.exp(np.clip(logw, -50, 50)), clip)
    G = (tr["r"] * gamma ** np.arange(T)).sum(1)
    ess = float(w.sum() ** 2 / (w ** 2).sum())
    return float((w * G).sum() / w.sum()), ess


def dr_estimate(pi_probs_fn, q_res, log, gamma, step_clip=10.0):
    """Doubly robust (per-decision, backward recursion) with the FQE Q-function as baseline.
    V_DR(t) = Vhat(s_t) + rho_t (r_t + gamma V_DR(t+1) - Qhat(s_t,a_t)),  V_DR(T) = Vhat(s_T) (truncation bootstrap)."""
    tr = trajectories(log)
    n_ep, T = tr["a"].shape
    d = tr["s_cont"].shape[-1]
    Sflat, S2flat = tr["s_cont"].reshape(-1, d), tr["s2_cont"].reshape(-1, d)
    P, P2 = pi_probs_fn(Sflat).reshape(n_ep, T, -1), pi_probs_fn(S2flat).reshape(n_ep, T, -1)
    Q = q_values(q_res, Sflat).reshape(n_ep, T, -1)
    Q2 = q_values(q_res, S2flat).reshape(n_ep, T, -1)
    V = (P * Q).sum(2)
    V2 = (P2 * Q2).sum(2)
    Qa = np.take_along_axis(Q, tr["a"][..., None], 2)[..., 0]
    rho = np.minimum(np.take_along_axis(P, tr["a"][..., None], 2)[..., 0] / tr["behavior_prob"], step_clip)
    vdr = V2[:, -1]
    for t in range(T - 1, -1, -1):
        vdr = V[:, t] + rho[:, t] * (tr["r"][:, t] + gamma * vdr - Qa[:, t])
    return float(vdr.mean())
