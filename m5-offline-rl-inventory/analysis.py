"""Diagnostics: CE model-error study and coverage (log-support) diagnostic."""
import numpy as np
import pandas as pd

from candidates.ce import FittedMDP
from env import DemandModel, Day, EnvParams, features, transition, unpack


def model_error(mdp: FittedMDP, log, p: EnvParams, disc, dm: DemandModel, seed=0, n_rep=3, n_draw=30):
    """Compare fitted P_hat / r_hat with the true simulator for every (s,a): states are those present in the log
    (representative logged continuous states), actions are ALL actions (so unvisited pairs are included).
    Returns a DataFrame with reward error (in units of r_scale) and total-variation distance of next discrete state."""
    rng = np.random.default_rng(seed)
    s_all, sd = log["s_cont"], log["s_discrete"]
    L, m = p.lead_time, p.mean_demand
    rows = []
    for s in np.unique(sd):
        idx = np.where(sd == s)[0]
        idx = idx[rng.permutation(len(idx))[:n_rep]]
        for a in range(p.n_actions):
            rs, nxt = [], np.zeros(mdp.nS)
            for i in idx:
                sc = s_all[i]
                inv = sc[:1 + L] * m
                pos, on, wd, ev, prel = unpack(sc, p)
                price = prel * p.mean_price
                d = dm.sample(rng, np.full(n_draw, int(wd)), np.zeros(n_draw, int))
                ninv, r, _ = transition(np.tile(inv, (n_draw, 1)), np.full(n_draw, p.actions[a]), d, Day(price), p)
                rs.append(r.mean())
                s2 = features(ninv, np.full(n_draw, (int(wd) + 1) % 7), np.zeros(n_draw), np.zeros(n_draw),
                              np.full(n_draw, price), np.full(n_draw, sc[-1] * m), p)
                nxt += np.bincount(disc(s2), minlength=mdp.nS) / n_draw
            nxt /= len(idx)
            visited = mdp.n[s, a] > 0
            rhat = mdp.R[s, a] if visited else np.nan
            tv = 0.5 * np.abs(mdp.P[s, a] - nxt).sum() if visited else np.nan
            rows.append({"s": int(s), "a": a, "n": mdp.n[s, a], "visited": visited, "r_true": float(np.mean(rs)),
                         "r_err": abs(rhat - np.mean(rs)) / p.r_scale if visited else np.nan, "tv": tv})
    return pd.DataFrame(rows)


def policy_model_error(pol_greedy_disc, a_greedy, err_df, nA):
    """Mean TV error of the model at the (state, greedy-action) pairs a policy actually uses (visited pairs only)."""
    tab = np.full((err_df["s"].max() + 1, nA), np.nan)
    tab[err_df["s"].to_numpy(), err_df["a"].to_numpy()] = err_df["tv"].to_numpy()
    v = tab[pol_greedy_disc.clip(0, tab.shape[0] - 1), a_greedy]
    return float(np.nanmean(v)) if np.isfinite(v).any() else np.nan


def support_fraction(policy, held_out_log, train_counts, disc, n_min=5):
    """Fraction of held-out states where the policy's greedy action has >= n_min logged occurrences of (s_disc, a)."""
    S = held_out_log["s_cont"]
    a = policy.greedy(S)
    return float((train_counts[disc(S), a] >= n_min).mean())
