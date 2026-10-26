"""Ground truth: run every candidate in the simulator on the TEST period with common random numbers."""
import numpy as np

from env import make_episodes, rollout


def _act_fn(policy, N, T, seed, ids, soft):
    if policy.stochastic or soft:
        U = np.stack([np.random.default_rng([seed, int(i), 13]).random(T) for i in ids])
        probs = policy.probs if (soft or policy.stochastic) else policy.eval_probs

        def f(s, t):
            P = probs(s)
            return (np.cumsum(P, 1) < U[:, t:t + 1]).sum(1).clip(0, P.shape[1] - 1)
        return f
    return lambda s, t: policy.greedy(s)


def true_returns(policy, ep, p, seed, soft=False):
    N, T = ep.demand.shape
    return rollout(ep, p, _act_fn(policy, N, T, seed, ep.ids, soft))["ret"]


def bootstrap_ci(x, n_boot=500, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    m = np.array([x[rng.integers(len(x), size=len(x))].mean() for _ in range(n_boot)])
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def truth_table(policies, series, cfg, p, dm, n_episodes, n_episodes_mcts, horizon, seed, mode="gen", soft=False):
    """Returns (dict name -> per-episode returns array). MCTS candidates use the first n_episodes_mcts episodes
    (same CRN draws, fewer of them, because planning is expensive)."""
    ep = make_episodes(series, cfg, dm, "test", np.arange(n_episodes) + 5_000_000, seed, mode, horizon)
    out = {}
    for pol in policies:
        if pol.family == "MCTS":
            sub = type(ep)(*[getattr(ep, f)[:n_episodes_mcts] for f in ep.__dataclass_fields__])
            out[pol.name] = true_returns(pol, sub, p, seed, soft)
        else:
            out[pol.name] = true_returns(pol, ep, p, seed, soft)
    return out


def summarize(returns):
    rows = {}
    for k, x in returns.items():
        lo, hi = bootstrap_ci(x)
        rows[k] = {"truth": float(np.mean(x)), "truth_lo": lo, "truth_hi": hi, "n_ep": len(x)}
    return rows
