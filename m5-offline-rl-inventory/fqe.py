"""Fitted Q Evaluation. Target y = r + gamma*(1-done)*sum_a pi(a|s') Q_target(s',a): the EVALUATED policy's action
(exact expectation for stochastic pi), never the max. Truncated transitions bootstrap like any other (done=False)."""
import numpy as np
import torch
import torch.nn.functional as F

from candidates.dqn import mlp

torch.set_num_threads(4)


def fqe_fit(S, A, R, S2, D, P2, S0, P0, r_scale, gamma, rounds=25, steps=120, lr=1e-3, hidden=64, batch=512,
            seed=0, loss="huber", weight_decay=0.0, huber_delta=5.0):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    T = lambda x, dt=torch.float32: torch.as_tensor(np.asarray(x), dtype=dt)
    S, S2, A_, R_, D_, P2, S0t, P0 = T(S), T(S2), T(A, torch.long), T(R) / r_scale, T(D), T(P2), T(S0), T(P0)
    nA = P2.shape[1]
    q = mlp(S.shape[1], nA, hidden)
    tgt = mlp(S.shape[1], nA, hidden)
    tgt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=lr, weight_decay=weight_decay)
    n = len(S)
    hist = []
    probe = torch.randint(0, n, (min(n, 4000),), generator=g)
    for k in range(rounds):
        for _ in range(steps):
            idx = torch.randint(0, n, (batch,), generator=g)
            with torch.no_grad():
                y = R_[idx] + gamma * (1 - D_[idx]) * (P2[idx] * tgt(S2[idx])).sum(1)
            pred = q(S[idx]).gather(1, A_[idx][:, None]).squeeze(1)
            l = F.huber_loss(pred, y, delta=huber_delta) if loss == "huber" else F.mse_loss(pred, y)
            opt.zero_grad(); l.backward(); opt.step()
        with torch.no_grad():
            pi_ = probe
            y = R_[pi_] + gamma * (1 - D_[pi_]) * (P2[pi_] * tgt(S2[pi_])).sum(1)
            be = float(((q(S[pi_]).gather(1, A_[pi_][:, None]).squeeze(1) - y) ** 2).mean())
            v0 = (P0 * q(S0t)).sum(1) * r_scale
        hist.append({"round": k + 1, "bellman_mse": be, "value": float(v0.mean())})
        tgt.load_state_dict(q.state_dict())            # refresh target after each outer round
    with torch.no_grad():
        v0 = ((P0 * q(S0t)).sum(1) * r_scale).numpy()
    return {"value": float(v0.mean()), "start_values": v0, "net": q, "history": hist, "r_scale": r_scale}


def q_values(res, S):
    with torch.no_grad():
        return res["net"](torch.as_tensor(np.asarray(S), dtype=torch.float32)).numpy() * res["r_scale"]


def bootstrap_mean_ci(x, n_boot=500, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    m = np.array([x[rng.integers(len(x), size=len(x))].mean() for _ in range(n_boot)])
    return float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2))


def fqe_evaluate(policy_eval_probs, log, r_scale, gamma, seeds=3, sub_idx=None, **kw):
    """policy_eval_probs: callable S -> (N,A). `log` is the FQE-log dict. sub_idx restricts to a fixed random subsample
    of transitions (used for expensive MCTS candidates)."""
    S, A, R, S2, D = log["s_cont"], log["a"], log["r"], log["s2_cont"], log["done"]
    if sub_idx is not None:
        S, A, R, S2, D = S[sub_idx], A[sub_idx], R[sub_idx], S2[sub_idx], D[sub_idx]
    P2 = policy_eval_probs(S2)
    S0 = log["s0"]
    P0 = policy_eval_probs(S0)
    runs = [fqe_fit(S, A, R, S2, D, P2, S0, P0, r_scale, gamma, seed=s, **kw) for s in range(seeds)]
    vals = np.array([r["value"] for r in runs])
    sv = np.mean([r["start_values"] for r in runs], axis=0)
    lo, hi = bootstrap_mean_ci(sv)
    return {"value": float(vals.mean()), "seed_std": float(vals.std(ddof=1)) if seeds > 1 else 0.0,
            "ci": (lo, hi), "runs": runs, "history": runs[0]["history"]}
