"""Offline DQN and CQL-style DQN agents (optional, PyTorch CPU). Trained on the TRAIN-log only."""
from __future__ import annotations
import numpy as np
from .base import Agent

try:
    import torch
    import torch.nn as nn
    HAVE_TORCH = True
except Exception:  # pragma: no cover
    HAVE_TORCH = False


def train_q_agent(domain, log, steps, cql_alpha, seed, name, family, gamma, reward_scale=None, hidden=64):
    if not HAVE_TORCH:
        raise RuntimeError("PyTorch not available")
    reward_scale = getattr(domain, "reward_scale", 100.0) if reward_scale is None else reward_scale
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    W, H1, k = log.raw.shape
    Xs = log.phi(domain).astype(np.float32)
    d = Xs.shape[2]
    S = torch.from_numpy(Xs[:, :-1].reshape(-1, d))
    Sn = torch.from_numpy(Xs[:, 1:].reshape(-1, d))
    A = torch.from_numpy(log.act.reshape(-1).astype(np.int64))
    Rw = torch.from_numpy((log.r.reshape(-1) * reward_scale).astype(np.float32))
    nA = domain.n_actions
    net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, nA))
    tgt = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, nA))
    tgt.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    N = S.shape[0]
    for it in range(steps):
        idx = torch.from_numpy(rng.integers(0, N, 256))
        q = net(S[idx])
        qsa = q.gather(1, A[idx][:, None])[:, 0]
        with torch.no_grad():
            y = Rw[idx] + gamma * tgt(Sn[idx]).max(dim=1).values
        loss = nn.functional.smooth_l1_loss(qsa, y)
        if cql_alpha > 0:
            loss = loss + cql_alpha * (torch.logsumexp(q, dim=1) - qsa).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (it + 1) % 250 == 0:
            tgt.load_state_dict(net.state_dict())
    net.eval()

    def fn(raw):
        x = torch.from_numpy(domain.phi(raw).astype(np.float32))
        with torch.no_grad():
            return net(x).argmax(dim=1).numpy()

    return Agent(name, family, fn, {"steps": steps, "cql_alpha": cql_alpha})


def build_learned_agents(domain, train_log, gamma, seed, lengths=(1500, 6000), cql_alphas=(0.1, 1.0)):
    if not HAVE_TORCH:
        print("PyTorch missing: skipping DQN/CQL agents")
        return []
    out = []
    for L in lengths:
        out.append(train_q_agent(domain, train_log, L, 0.0, seed + L, f"dqn_{L}", "dqn", gamma))
    for a in cql_alphas:
        out.append(train_q_agent(domain, train_log, lengths[-1], a, seed + 7, f"cql_a{a}", "cql", gamma))
    return out
