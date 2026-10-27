"""Policies, value networks and shared policy-gradient utilities (PyTorch, CPU)."""
import math
import numpy as np
import torch
import torch.nn as nn

SIGMA_FLOOR = 0.01
ACT_SCALE = 30.0  # actions are O(1-3%) of load; rescale for the Q network input


class GaussianPolicy(nn.Module):
    """a ~ N(mu(s), sigma^2) with state-independent log sigma. mean: 'linear' (default) or 'mlp' (32 tanh)."""

    def __init__(self, n_in, kind="linear", init_sigma=0.03, hidden=32):
        super().__init__()
        self.kind = kind
        if kind == "linear":
            self.mu = nn.Linear(n_in, 1)
            nn.init.zeros_(self.mu.weight)
            nn.init.zeros_(self.mu.bias)
        else:
            self.mu = nn.Sequential(nn.Linear(n_in, hidden), nn.Tanh(), nn.Linear(hidden, 1))
            nn.init.zeros_(self.mu[2].weight)
            nn.init.zeros_(self.mu[2].bias)
        self.log_std = nn.Parameter(torch.tensor(math.log(init_sigma)))

    def mean(self, s):
        return self.mu(s).squeeze(-1)

    @property
    def sigma(self):
        return torch.clamp(self.log_std.exp(), min=SIGMA_FLOOR)

    def log_prob(self, s, a):
        mu, sg = self.mean(s), self.sigma
        return -0.5 * ((a - mu) / sg) ** 2 - torch.log(sg) - 0.5 * math.log(2 * math.pi)

    def entropy(self):
        return torch.log(self.sigma) + 0.5 * math.log(2 * math.pi * math.e)

    def sample(self, s, generator=None):
        mu = self.mean(s)
        eps = torch.randn(mu.shape, generator=generator)
        return (mu + self.sigma * eps).detach()

    def theta(self):
        """Flat vector of the *mean* parameters (used for parameter-space distances)."""
        return torch.cat([p.detach().reshape(-1) for p in self.mu.parameters()]).numpy().copy()


def kl_gauss(mu_old, sg_old, mu_new, sg_new):
    """Mean KL(old || new) between diagonal Gaussians (per-step, scalar action)."""
    return (torch.log(sg_new / sg_old) + (sg_old ** 2 + (mu_old - mu_new) ** 2) / (2 * sg_new ** 2) - 0.5).mean()


class ValueNet(nn.Module):
    def __init__(self, n_in, hidden=32, kind="mlp"):
        super().__init__()
        self.net = nn.Linear(n_in, 1) if kind == "linear" else nn.Sequential(
            nn.Linear(n_in, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, s):
        return self.net(s).squeeze(-1)


class QNet(nn.Module):
    def __init__(self, n_in, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in + 1, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, s, a):
        return self.net(torch.cat([s, a.unsqueeze(-1) * ACT_SCALE], -1)).squeeze(-1)


def gae(rewards, values, gamma=1.0, lam=0.95):
    """rewards (B,T); values (B,T+1) with values[:,T] the bootstrap (0 at episode end). Returns adv, ret."""
    B, T = rewards.shape
    adv = torch.zeros_like(rewards)
    last = torch.zeros(B)
    for t in reversed(range(T)):
        delta = rewards[:, t] + gamma * values[:, t + 1] - values[:, t]
        last = delta + gamma * lam * last
        adv[:, t] = last
    return adv, adv + values[:, :T]


def chunked_grad(params, loss_fn, n_items, n_chunks=4):
    """Gradient of the mean of loss_fn(slice) over equal chunks. Returns (mean grad, snr, variance of the mean).
    SNR = ||g|| / sqrt(trace Var(chunk grads) / n_chunks)."""
    edges = np.linspace(0, n_items, n_chunks + 1).astype(int)
    gs = []
    for a, b in zip(edges[:-1], edges[1:]):
        gs.append(flat_grad(loss_fn(slice(a, b)), params))
    G = torch.stack(gs)
    g = G.mean(0)
    var_mean = float(((G - g) ** 2).sum(1).sum() / max(n_chunks - 1, 1) / n_chunks)
    snr = float(g.norm() / (math.sqrt(var_mean) + 1e-12))
    return g, snr, var_mean


def flat_grad(loss, params, retain=False, create=False):
    g = torch.autograd.grad(loss, params, retain_graph=retain, create_graph=create, allow_unused=True)
    return torch.cat([(x if x is not None else torch.zeros_like(p)).reshape(-1) for x, p in zip(g, params)])


def set_flat(params, vec):
    i = 0
    for p in params:
        n = p.numel()
        p.data.copy_(vec[i:i + n].view_as(p))
        i += n


def get_flat(params):
    return torch.cat([p.detach().reshape(-1) for p in params])


def explained_variance(pred, target):
    v = target.var()
    return float(1 - (target - pred).var() / (v + 1e-12))
