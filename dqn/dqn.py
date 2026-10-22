"""
dqn.py -- Q-network, replay buffer, and DQN agent (with optional Double DQN,
hard/soft target updates, and ablation flags for disabling replay / target
network) used to learn the ordering policy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden_sizes=(128, 128)):
        super().__init__()
        layers = []
        in_dim = state_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """Fixed-capacity circular buffer storing (s, a, r, s', done)."""

    def __init__(self, capacity: int, state_dim: int, seed: Optional[int] = None):
        self.capacity = capacity
        self.state_dim = state_dim
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self._idx = 0
        self._size = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self._size

    def push(self, s, a, r, s2, done):
        i = self._idx
        self.states[i] = s
        self.actions[i] = a
        self.rewards[i] = r
        self.next_states[i] = s2
        self.dones[i] = float(done)
        self._idx = (self._idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = self._rng.integers(0, self._size, size=batch_size)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )


@dataclass
class DQNConfig:
    state_dim: int
    n_actions: int
    hidden_sizes: Tuple[int, ...] = (128, 128)
    gamma: float = 0.97
    lr: float = 5e-4
    batch_size: int = 64
    buffer_capacity: int = 100_000
    warmup_steps: int = 1000
    target_update_mode: str = "hard"  # "hard" or "soft"
    target_sync_every: int = 500
    tau: float = 0.005
    double_dqn: bool = True
    grad_clip_norm: float = 10.0
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 20_000
    use_replay: bool = True       # ablation flag: if False, learn on-line from the latest transition only
    use_target_network: bool = True  # ablation flag: if False, bootstrap off the online net itself
    seed: int = 42
    device: str = "cpu"


class DQNAgent:
    def __init__(self, config: DQNConfig):
        self.cfg = config
        torch.manual_seed(config.seed)
        self.device = torch.device(config.device)

        self.q = QNetwork(config.state_dim, config.n_actions, config.hidden_sizes).to(self.device)
        self.target_q = QNetwork(config.state_dim, config.n_actions, config.hidden_sizes).to(self.device)
        self.target_q.load_state_dict(self.q.state_dict())
        self.target_q.eval()

        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=config.lr)
        self.buffer = ReplayBuffer(
            config.buffer_capacity if config.use_replay else max(config.batch_size, 1),
            config.state_dim,
            seed=config.seed,
        )

        self.total_steps = 0
        self._last_loss = 0.0
        self._last_max_q = 0.0

    # ------------------------------------------------------------------ #
    def epsilon(self) -> float:
        c = self.cfg
        frac = min(1.0, self.total_steps / max(1, c.eps_decay_steps))
        return c.eps_start + frac * (c.eps_end - c.eps_start)

    def act(self, obs: np.ndarray, greedy: bool = False) -> int:
        eps = 0.0 if greedy else self.epsilon()
        if (not greedy) and np.random.rand() < eps:
            return int(np.random.randint(self.cfg.n_actions))
        with torch.no_grad():
            s = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.q(s)
            self._last_max_q = float(q_values.max().item())
            return int(q_values.argmax(dim=1).item())

    def store(self, s, a, r, s2, done):
        self.buffer.push(s, a, r, s2, done)

    def _sync_target(self):
        if self.cfg.target_update_mode == "hard":
            if self.total_steps % max(1, self.cfg.target_sync_every) == 0:
                self.target_q.load_state_dict(self.q.state_dict())
        else:  # soft / Polyak
            tau = self.cfg.tau
            with torch.no_grad():
                for tp, p in zip(self.target_q.parameters(), self.q.parameters()):
                    tp.data.mul_(1 - tau).add_(tau * p.data)

    def learn_step(self) -> Optional[float]:
        c = self.cfg
        if len(self.buffer) < max(c.warmup_steps, c.batch_size):
            return None

        s, a, r, s2, done = self.buffer.sample(c.batch_size)
        s = torch.as_tensor(s, device=self.device)
        a = torch.as_tensor(a, device=self.device).unsqueeze(1)
        r = torch.as_tensor(r, device=self.device)
        s2 = torch.as_tensor(s2, device=self.device)
        done = torch.as_tensor(done, device=self.device)

        q_sa = self.q(s).gather(1, a).squeeze(1)

        with torch.no_grad():
            target_net = self.target_q if c.use_target_network else self.q
            if c.double_dqn:
                next_actions = self.q(s2).argmax(dim=1, keepdim=True)
                next_q = target_net(s2).gather(1, next_actions).squeeze(1)
            else:
                next_q = target_net(s2).max(dim=1).values
            y = r + c.gamma * next_q * (1.0 - done)

        loss = F.smooth_l1_loss(q_sa, y)  # Huber loss
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), c.grad_clip_norm)
        self.optimizer.step()

        if c.use_target_network:
            self._sync_target()

        self._last_loss = float(loss.item())
        return self._last_loss

    def step_counter(self):
        self.total_steps += 1

    def save(self, path: str):
        torch.save(
            {"q_state_dict": self.q.state_dict(), "config": self.cfg.__dict__},
            path,
        )

    def load(self, path: str):
        # Trusted, locally-produced checkpoint (our own training run), so we
        # disable the weights_only restriction added in newer PyTorch to allow
        # the accompanying plain-Python config dict to unpickle.
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.q.load_state_dict(ckpt["q_state_dict"])
        self.target_q.load_state_dict(ckpt["q_state_dict"])
