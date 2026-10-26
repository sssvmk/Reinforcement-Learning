"""Offline DQN (replay of a fixed log, target network, Huber, Adam, grad clipping, optional Double DQN).
Truncated transitions bootstrap normally (done is False at the horizon end)."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from candidates.base import Policy

torch.set_num_threads(4)


def mlp(d_in, d_out, hidden=64, layers=2):
    mods, d = [], d_in
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.ReLU()]
        d = hidden
    mods.append(nn.Linear(d, d_out))
    return nn.Sequential(*mods)


def conservative_penalty(Q, a):
    """CQL(H) penalty: E[logsumexp_a Q(s,a) - Q(s, a_data)] >= 0."""
    return (torch.logsumexp(Q, 1) - Q.gather(1, a[:, None]).squeeze(1)).mean()


def train_dqn(S, A, R, S2, D, n_actions, r_scale, gamma, steps, snapshots=None, seed=0, alpha=0.0, double=False,
              hidden=64, lr=1e-3, batch=128, sync_every=100, tau=None, huber_delta=5.0, return_net=False):
    """Returns {step: state_dict} for every step in `snapshots` (default: the last step)."""
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    S, S2 = torch.as_tensor(S, dtype=torch.float32), torch.as_tensor(S2, dtype=torch.float32)
    A = torch.as_tensor(A, dtype=torch.long)
    R = torch.as_tensor(R / r_scale, dtype=torch.float32)
    D = torch.as_tensor(D, dtype=torch.float32)
    q = mlp(S.shape[1], n_actions, hidden)
    tgt = mlp(S.shape[1], n_actions, hidden)
    tgt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=lr)
    snaps, log = {}, []
    want = set(snapshots or [steps])
    n = len(S)
    for step in range(1, steps + 1):
        idx = torch.randint(0, n, (batch,), generator=g)
        s, a, r, s2, d = S[idx], A[idx], R[idx], S2[idx], D[idx]
        Qs = q(s)
        with torch.no_grad():
            if double:
                a2 = q(s2).argmax(1, keepdim=True)
                nxt = tgt(s2).gather(1, a2).squeeze(1)
            else:
                nxt = tgt(s2).max(1).values
            y = r + gamma * (1 - d) * nxt
        td = F.huber_loss(Qs.gather(1, a[:, None]).squeeze(1), y, delta=huber_delta)
        pen = conservative_penalty(Qs, a)
        loss = td + alpha * pen if alpha != 0 else td
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(q.parameters(), 10.0)
        opt.step()
        if tau:
            with torch.no_grad():
                for pt, pq in zip(tgt.parameters(), q.parameters()):
                    pt.mul_(1 - tau).add_(tau * pq)
        elif step % sync_every == 0:
            tgt.load_state_dict(q.state_dict())
        if step % 50 == 0:
            log.append((step, float(td.detach()), float(pen.detach())))
        if step in want:
            snaps[step] = {k: v.clone() for k, v in q.state_dict().items()}
    out = {"snapshots": snaps, "log": log}
    if return_net:
        out["q"], out["tgt"] = q, tgt
    return out


class QPolicy(Policy):
    family = "DQN"

    def __init__(self, name, state_dict, d_in, n_actions, hidden, r_scale, s0, config, family="DQN"):
        super().__init__(name, n_actions, config)
        self.family = family
        self.net = mlp(d_in, n_actions, hidden)
        self.net.load_state_dict(state_dict)
        self.net.eval()
        self.r_scale = r_scale
        with torch.no_grad():
            self.own_value = float(self.net(torch.as_tensor(s0, dtype=torch.float32)).max(1).values.mean()) * r_scale

    def q_values(self, S):
        with torch.no_grad():
            return self.net(torch.as_tensor(np.asarray(S), dtype=torch.float32)).numpy() * self.r_scale

    def greedy(self, S):
        return self.q_values(S).argmax(1)

    def __getstate__(self):
        return self.__dict__
