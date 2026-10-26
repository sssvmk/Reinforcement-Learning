"""Behaviour cloning: MLP classifier on logged actions (cross-entropy)."""
import numpy as np
import torch
import torch.nn.functional as F

from candidates.base import Policy
from candidates.dqn import mlp


def train_bc(S, A, n_actions, hidden=64, epochs=20, lr=1e-3, batch=256, seed=0):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    X = torch.as_tensor(S, dtype=torch.float32)
    y = torch.as_tensor(A, dtype=torch.long)
    net = mlp(X.shape[1], n_actions, hidden)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = len(X)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            loss = F.cross_entropy(net(X[idx]), y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return net


class BCPolicy(Policy):
    family = "BC"

    def __init__(self, name, net, n_actions, stochastic, config):
        super().__init__(name, n_actions, config)
        self.net, self.stochastic = net, stochastic
        self.net.eval()

    def _p(self, S):
        with torch.no_grad():
            return torch.softmax(self.net(torch.as_tensor(np.asarray(S), dtype=torch.float32)), 1).numpy()

    def greedy(self, S):
        return self._p(S).argmax(1)

    def probs(self, S, eps=None):
        if self.stochastic:
            return self._p(S)
        return super().probs(S) if eps is None else super().probs(S, eps)


def bc_metrics(pol: BCPolicy, S, A, behavior_probs):
    """Action accuracy vs. logged actions and mean KL(pi_behavior || pi_BC) over logged states."""
    P = pol._p(S)
    acc = float((P.argmax(1) == A).mean())
    kl = float(np.mean(np.sum(behavior_probs * (np.log(behavior_probs + 1e-12) - np.log(P + 1e-12)), 1)))
    return {"bc_accuracy": acc, "bc_kl_to_behavior": kl}
