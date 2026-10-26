"""Offline Dyna-Q (NumPy/pure Python).

Offline adaptation: 'real experience' is replayed from a fixed log (several epochs, random order); a tabular
model stores observed (s,a)->(r,s'); n_planning extra updates per real step sample previously observed (s,a)
from that model. Dyna-Q+ exploration bonuses are NOT applicable offline and are skipped.
The greedy policy is restricted to actions observed in the log for that state (support masking)."""
import random

import numpy as np

from candidates.base import Policy


def train_dynaq(s, a, r, s2, nS, nA, gamma, lr=0.1, epochs=5, n_planning=0, model="last", seed=0):
    rng = np.random.default_rng(seed)
    rp = random.Random(seed)
    Q = [[0.0] * nA for _ in range(nS)]
    M = {}            # (s,a) -> (r, s') or list of them
    keys = []
    s, a, r, s2 = (np.asarray(x) for x in (s, a, r, s2))
    n = len(s)
    for _ in range(epochs):
        order = rng.permutation(n)
        for i in order:
            si, ai, ri, s2i = int(s[i]), int(a[i]), float(r[i]), int(s2[i])
            q2 = max(Q[s2i])                           # truncation is never terminal: always bootstrap
            Q[si][ai] += lr * (ri + gamma * q2 - Q[si][ai])
            k = (si, ai)
            if model == "last":
                if k not in M:
                    keys.append(k)
                M[k] = (ri, s2i)
            else:
                if k not in M:
                    keys.append(k)
                    M[k] = []
                M[k].append((ri, s2i))
            for _ in range(n_planning):
                kk = keys[rp.randrange(len(keys))]
                v = M[kk]
                rr, ss = v if model == "last" else v[rp.randrange(len(v))]
                Q[kk[0]][kk[1]] += lr * (rr + gamma * max(Q[ss]) - Q[kk[0]][kk[1]])
    visited = np.zeros((nS, nA), bool)
    visited[s, a] = True
    return np.array(Q), visited


def train_q_learning_reference(s, a, r, s2, nS, nA, gamma, lr, epochs, seed):
    """Plain offline Q-learning with the same replay order (used by tests)."""
    return train_dynaq(s, a, r, s2, nS, nA, gamma, lr, epochs, 0, seed=seed)[0]


class DynaQPolicy(Policy):
    family = "DynaQ"

    def __init__(self, name, disc, Q, visited, mu0_disc, config):
        super().__init__(name, disc.p.n_actions, config)
        self.disc, self.Q = disc, Q
        Qm = np.where(visited, Q, -np.inf)
        self.has = visited.any(1)
        self.pi = np.where(self.has, Qm.argmax(1), int(visited.sum(0).argmax()))
        self.own_value = float(np.mean(np.max(np.where(visited, Q, -np.inf), 1)[mu0_disc]))

    def greedy(self, S):
        return self.pi[self.disc(S)]
