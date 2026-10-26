"""Common candidate-policy interface."""
import json
import pickle
from pathlib import Path

import numpy as np

SOFT_EPS = 0.1   # epsilon of the epsilon-soft version used by importance-sampling estimators


class Policy:
    """act(state) -> greedy action index (accepts one state or a batch).
    probs(S) -> (N, A) action probabilities of the *evaluated* stochastic version:
      - deterministic candidates: epsilon-soft  (1-eps)*onehot(greedy) + eps/A  (eps=SOFT_EPS)
      - stochastic candidates (BC-sampling): their own softmax.
    eval_probs(S): what FQE evaluates (greedy one-hot, or the policy's own softmax when stochastic)."""
    family = "ref"
    stochastic = False
    own_value = float("nan")

    def __init__(self, name, n_actions, config=None):
        self.name, self.n_actions, self.config = name, n_actions, dict(config or {})

    def greedy(self, S):
        raise NotImplementedError

    def act(self, s):
        s = np.asarray(s)
        a = self.greedy(s[None] if s.ndim == 1 else s)
        return int(a[0]) if s.ndim == 1 else a

    def probs(self, S, eps=SOFT_EPS):
        a = self.greedy(S)
        P = np.full((len(a), self.n_actions), eps / self.n_actions)
        P[np.arange(len(a)), a] += 1 - eps
        return P

    def eval_probs(self, S):
        if self.stochastic:
            return self.probs(S)
        a = self.greedy(S)
        P = np.zeros((len(a), self.n_actions))
        P[np.arange(len(a)), a] = 1.0
        return P

    def sample(self, S, rng, soft=True):
        P = self.probs(S) if soft else self.eval_probs(S)
        u = rng.random(len(P))[:, None]
        return (np.cumsum(P, 1) < u).sum(1).clip(0, self.n_actions - 1)

    def save(self, out_dir):
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        meta = {"name": self.name, "family": self.family, **{k: (v if isinstance(v, (int, float, str, bool, list, dict, type(None))) else str(v)) for k, v in self.config.items()}}
        (d / f"{self.name}.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
        with open(d / f"{self.name}.pkl", "wb") as f:
            pickle.dump(self, f)
