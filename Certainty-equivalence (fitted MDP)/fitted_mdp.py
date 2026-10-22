"""Certainty-equivalence (fitted MDP) evaluation:
1. Fit an MLE transition model P(s'|s,a) and average reward R(s,a) from logged
   (s, a, r, s') tuples collected under the behavior policy.
2. Run dynamic-programming policy evaluation / value iteration on the fitted
   tabular MDP to estimate V and Q.
"""
import numpy as np

from discretize import NUM_STATES, NUM_ACTIONS


class FittedMDP:
    def __init__(self, gamma: float = 0.99):
        self.gamma = gamma
        self.counts = np.zeros((NUM_STATES, NUM_ACTIONS, NUM_STATES))
        self.reward_sum = np.zeros((NUM_STATES, NUM_ACTIONS))
        self.sa_counts = np.zeros((NUM_STATES, NUM_ACTIONS))

    def add_transition(self, s: int, a: int, r: float, s_next: int) -> None:
        self.counts[s, a, s_next] += 1
        self.reward_sum[s, a] += r
        self.sa_counts[s, a] += 1

    def fit(self):
        """Builds P(s'|s,a) and R(s,a) from accumulated counts.
        Unvisited (s,a) pairs default to a self-loop with zero reward.
        """
        p = np.zeros_like(self.counts)
        r = np.zeros_like(self.reward_sum)
        for s in range(NUM_STATES):
            for a in range(NUM_ACTIONS):
                n = self.sa_counts[s, a]
                if n > 0:
                    p[s, a] = self.counts[s, a] / n
                    r[s, a] = self.reward_sum[s, a] / n
                else:
                    p[s, a, s] = 1.0
                    r[s, a] = 0.0
        self.P = p
        self.R = r
        return p, r

    def value_iteration(self, tol: float = 1e-6, max_iters: int = 10_000):
        V = np.zeros(NUM_STATES)
        for _ in range(max_iters):
            Q = self.R + self.gamma * np.einsum("san,n->sa", self.P, V)
            V_new = Q.max(axis=1)
            if np.max(np.abs(V_new - V)) < tol:
                V = V_new
                break
            V = V_new
        policy = Q.argmax(axis=1)
        return V, Q, policy

    def evaluate_policy(self, action_of_state: np.ndarray, tol: float = 1e-6, max_iters: int = 10_000):
        """Certainty-equivalence POLICY EVALUATION for a fixed (given) policy,
        as opposed to control/value-iteration above.
        """
        V = np.zeros(NUM_STATES)
        for _ in range(max_iters):
            V_new = np.array([
                self.R[s, action_of_state[s]]
                + self.gamma * self.P[s, action_of_state[s]].dot(V)
                for s in range(NUM_STATES)
            ])
            if np.max(np.abs(V_new - V)) < tol:
                V = V_new
                break
            V = V_new
        return V
