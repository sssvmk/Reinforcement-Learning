"""SARSA(lambda) applied to the trading MDP: generalizes SARSA(0) using
eligibility traces (the efficient "backward view" of the lambda-return),
propagating a single step's TD error backward through every (s,a) pair
visited earlier in the episode, weighted by how recently/frequently each
was visited.

lambda=0 reduces to plain SARSA(0); lambda=1 behaves like (episodic) Monte
Carlo control. Traces are reset at the start of every episode (accumulating
traces, as in Silver's Lecture 5).

A constant step size (alpha) is used here for simplicity/vectorized trace
updates across the whole Q-table each step; this is a common practical
simplification (it does not strictly satisfy the Robbins-Monro conditions
used by SARSA(0)'s per-(s,a) decaying alpha, so convergence is empirical
rather than guaranteed -- see README).
"""
import numpy as np

from discretize import NUM_STATES, NUM_ACTIONS, state_to_id, action_bucket_to_shares
from trading_env import TradingEnv


class SARSALambda:
    def __init__(self, gamma: float = 0.99, lam: float = 0.9, alpha: float = 0.05,
                 n0: float = 100.0, seed: int = 0):
        self.gamma = gamma
        self.lam = lam
        self.alpha = alpha
        self.n0 = n0
        self.Q = np.zeros((NUM_STATES, NUM_ACTIONS))
        self.N_s = np.zeros(NUM_STATES)
        self.visit_count_sa = np.zeros((NUM_STATES, NUM_ACTIONS))  # diagnostic only
        self.rng = np.random.default_rng(seed)

    def epsilon_for_state(self, s: int) -> float:
        return self.n0 / (self.n0 + self.N_s[s])

    def select_action(self, s: int) -> int:
        eps = self.epsilon_for_state(s)
        if self.rng.random() < eps:
            return int(self.rng.integers(0, NUM_ACTIONS))
        return int(np.argmax(self.Q[s]))

    def _shares_for(self, env: TradingEnv, obs: dict, a: int) -> float:
        price = obs["close"]
        max_buy = env._max_buy_shares(price)
        max_short = env._max_short_shares(price)
        return action_bucket_to_shares(a, max_buy, max_short)

    def run_episode(self, env: TradingEnv) -> None:
        E = np.zeros((NUM_STATES, NUM_ACTIONS))  # eligibility traces, reset each episode

        obs = env.reset()
        s = state_to_id(obs)
        a = self.select_action(s)
        done = False
        while not done:
            shares = self._shares_for(env, obs, a)
            result = env.step(shares)
            s_next = state_to_id(result.state)
            done = result.done

            self.N_s[s] += 1
            self.visit_count_sa[s, a] += 1

            if done:
                td_target = result.reward
                a_next = None
            else:
                a_next = self.select_action(s_next)
                td_target = result.reward + self.gamma * self.Q[s_next, a_next]

            td_error = td_target - self.Q[s, a]

            E[s, a] += 1.0  # accumulating trace
            self.Q += self.alpha * td_error * E
            E *= self.gamma * self.lam

            obs = result.state
            s = s_next
            if not done:
                a = a_next

    def train(self, env: TradingEnv, n_episodes: int) -> None:
        for _ in range(n_episodes):
            self.run_episode(env)

    def greedy_policy(self) -> np.ndarray:
        return np.argmax(self.Q, axis=1)

    def value_function(self) -> np.ndarray:
        return np.max(self.Q, axis=1)
