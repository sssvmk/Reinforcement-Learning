"""GLIE Monte Carlo Control applied to the trading MDP (OHLCV + RSI + PSAR
state, discretized as in discretize.py), reusing the same TradingEnv built
for the certainty-equivalence example.

GLIE = Greedy in the Limit with Infinite Exploration:
  - epsilon_k -> 0 as the (state-)visit count k -> infinity
  - every (state, action) pair is still visited infinitely often in the limit
Under these two conditions, first-visit Monte Carlo control provably
converges to Q* (and the greedy policy derived from it converges to an
optimal policy) -- directly from sampled episodes, with no model of the
environment's dynamics or rewards required (unlike the certainty-equivalence
approach, which fits P(s'|s,a)/R(s,a) first).
"""
import numpy as np

from discretize import NUM_STATES, NUM_ACTIONS, state_to_id, action_bucket_to_shares
from trading_env import TradingEnv


class GLIEMonteCarloControl:
    def __init__(self, gamma: float = 0.99, n0: float = 100.0, seed: int = 0):
        self.gamma = gamma
        self.n0 = n0
        self.Q = np.zeros((NUM_STATES, NUM_ACTIONS))
        self.N_sa = np.zeros((NUM_STATES, NUM_ACTIONS))   # visit counts per (s,a)
        self.N_s = np.zeros(NUM_STATES)                    # visit counts per state (drives epsilon)
        self.episode_count = 0
        self.rng = np.random.default_rng(seed)

    def epsilon_for_state(self, s: int) -> float:
        # State-dependent decaying epsilon (as in Silver's Blackjack GLIE example):
        # frequently-visited states explore less, guaranteeing epsilon -> 0 overall
        # while every (s,a) pair is still visited infinitely often in the limit.
        return self.n0 / (self.n0 + self.N_s[s])

    def select_action(self, s: int) -> int:
        eps = self.epsilon_for_state(s)
        if self.rng.random() < eps:
            return int(self.rng.integers(0, NUM_ACTIONS))
        return int(np.argmax(self.Q[s]))

    def generate_episode(self, env: TradingEnv):
        """Rolls out one full episode under the current epsilon-greedy policy.
        Returns a list of (state_id, action, reward) transitions.
        """
        obs = env.reset()
        trajectory = []
        done = False
        while not done:
            s_id = state_to_id(obs)
            a = self.select_action(s_id)
            price = obs["close"]
            max_buy = env._max_buy_shares(price)
            max_short = env._max_short_shares(price)
            shares = action_bucket_to_shares(a, max_buy, max_short)
            result = env.step(shares)
            trajectory.append((s_id, a, result.reward))
            obs = result.state
            done = result.done
        return trajectory

    def update_from_episode(self, trajectory) -> None:
        """First-visit MC update: compute the discounted return following the
        first occurrence of each (s,a) pair in this episode, then apply an
        incremental-mean update to Q(s,a).
        """
        T = len(trajectory)
        returns = np.zeros(T)
        g = 0.0
        for t in reversed(range(T)):
            _, _, r = trajectory[t]
            g = r + self.gamma * g
            returns[t] = g

        seen = set()
        for t, (s, a, _) in enumerate(trajectory):
            if (s, a) in seen:
                continue
            seen.add((s, a))
            self.N_sa[s, a] += 1
            self.N_s[s] += 1
            self.Q[s, a] += (returns[t] - self.Q[s, a]) / self.N_sa[s, a]

        self.episode_count += 1

    def train(self, env: TradingEnv, n_episodes: int) -> None:
        for _ in range(n_episodes):
            trajectory = self.generate_episode(env)
            self.update_from_episode(trajectory)

    def greedy_policy(self) -> np.ndarray:
        return np.argmax(self.Q, axis=1)

    def value_function(self) -> np.ndarray:
        return np.max(self.Q, axis=1)
