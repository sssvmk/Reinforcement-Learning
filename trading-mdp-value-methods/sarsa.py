"""On-policy TD control: SARSA(0) applied to the trading MDP.

Unlike GLIE Monte Carlo control (glie_mc_control.py), SARSA updates Q after
every single step using the TD target r + gamma*Q(s',a') -- it does not wait
for the episode to finish, and it bootstraps from the current Q estimate of
the action actually selected next ("SARSA" = State-Action-Reward-State-Action).

Step size: alpha(s,a) = 1/N(s,a), satisfying the Robbins-Monro conditions
(sum alpha = inf, sum alpha^2 < inf) that, together with GLIE exploration,
guarantee convergence to Q* in the tabular setting (Lecture 5).
"""
import numpy as np

from discretize import NUM_STATES, NUM_ACTIONS, state_to_id, action_bucket_to_shares
from trading_env import TradingEnv


class SARSA:
    def __init__(self, gamma: float = 0.99, n0: float = 100.0, seed: int = 0):
        self.gamma = gamma
        self.n0 = n0
        self.Q = np.zeros((NUM_STATES, NUM_ACTIONS))
        self.N_sa = np.zeros((NUM_STATES, NUM_ACTIONS))
        self.N_s = np.zeros(NUM_STATES)
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
            self.N_sa[s, a] += 1
            alpha = 1.0 / self.N_sa[s, a]

            if done:
                td_target = result.reward  # episode ends -- no bootstrap past it
                a_next = None
            else:
                a_next = self.select_action(s_next)
                td_target = result.reward + self.gamma * self.Q[s_next, a_next]

            self.Q[s, a] += alpha * (td_target - self.Q[s, a])

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
