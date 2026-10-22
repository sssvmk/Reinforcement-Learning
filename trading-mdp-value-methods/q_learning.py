"""Off-policy TD control: Q-learning applied to the trading MDP.

Like SARSA(0), Q-learning updates Q after every single step using a TD
target -- but instead of bootstrapping from the action the current
epsilon-greedy policy actually selects next (on-policy), it bootstraps from
the GREEDY action at s', i.e. max_a' Q(s', a') -- directly sampling the
Bellman OPTIMALITY equation regardless of what action is taken next. This
makes Q-learning off-policy: the behavior policy (epsilon-greedy, for
exploration) and the target policy (greedy, implicit in the max) can differ,
while Q still provably converges to Q*.

Step size: alpha(s,a) = 1/N(s,a), satisfying the Robbins-Monro conditions
used for SARSA(0) (Lecture 5's "Unifying Summary": Q-learning is the sampled
version of value iteration, exactly as SARSA is the sampled version of
policy iteration's evaluation step).
"""
import numpy as np

from discretize import NUM_STATES, NUM_ACTIONS, state_to_id, action_bucket_to_shares
from trading_env import TradingEnv


class QLearning:
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
        """Behavior policy: epsilon-greedy (exploration)."""
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
        done = False
        while not done:
            s = state_to_id(obs)
            a = self.select_action(s)  # behavior policy chooses the actual action taken
            shares = self._shares_for(env, obs, a)
            result = env.step(shares)
            s_next = state_to_id(result.state)
            done = result.done

            self.N_s[s] += 1
            self.N_sa[s, a] += 1
            alpha = 1.0 / self.N_sa[s, a]

            if done:
                td_target = result.reward  # episode ends -- no bootstrap past it
            else:
                # Target policy is GREEDY regardless of what the behavior policy
                # will actually pick next -- this is what makes Q-learning off-policy.
                td_target = result.reward + self.gamma * np.max(self.Q[s_next])

            self.Q[s, a] += alpha * (td_target - self.Q[s, a])
            obs = result.state

    def train(self, env: TradingEnv, n_episodes: int) -> None:
        for _ in range(n_episodes):
            self.run_episode(env)

    def greedy_policy(self) -> np.ndarray:
        return np.argmax(self.Q, axis=1)

    def value_function(self) -> np.ndarray:
        return np.max(self.Q, axis=1)
