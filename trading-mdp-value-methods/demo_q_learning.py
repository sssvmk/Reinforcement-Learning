"""End-to-end demo for Q-learning on the trading MDP, using the same
synthetic OHLCV data and TradingEnv as the other demos. Also reports the
SARSA(0) result side-by-side to highlight the on-policy vs. off-policy
difference (Lecture 5's canonical contrast).
"""
import numpy as np

from demo import make_synthetic_ohlcv
from trading_env import TradingEnv
from discretize import NUM_STATES, NUM_ACTIONS
from q_learning import QLearning
from sarsa import SARSA


def main():
    market_df = make_synthetic_ohlcv(n=1500)
    env = TradingEnv(market_df, initial_shares=5000.0, initial_cash=100_000.0, initial_avg_entry=300.0)

    n_episodes = 20_000

    q_agent = QLearning(gamma=0.99, n0=100.0, seed=0)
    q_agent.train(env, n_episodes=n_episodes)

    print("=== Q-learning (off-policy) ===")
    print(f"Visited (s,a) pairs: {int((q_agent.N_sa > 0).sum())} / {NUM_STATES * NUM_ACTIONS}")
    print("V(s) = max_a Q(s,a):")
    print(np.round(q_agent.value_function(), 2))
    print("Greedy policy:")
    print(q_agent.greedy_policy())

    sarsa_agent = SARSA(gamma=0.99, n0=100.0, seed=0)
    sarsa_agent.train(env, n_episodes=n_episodes)

    print("\n=== SARSA(0) (on-policy, for comparison) ===")
    print(f"Visited (s,a) pairs: {int((sarsa_agent.N_sa > 0).sum())} / {NUM_STATES * NUM_ACTIONS}")
    print("V(s) = max_a Q(s,a):")
    print(np.round(sarsa_agent.value_function(), 2))
    print("Greedy policy:")
    print(sarsa_agent.greedy_policy())

    diff_states = np.where(q_agent.greedy_policy() != sarsa_agent.greedy_policy())[0]
    print(f"\nStates where Q-learning and SARSA(0) greedy policies differ: {diff_states.tolist()}")


if __name__ == "__main__":
    main()
