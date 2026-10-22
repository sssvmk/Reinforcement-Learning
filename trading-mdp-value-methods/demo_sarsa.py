"""End-to-end demo for SARSA(0) and SARSA(lambda) on the trading MDP,
using the same synthetic OHLCV data and TradingEnv as the other demos.
"""
import numpy as np

from demo import make_synthetic_ohlcv
from trading_env import TradingEnv
from discretize import NUM_STATES, NUM_ACTIONS
from sarsa import SARSA
from sarsa_lambda import SARSALambda


def run_sarsa(env, n_episodes=20_000):
    agent = SARSA(gamma=0.99, n0=100.0, seed=0)
    agent.train(env, n_episodes=n_episodes)
    return agent


def run_sarsa_lambda(env, n_episodes=20_000, lam=0.9):
    agent = SARSALambda(gamma=0.99, lam=lam, alpha=0.05, n0=100.0, seed=0)
    agent.train(env, n_episodes=n_episodes)
    return agent


def main():
    market_df = make_synthetic_ohlcv(n=1500)
    env = TradingEnv(market_df, initial_shares=5000.0, initial_cash=100_000.0, initial_avg_entry=300.0)

    n_episodes = 20_000

    sarsa_agent = run_sarsa(env, n_episodes)
    print("=== SARSA(0) ===")
    print(f"Visited (s,a) pairs: {int((sarsa_agent.N_sa > 0).sum())} / {NUM_STATES * NUM_ACTIONS}")
    print("V(s) = max_a Q(s,a):")
    print(np.round(sarsa_agent.value_function(), 2))
    print("Greedy policy:")
    print(sarsa_agent.greedy_policy())

    sarsa_lambda_agent = run_sarsa_lambda(env, n_episodes, lam=0.9)
    print("\n=== SARSA(lambda=0.9) ===")
    print(f"Visited (s,a) pairs: {int((sarsa_lambda_agent.visit_count_sa > 0).sum())} / {NUM_STATES * NUM_ACTIONS}")
    print("V(s) = max_a Q(s,a):")
    print(np.round(sarsa_lambda_agent.value_function(), 2))
    print("Greedy policy:")
    print(sarsa_lambda_agent.greedy_policy())


if __name__ == "__main__":
    main()
