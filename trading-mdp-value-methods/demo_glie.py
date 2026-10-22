"""End-to-end GLIE Monte Carlo Control demo on the trading MDP:
1. Generate synthetic OHLCV data (swap in real historical data for real use).
2. Train GLIEMonteCarloControl directly from sampled episodes (model-free --
   no transition/reward model is ever fit, unlike demo.py's certainty-
   equivalence approach).
3. Report the learned Q-table coverage, V(s) = max_a Q(s,a), and the greedy
   policy, and compare against the certainty-equivalence fitted-MDP result.
"""
import numpy as np

from demo import make_synthetic_ohlcv
from trading_env import TradingEnv
from glie_mc_control import GLIEMonteCarloControl
from discretize import NUM_STATES, NUM_ACTIONS


def main():
    market_df = make_synthetic_ohlcv(n=1500)
    env = TradingEnv(market_df, initial_shares=5000.0, initial_cash=100_000.0, initial_avg_entry=300.0)

    agent = GLIEMonteCarloControl(gamma=0.99, n0=100.0, seed=0)
    n_episodes = 20_000
    agent.train(env, n_episodes=n_episodes)

    visited = int((agent.N_sa > 0).sum())
    print(f"Trained on {n_episodes} episodes (5-day each).")
    print(f"Visited (s,a) pairs: {visited} / {NUM_STATES * NUM_ACTIONS}")

    print("\nLearned V(s) = max_a Q(s,a):")
    print(np.round(agent.value_function(), 2))

    print("\nGreedy policy (action bucket index per state, [-1,-0.5,0,0.5,1] x max size):")
    print(agent.greedy_policy())

    print("\nFinal epsilon per state (GLIE decay, N0=100):")
    eps = np.array([agent.epsilon_for_state(s) for s in range(NUM_STATES)])
    print(np.round(eps, 4))

    print("\nVisit counts N(s,a):")
    print(agent.N_sa.astype(int))


if __name__ == "__main__":
    main()
