"""End-to-end demo:
1. Generate synthetic OHLCV data (replace with real historical data in practice).
2. Compute RSI + PSAR.
3. Roll out the RSI/PSAR behavior policy through TradingEnv to build an offline
   (s, a, r, s') dataset.
4. Fit the certainty-equivalence MDP and run value iteration / policy evaluation.
"""
import numpy as np
import pandas as pd

from indicators import compute_rsi, compute_psar
from trading_env import TradingEnv
from discretize import state_to_id, action_bucket_to_shares, NUM_ACTIONS
from behavior_policy import rsi_psar_policy
from fitted_mdp import FittedMDP


def make_synthetic_ohlcv(n: int = 1000, start_price: float = 300.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(loc=0.0002, scale=0.01, size=n)
    close = start_price * np.exp(np.cumsum(returns))
    high = close * (1 + rng.uniform(0.0, 0.01, size=n))
    low = close * (1 - rng.uniform(0.0, 0.01, size=n))
    open_ = np.roll(close, 1)
    open_[0] = start_price
    volume = rng.integers(1_000, 10_000, size=n).astype(float)
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})
    df["rsi"] = compute_rsi(df["close"])
    df["psar"] = compute_psar(df["high"], df["low"], df["close"])
    return df


def collect_offline_dataset(env: TradingEnv, market_df: pd.DataFrame, n_episodes: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    mdp = FittedMDP(gamma=0.99)
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        while not done:
            s_id = state_to_id(obs)
            a_bucket = rsi_psar_policy(obs, rng)
            price = obs["close"]
            max_buy = env._max_buy_shares(price)
            max_short = env._max_short_shares(price)
            shares = action_bucket_to_shares(a_bucket, max_buy, max_short)
            result = env.step(shares)
            s_next_id = state_to_id(result.state)
            mdp.add_transition(s_id, a_bucket, result.reward, s_next_id)
            obs = result.state
            done = result.done
    return mdp


def main():
    market_df = make_synthetic_ohlcv(n=1500)
    env = TradingEnv(market_df, initial_shares=5000.0, initial_cash=100_000.0, initial_avg_entry=300.0)

    mdp = collect_offline_dataset(env, market_df, n_episodes=300)
    mdp.fit()

    V_opt, Q_opt, greedy_policy = mdp.value_iteration()
    print("Visited (s,a) pairs:", int((mdp.sa_counts > 0).sum()), "/", mdp.sa_counts.size)
    print("\nFitted-MDP optimal value function V*(s) by discrete state id:")
    print(np.round(V_opt, 2))
    print("\nGreedy action bucket per state (index into [-1,-0.5,0,0.5,1] x max size):")
    print(greedy_policy)

    # Certainty-equivalence evaluation of the behavior policy itself (not just control):
    behavior_action_of_state = greedy_policy.copy()
    for s in range(len(behavior_action_of_state)):
        if mdp.sa_counts[s].sum() == 0:
            continue
        behavior_action_of_state[s] = int(np.argmax(mdp.sa_counts[s]))  # most-taken action in that state
    V_behavior = mdp.evaluate_policy(behavior_action_of_state)
    print("\nFitted-MDP evaluation of the (approximate) behavior policy V_pi(s):")
    print(np.round(V_behavior, 2))


if __name__ == "__main__":
    main()
