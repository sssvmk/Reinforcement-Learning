"""Discretization of the continuous trading state/action into finite buckets,
so a tabular certainty-equivalence (fitted MDP) model can be fit.
"""
import numpy as np


def rsi_bucket(rsi: float) -> int:
    if rsi < 30:
        return 0  # oversold
    if rsi > 70:
        return 2  # overbought
    return 1      # neutral


def trend_bucket(close: float, psar: float) -> int:
    return 1 if close > psar else 0  # 1 = uptrend (price above PSAR), 0 = downtrend


def position_bucket(shares_held: float) -> int:
    if shares_held > 0:
        return 2  # long
    if shares_held < 0:
        return 0  # short
    return 1      # flat


def state_to_id(obs: dict) -> int:
    r = rsi_bucket(obs["rsi"])
    tr = trend_bucket(obs["close"], obs["psar"])
    p = position_bucket(obs["shares_held"])
    # 3 (rsi) x 2 (trend) x 3 (position) = 18 discrete states
    return r * 6 + tr * 3 + p


NUM_STATES = 3 * 2 * 3

# Discrete action buckets, expressed as a fraction of the max allowed order size
# for the given state (sign applied by the bucket itself).
ACTION_FRACTIONS = [-1.0, -0.5, 0.0, 0.5, 1.0]
NUM_ACTIONS = len(ACTION_FRACTIONS)


def action_bucket_to_shares(bucket: int, max_buy: float, max_sell_short: float) -> float:
    frac = ACTION_FRACTIONS[bucket]
    if frac >= 0:
        return frac * max_buy
    return frac * max_sell_short  # max_sell_short is a positive magnitude; frac is negative
