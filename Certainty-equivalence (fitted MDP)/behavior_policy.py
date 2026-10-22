"""Simple rule-based behavior policy (RSI + PSAR) used to generate logged
(s, a, r, s') experience, since no real historical trade log is available.
This stands in for 'the behavior policy that generated the offline dataset'.
"""
import numpy as np

from discretize import NUM_ACTIONS, ACTION_FRACTIONS


def rsi_psar_policy(obs: dict, rng: np.random.Generator) -> int:
    """Returns a discrete action bucket index (see discretize.ACTION_FRACTIONS)."""
    rsi = obs["rsi"]
    uptrend = obs["close"] > obs["psar"]

    if rsi < 30 and uptrend:
        base = 4   # strong buy
    elif rsi < 40:
        base = 3   # mild buy
    elif rsi > 70 and not uptrend:
        base = 0   # strong sell/short
    elif rsi > 60:
        base = 1   # mild sell
    else:
        base = 2   # hold

    # small exploration noise so the logged dataset covers neighboring buckets too
    if rng.random() < 0.15:
        base = int(np.clip(base + rng.choice([-1, 1]), 0, NUM_ACTIONS - 1))
    return base
