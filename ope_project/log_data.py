"""Behavior policies and logged data. Behavior b is epsilon-soft with known probabilities:
b(a|s) = (1-eps)*1[a = rule(s)] + eps/nA."""
from __future__ import annotations
import numpy as np
from common import Log, new_window_ids


def behavior_prob(taken, rule_action, eps, n_actions):
    return (1.0 - eps) * (taken == rule_action) + eps / n_actions


def sample_starts(lo, hi, H, n, rng):
    """Window start days s with s + H <= hi - 1, so the next-day price exists for the last transition."""
    cand = np.arange(lo, hi - H)
    if len(cand) <= 0:
        raise ValueError("split too short for a window")
    if n <= len(cand):
        return np.sort(rng.choice(cand, size=n, replace=False))
    return np.sort(rng.choice(cand, size=n, replace=True))


def simulate_trading_log(data, lo, hi, n_windows, eps, H, cost_bps, rng, kind="", starts=None):
    """Behavior = follow SAR-trend rule (long iff sar<low) w.p. 1-eps else uniform over {flat, long}."""
    cost = cost_bps / 1e4
    if starts is None:
        starts = sample_starts(lo, hi, H, n_windows, rng)
    W = len(starts)
    raw = np.zeros((W, H + 1, 2))
    act = np.zeros((W, H), dtype=np.int64)
    r = np.zeros((W, H))
    bp = np.zeros((W, H))
    pos = np.zeros(W, dtype=np.int64)
    t = starts.copy()
    raw[:, 0, 0], raw[:, 0, 1] = t, pos
    for k in range(H):
        rule = data.sar_up[t].astype(np.int64)
        explore = rng.random(W) < eps
        a = np.where(explore, rng.integers(0, 2, W), rule)
        bp[:, k] = behavior_prob(a, rule, eps, 2)
        r[:, k] = a * data.fwd_ret[t] - cost * np.abs(a - pos)
        act[:, k] = a
        pos = a
        t = t + 1
        raw[:, k + 1, 0], raw[:, k + 1, 1] = t, pos
    assert t.max() <= hi - 1
    return Log(raw, act, r, bp, new_window_ids(W), starts, kind, eps)


def behavior_value(log, gamma):
    """Empirical discounted H-step return of the behavior policy (its true value on these windows)."""
    disc = gamma ** np.arange(log.H)
    return float((log.r * disc).sum(axis=1).mean())
