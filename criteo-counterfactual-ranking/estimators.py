"""Off-policy evaluation estimators for the Criteo ranking logs.

All estimators operate on three parallel per-impression arrays:

    w : importance weight  = pi_e(logged banner | x) / propensity(logged banner | x)
    r : observed reward    = wasAdClicked (0/1)
    c : sampling-correction weight = 10.0 if r==0 else 1.0 (see parser.py)

"c" corrects for the fact that only 10% of non-clicked impressions were
kept in the log; every sum below is a `c`-weighted sum/average, which is
equivalent to re-inflating the dropped non-clicks before averaging.
"""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np


def _clip(w: np.ndarray, clip_m: float = None) -> np.ndarray:
    return w if clip_m is None else np.minimum(w, clip_m)


def ips(w: np.ndarray, r: np.ndarray, c: np.ndarray, clip_m: float = None) -> float:
    """Inverse propensity score estimate of policy value: sum(c*w*r)/sum(c)."""
    w = _clip(w, clip_m)
    return float(np.sum(c * w * r) / np.sum(c))


def snips(w: np.ndarray, r: np.ndarray, c: np.ndarray, clip_m: float = None) -> float:
    """Self-normalized IPS: sum(c*w*r)/sum(c*w). Trades a small bias for
    much lower variance than plain IPS (standard practice for off-policy
    evaluation with noisy propensities)."""
    w = _clip(w, clip_m)
    denom = np.sum(c * w)
    if denom <= 0:
        return float("nan")
    return float(np.sum(c * w * r) / denom)


def effective_sample_size(w: np.ndarray, c: np.ndarray) -> float:
    """Kish's effective sample size applied to the combined weight
    x_i = c_i*w_i used by SNIPS/IPS: ESS = (sum x_i)^2 / sum(x_i^2). By
    Cauchy-Schwarz this is always <= n (equality iff all x_i are equal);
    low ESS relative to n signals an unreliable, high-variance estimate
    (a few impressions dominating the weighted sum)."""
    x = c * w
    num = np.sum(x) ** 2
    den = np.sum(x ** 2)
    if den <= 0:
        return float("nan")
    return float(num / den)


def doubly_robust(
    w: np.ndarray,
    r: np.ndarray,
    c: np.ndarray,
    baseline: np.ndarray,
    r_hat_logged: np.ndarray,
    clip_m: float = None,
) -> float:
    """Doubly robust estimator:

        DR_i = baseline_i + w_i * (r_i - r_hat(x_i, a_i))

    where `baseline_i` = E_{a~pi_e}[r_hat(x_i, a)] (approximated upstream
    by Monte-Carlo sampling banners from the policy and averaging the
    reward model's predictions -- see train.py `estimate_dr_baseline`) and
    `r_hat_logged_i` = r_hat(x_i, a_i), the reward model's prediction for
    the actually-logged banner. DR is consistent if EITHER the reward
    model OR the importance weights are correct, and typically has lower
    variance than plain IPS/SNIPS when the reward model is even roughly
    calibrated.

    We treat each banner as a single joint "slate action" (not per-slot),
    which is the simplest valid DR construction for ranking/slate logs and
    is documented as an assumption in README.md."""
    w = _clip(w, clip_m)
    per_impression = baseline + w * (r - r_hat_logged)
    return float(np.sum(c * per_impression) / np.sum(c))


def bootstrap_ci(
    estimator_fn: Callable[[np.ndarray], float],
    n: int,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """Percentile bootstrap CI for any estimator expressed as a function of
    a (possibly resampled-with-replacement) index array over impressions.
    Returns (point_estimate, lower, upper)."""
    rng = np.random.default_rng(seed)
    point = estimator_fn(np.arange(n))
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = estimator_fn(idx)
    boots = boots[np.isfinite(boots)]
    if boots.size == 0:
        return point, float("nan"), float("nan")
    lower = float(np.percentile(boots, 100 * alpha / 2))
    upper = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return point, lower, upper


def evaluate_policy(
    w: np.ndarray,
    r: np.ndarray,
    c: np.ndarray,
    clip_m: float = 20.0,
    baseline: np.ndarray = None,
    r_hat_logged: np.ndarray = None,
    n_boot: int = 2000,
    seed: int = 0,
) -> Dict[str, object]:
    """Compute the full estimator battery (IPS, SNIPS, clipped IPS/SNIPS,
    DR if a reward model is supplied, ESS, mean weight) with bootstrap CIs,
    for one policy on one (typically test) slice."""
    n = len(w)

    def _ips_fn(idx):
        return ips(w[idx], r[idx], c[idx])

    def _snips_fn(idx):
        return snips(w[idx], r[idx], c[idx])

    def _ips_clip_fn(idx):
        return ips(w[idx], r[idx], c[idx], clip_m=clip_m)

    def _snips_clip_fn(idx):
        return snips(w[idx], r[idx], c[idx], clip_m=clip_m)

    out: Dict[str, object] = {
        "n": n,
        "mean_weight": float(np.mean(w)),
        "ess": effective_sample_size(w, c),
        "ess_fraction": effective_sample_size(w, c) / n if n else float("nan"),
    }
    for name, fn in [("ips", _ips_fn), ("snips", _snips_fn), ("ips_clipped", _ips_clip_fn), ("snips_clipped", _snips_clip_fn)]:
        point, lo, hi = bootstrap_ci(fn, n, n_boot=n_boot, seed=seed)
        out[name] = {"point": point, "ci_low": lo, "ci_high": hi}

    if baseline is not None and r_hat_logged is not None:
        def _dr_fn(idx):
            return doubly_robust(w[idx], r[idx], c[idx], baseline[idx], r_hat_logged[idx], clip_m=clip_m)

        point, lo, hi = bootstrap_ci(_dr_fn, n, n_boot=n_boot, seed=seed)
        out["dr"] = {"point": point, "ci_low": lo, "ci_high": hi}

    return out
