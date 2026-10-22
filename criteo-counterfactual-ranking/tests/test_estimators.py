import numpy as np
import pytest

from estimators import (
    bootstrap_ci,
    doubly_robust,
    effective_sample_size,
    ips,
    snips,
)


def test_ips_matches_manual_formula():
    w = np.array([1.0, 2.0, 0.5])
    r = np.array([1.0, 0.0, 1.0])
    c = np.array([1.0, 10.0, 1.0])
    expected = (1 * 1.0 * 1.0 + 10 * 2.0 * 0.0 + 1 * 0.5 * 1.0) / (1 + 10 + 1)
    assert ips(w, r, c) == pytest.approx(expected)


def test_snips_matches_manual_formula():
    w = np.array([1.0, 2.0, 0.5])
    r = np.array([1.0, 0.0, 1.0])
    c = np.array([1.0, 10.0, 1.0])
    expected_num = 1 * 1.0 * 1.0 + 10 * 2.0 * 0.0 + 1 * 0.5 * 1.0
    expected_den = 1 * 1.0 + 10 * 2.0 + 1 * 0.5
    assert snips(w, r, c) == pytest.approx(expected_num / expected_den)


def test_weight_all_ones_gives_snips_equal_to_corrected_ctr():
    """Sanity check #1 (algebraic form): if the evaluated policy equals the
    logging policy exactly, w==1 for every impression, and SNIPS collapses
    to exactly the 10x-subsample-corrected CTR -- i.e. the logging-policy
    'off-policy' evaluation of itself must recover the observed corrected
    CTR, not some other number."""
    rng = np.random.default_rng(0)
    r = rng.integers(0, 2, size=50).astype(np.float64)
    c = np.where(r == 0, 10.0, 1.0)
    w = np.ones(50)
    corrected_ctr = np.sum(c * r) / np.sum(c)
    assert snips(w, r, c) == pytest.approx(corrected_ctr)
    assert ips(w, r, c) == pytest.approx(corrected_ctr)


def test_clipping_reduces_influence_of_large_weights():
    w = np.array([1.0, 1000.0])
    r = np.array([0.0, 1.0])
    c = np.array([1.0, 1.0])
    unclipped = ips(w, r, c)
    clipped = ips(w, r, c, clip_m=5.0)
    assert clipped < unclipped


def test_effective_sample_size_equals_n_when_weights_uniform():
    w = np.ones(20)
    c = np.ones(20)
    assert effective_sample_size(w, c) == pytest.approx(20.0)


def test_effective_sample_size_shrinks_with_skewed_weights():
    w = np.array([1.0] * 19 + [100.0])
    c = np.ones(20)
    ess = effective_sample_size(w, c)
    assert ess < 20.0


def test_effective_sample_size_never_exceeds_n_with_nonuniform_c():
    """Cauchy-Schwarz guarantees ESS <= n for ANY non-negative c, w -- this
    must hold even when c itself is highly non-uniform (e.g. the 10x/1x
    subsample-correction weights used throughout this project), which a
    naive 'sum(c*w^2)' denominator (instead of 'sum((c*w)^2)') would
    violate."""
    rng = np.random.default_rng(2)
    n = 50
    w = np.ones(n)
    c = np.where(rng.random(n) < 0.9, 10.0, 1.0)
    ess = effective_sample_size(w, c)
    assert ess <= n + 1e-9


def test_doubly_robust_reduces_to_baseline_when_reward_model_perfect():
    """If r_hat_logged == r exactly (a perfect reward model on the logged
    action), the correction term w*(r - r_hat) vanishes and DR collapses to
    the plain average of the baseline term."""
    w = np.array([2.0, 3.0, 0.1])
    r = np.array([1.0, 0.0, 1.0])
    c = np.array([1.0, 1.0, 1.0])
    baseline = np.array([0.3, 0.2, 0.4])
    r_hat_logged = r.copy()
    dr = doubly_robust(w, r, c, baseline, r_hat_logged)
    assert dr == pytest.approx(np.mean(baseline))


def test_bootstrap_ci_contains_point_estimate_and_is_ordered():
    rng = np.random.default_rng(1)
    w = rng.uniform(0.5, 2.0, size=100)
    r = rng.integers(0, 2, size=100).astype(np.float64)
    c = np.ones(100)

    def fn(idx):
        return snips(w[idx], r[idx], c[idx])

    point, lo, hi = bootstrap_ci(fn, n=100, n_boot=500, seed=0)
    assert lo <= point <= hi
