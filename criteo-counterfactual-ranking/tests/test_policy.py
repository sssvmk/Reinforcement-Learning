import numpy as np
import pytest

from policy import PlackettLucePolicy


def _toy_phis(n, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    phis = []
    for i in range(n):
        # 2-3 random nonzero hashed slots per candidate
        k = rng.integers(2, 4)
        idxs = rng.choice(dim, size=k, replace=False)
        vals = rng.normal(size=k)
        phis.append({int(ix): float(v) for ix, v in zip(idxs, vals)})
    return phis


def test_pl_probabilities_of_all_permutations_sum_to_one():
    dim = 16
    policy = PlackettLucePolicy(dim=dim, temperature=1.0, seed=0)
    policy.theta = np.random.default_rng(1).normal(size=dim)
    phis = _toy_phis(4, dim=dim, seed=2)
    log_scores = policy.log_scores(phis)

    for nb_slots in (1, 2, 3, 4):
        logps = policy.log_prob_all_permutations(log_scores, nb_slots)
        total = sum(np.exp(lp) for lp in logps.values())
        assert total == pytest.approx(1.0, abs=1e-6)


def test_prefix_formula_matches_brute_force_for_logged_order():
    """log_prob_logged_banner (fast, prefix-based) must agree with the
    brute-force log_prob_all_permutations for the specific permutation
    (0, 1, ..., nb_slots-1), which is how the logged banner is encoded."""
    dim = 12
    policy = PlackettLucePolicy(dim=dim, temperature=0.7, seed=0)
    policy.theta = np.random.default_rng(3).normal(size=dim)
    phis = _toy_phis(5, dim=dim, seed=4)
    log_scores = policy.log_scores(phis)

    for nb_slots in (1, 2, 3, 4, 5):
        fast = policy.log_prob_logged_banner(log_scores, nb_slots)
        brute = policy.log_prob_all_permutations(log_scores, nb_slots)
        expected_perm = tuple(range(nb_slots))
        assert fast == pytest.approx(brute[expected_perm], abs=1e-9)


def test_sample_banner_matches_analytic_distribution_chi_square():
    """Empirical frequency of sampled permutations should match the
    analytic PL probabilities within chi-square tolerance."""
    from scipy.stats import chisquare

    dim = 8
    policy = PlackettLucePolicy(dim=dim, temperature=1.0, seed=42)
    policy.theta = np.random.default_rng(5).normal(size=dim)
    phis = _toy_phis(3, dim=dim, seed=6)
    nb_slots = 2
    log_scores = policy.log_scores(phis)
    analytic = policy.log_prob_all_permutations(log_scores, nb_slots)
    perms = list(analytic.keys())
    analytic_probs = np.array([np.exp(analytic[p]) for p in perms])

    n_samples = 20000
    counts = {p: 0 for p in perms}
    for _ in range(n_samples):
        banner = tuple(policy.sample_banner(phis, nb_slots))
        counts[banner] += 1
    observed = np.array([counts[p] for p in perms], dtype=np.float64)
    expected = analytic_probs * n_samples

    # Merge any near-zero-expected bins to keep the chi-square test well
    # posed (shouldn't occur here since there are only 3*2=6 permutations
    # with non-trivial analytic probability).
    assert (expected > 5).all()
    stat, pvalue = chisquare(observed, f_exp=expected)
    assert pvalue > 0.01, f"chi-square test failed: stat={stat}, p={pvalue}"


def test_uniform_policy_has_zero_theta_and_equal_scores():
    policy = PlackettLucePolicy.uniform(dim=10, seed=0)
    phis = _toy_phis(4, dim=10, seed=1)
    log_scores = policy.log_scores(phis)
    assert np.allclose(log_scores, 0.0)
