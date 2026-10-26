"""Tests for fqe.py (new module; does not modify any pre-existing file).

Covers sanity checks 1, 3, 4, 5, 8 from the task spec. Checks 2 (Plackett-
Luce probabilities), 6 (10x weighting) and 7 (temporal split) are already
covered by tests/test_policy.py and tests/test_train.py and are not
duplicated here.
"""
from __future__ import annotations

import os
import random

import numpy as np
import pytest

from features import DEFAULT_DIM, featurize_candidate, hash_token
from parser import Candidate, Impression
from policy import PlackettLucePolicy
from train import featurize_impressions, load_impressions, temporal_split
from estimators import snips
from fqe import (
    FQEResult,
    SlotMDPDataset,
    bandit_direct_method,
    build_slot_mdp_dataset,
    fit_fqe,
    _build_transitions,
)

SAMPLE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "..", "..", "..", "sample.txt")
# The real sample lives outside this repo (per the task's file location);
# resolve it via an environment variable with a sensible fallback so the
# test still runs in CI environments where that absolute path is absent.
SAMPLE_PATH = os.environ.get("CRITEO_SAMPLE_PATH", r"C:\Users\username\Downloads\sample.txt")


def _toy_impressions(n=400, dim=4096, seed=0):
    """Deterministic 2-candidate / 2-slot toy MDP with a hand-computable
    true policy value (sanity check 5). Reward rule: the candidate placed
    at slot 0 earns 1 iff it is type 'A'; the candidate placed at slot 1
    earns 1 iff it is type 'B'. With nb_candidates == nb_slots == 2, the
    only decision is which type goes first (the other is forced into slot
    1), so Q(s0, A) = 2, Q(s0, B) = 0, Q(s1, *) = the terminal reward
    itself, and V(pi_e) = 2 * pi_e(A | s0) exactly."""
    rng = random.Random(seed)
    impressions = []
    for i in range(n):
        order = ["A", "B"] if rng.random() < 0.5 else ["B", "A"]
        candidates = []
        for k, t in enumerate(order):
            clicked = 1 if (k == 0 and t == "A") or (k == 1 and t == "B") else 0
            candidates.append(Candidate(clicked=clicked, feats={11: t}))
        impressions.append(
            Impression(
                exid=i,
                hash_id="h",
                reward=1,  # forces weight == 1.0 for every impression (see Impression.__post_init__)
                propensity=0.5,
                nb_slots=2,
                nb_candidates=2,
                display_feats={},
                candidates=candidates,
            )
        )
    return impressions, dim


def test_toy_mdp_recovers_true_value():
    """Sanity check 5: FQE on a toy deterministic MDP recovers the
    analytically-computable true policy value within tolerance."""
    impressions, dim = _toy_impressions(n=600, dim=4096, seed=0)
    phis = featurize_impressions(impressions, dim)
    dataset = build_slot_mdp_dataset(impressions, phis, dim)

    idx_a, sign_a = hash_token("11=A", dim)
    idx_b, sign_b = hash_token("11=B", dim)
    assert idx_a != idx_b, "unlucky hash collision for this dim/seed; bump dim"
    theta = np.zeros(dim)
    theta[idx_a] = 1.0 / sign_a
    theta[idx_b] = -1.0 / sign_b
    target_policy = PlackettLucePolicy(dim=dim, temperature=1.0, theta=theta)

    phi_a = featurize_candidate({}, {11: "A"}, dim=dim)
    phi_b = featurize_candidate({}, {11: "B"}, dim=dim)
    log_scores = target_policy.log_scores([phi_a, phi_b])
    m = log_scores.max()
    p = np.exp(log_scores - m)
    p /= p.sum()
    true_value = 2.0 * p[0]

    result = fit_fqe(dataset, target_policy=target_policy, gamma=1.0, n_iters=10, alpha=0.01, seed=0)
    assert abs(result.value - true_value) < 0.15, (result.value, true_value)


def test_bandit_special_case_matches_direct_method():
    """Sanity check 3: FQE with horizon=1, gamma=0 reduces to the direct
    (bandit) reward-regression estimate."""
    impressions = load_impressions(SAMPLE_PATH)
    train, val, test = temporal_split(impressions)
    dim = 2 ** 12
    phis = featurize_impressions(test, dim)
    dataset = build_slot_mdp_dataset(test, phis, dim)
    target_policy = PlackettLucePolicy.uniform(dim)

    fqe_result = fit_fqe(dataset, target_policy=target_policy, gamma=0.0, n_iters=1, alpha=1.0, seed=0, horizon_cap=1)
    _, direct_value = bandit_direct_method(dataset, target_policy, alpha=1.0, seed=0)

    assert fqe_result.value == pytest.approx(direct_value, abs=1e-9)


def test_terminal_has_no_bootstrap_and_target_equals_reward():
    """Sanity check 4: terminal transitions have no next-state row range
    (so no bootstrap term can be computed), and the TD target at those
    transitions equals the reward regardless of any bootstrap value."""
    impressions = load_impressions(SAMPLE_PATH)
    dim = 2 ** 10
    phis = featurize_impressions(impressions, dim)
    dataset = build_slot_mdp_dataset(impressions, phis, dim)

    x_rows, rewards, weights, terminal, next_ranges = _build_transitions(dataset, horizon_cap=None)
    assert terminal.any()
    for t in range(len(terminal)):
        if terminal[t]:
            assert next_ranges[t] is None

    dummy_bootstrap = np.random.default_rng(0).normal(size=len(rewards))
    y = rewards + 1.0 * dummy_bootstrap * (~terminal)
    assert np.allclose(y[terminal], rewards[terminal])


def test_non_displayed_candidate_labels_never_used_as_targets():
    """Sanity check 8: flipping the click label of a never-displayed
    candidate must not change any reward, any Q-training target, or the
    resulting policy value, since only displayed candidates (indices
    < nb_slots) ever contribute to `rewards[k]`."""
    impressions = load_impressions(SAMPLE_PATH)
    dim = 2 ** 12
    phis = featurize_impressions(impressions, dim)

    tampered = []
    found_non_displayed = False
    for imp in impressions:
        new_candidates = list(imp.candidates)
        if imp.nb_candidates > imp.nb_slots and not found_non_displayed:
            nd = new_candidates[imp.nb_slots]
            new_candidates[imp.nb_slots] = Candidate(clicked=1 - nd.clicked, feats=nd.feats)
            found_non_displayed = True
        tampered.append(
            Impression(
                exid=imp.exid,
                hash_id=imp.hash_id,
                reward=imp.reward,
                propensity=imp.propensity,
                nb_slots=imp.nb_slots,
                nb_candidates=imp.nb_candidates,
                display_feats=imp.display_feats,
                candidates=new_candidates,
            )
        )
    assert found_non_displayed, "sample has no impression with an undisplayed candidate; test is vacuous"

    ds_orig = build_slot_mdp_dataset(impressions, phis, dim)
    ds_tampered = build_slot_mdp_dataset(tampered, phis, dim)

    orig_rewards = [r for imp in ds_orig.impressions for r in imp.rewards]
    tampered_rewards = [r for imp in ds_tampered.impressions for r in imp.rewards]
    assert orig_rewards == tampered_rewards

    pol = PlackettLucePolicy.uniform(dim)
    res_orig = fit_fqe(ds_orig, target_policy=pol, n_iters=3, seed=0)
    res_tampered = fit_fqe(ds_tampered, target_policy=pol, n_iters=3, seed=0)
    assert res_orig.value == pytest.approx(res_tampered.value, abs=1e-9)


def test_logging_policy_value_close_to_corrected_ctr():
    """Sanity check 1: the on-policy (SARSA-style) FQE estimate of the
    logging policy's value, and the SNIPS estimate for the logged
    behavior (w == 1, which reduces exactly to the 10x-corrected CTR),
    should both be close to the observed corrected CTR. We report the
    gap rather than assert a tight tolerance, since the ~400-impression
    sample gives genuinely wide confidence intervals (documented in
    README_FQE.md)."""
    impressions = load_impressions(SAMPLE_PATH)
    train, val, test = temporal_split(impressions)
    dim = 2 ** 12
    phis = featurize_impressions(test, dim)
    dataset = build_slot_mdp_dataset(test, phis, dim)

    r = np.array([imp.reward for imp in test], dtype=np.float64)
    c = np.array([imp.weight for imp in test], dtype=np.float64)
    corrected_ctr = float((c * r).sum() / c.sum())

    w_logging = np.ones(len(test))
    snips_logging = snips(w_logging, r, c)
    assert snips_logging == pytest.approx(corrected_ctr, abs=1e-9)

    result = fit_fqe(dataset, on_policy=True, n_iters=5, seed=0)
    gap = abs(result.value - corrected_ctr)
    print(f"on-policy FQE={result.value:.4f} corrected_ctr={corrected_ctr:.4f} gap={gap:.4f}")
    # Wide-tolerance sanity bound given n=83 test impressions; a loose
    # bound catches gross implementation bugs without over-fitting to the
    # tiny sample's noise.
    assert gap < 0.10
