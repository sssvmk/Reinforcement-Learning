import os

import numpy as np
import pytest

from parser import parse_impressions
from policy import PlackettLucePolicy
from train import load_impressions, policy_weights, temporal_split

SAMPLE_PATH = r"C:\Users\username\Downloads\sample.txt"


def _toy_phis(n, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    phis = []
    for i in range(n):
        k = rng.integers(2, 4)
        idxs = rng.choice(dim, size=k, replace=False)
        vals = rng.normal(size=k)
        phis.append({int(ix): float(v) for ix, v in zip(idxs, vals)})
    return phis


@pytest.mark.skipif(not os.path.exists(SAMPLE_PATH), reason="real Criteo sample.txt not found at expected path")
def test_temporal_split_no_overlap_and_strictly_ordered_exid():
    impressions = load_impressions(SAMPLE_PATH)
    train, val, test = temporal_split(impressions)

    assert len(train) + len(val) + len(test) == len(impressions)

    train_ids, val_ids, test_ids = (
        {imp.exid for imp in train},
        {imp.exid for imp in val},
        {imp.exid for imp in test},
    )
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    assert max(train_ids) <= min(val_ids)
    assert max(val_ids) <= min(test_ids)

    for split in (train, val, test):
        exids = [imp.exid for imp in split]
        assert exids == sorted(exids)


@pytest.mark.skipif(not os.path.exists(SAMPLE_PATH), reason="real Criteo sample.txt not found at expected path")
def test_non_click_subsample_weight_changes_ctr_estimate():
    impressions = load_impressions(SAMPLE_PATH)
    raw_ctr = sum(imp.reward for imp in impressions) / len(impressions)
    corrected_ctr = sum(imp.weight * imp.reward for imp in impressions) / sum(imp.weight for imp in impressions)
    # Because only clicked impressions have weight 1 and non-clicks have
    # weight 10, correcting for the 10% non-click subsampling must push the
    # corrected CTR well below the raw (uncorrected) CTR.
    assert corrected_ctr < raw_ctr
    assert corrected_ctr == pytest.approx(raw_ctr / (raw_ctr + 10 * (1 - raw_ctr)))


def test_logging_policy_weight_is_one_when_target_equals_logging_policy():
    """Sanity check #1: if the evaluated (target) policy IS the logging
    policy -- i.e. its Plackett-Luce probability of the logged banner
    equals the recorded propensity exactly -- every importance weight must
    equal 1. We can't recover the real hidden logging-policy scores, so we
    validate the mechanism synthetically: build a policy, compute its exact
    PL probability of a banner, and set that banner's 'propensity' to
    exactly that probability."""
    import math

    from parser import Candidate, Impression

    dim = 16
    policy = PlackettLucePolicy(dim=dim, temperature=1.0, seed=0)
    policy.theta = np.random.default_rng(7).normal(size=dim)

    impressions = []
    phis_all = []
    for i in range(10):
        phis = _toy_phis(6, dim=dim, seed=100 + i)
        nb_slots = 3
        log_scores = policy.log_scores(phis)
        true_prob = math.exp(policy.log_prob_logged_banner(log_scores, nb_slots))
        imp = Impression(
            exid=i,
            hash_id=f"h{i}",
            reward=0,
            propensity=true_prob,
            nb_slots=nb_slots,
            nb_candidates=6,
            display_feats={},
            candidates=[Candidate(clicked=0, feats={}) for _ in range(6)],
        )
        impressions.append(imp)
        phis_all.append(phis)

    w = policy_weights(policy, impressions, phis_all)
    assert np.allclose(w, 1.0, atol=1e-6)
