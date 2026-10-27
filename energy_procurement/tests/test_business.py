import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metrics import block_bootstrap_ci, bootstrap_ci, holm, agent_metrics
import costs


class FakeP:
    """Minimal problem with a hand-computable cost."""
    rho, pi, Cref = 0.0, 0.0, 1000.0

    def __init__(self, D=60, seed=0):
        rng = np.random.default_rng(seed)
        self.Lhat = np.full((D, 24), 10.0)
        self.L = self.Lhat * (1 + 0.02 * rng.normal(size=(D, 24)))
        self.pDA = np.full((D, 24), 50.0)
        self.pA = self.pDA + 5 + rng.normal(size=(D, 24))
        import pandas as pd
        self.dates = pd.date_range("2018-01-01", periods=D)

    def cost(self, q, idx=None):
        idx = slice(None) if idx is None else idx
        return costs.daily_cost(q, self.L[idx], self.pDA[idx], self.pA[idx], self.rho, self.pi)


def test_cost_accounting_identity_and_annualisation():
    P = FakeP()
    idx = np.arange(60)
    q = P.Lhat * 0.97
    ref = P.Lhat
    m = agent_metrics(P, q, idx, ref)
    by_hand = ((P.pDA * q + P.pA * np.maximum(P.L - q, 0)).sum(1)).mean()
    assert m["mean_eur"] == pytest.approx(by_hand)
    assert m["purchase_eur"] + m["shortfall_eur"] + m["surplus_credit_eur"] == pytest.approx(m["mean_eur"])
    assert m["savings_eur_year"] == pytest.approx(365 * m["savings_eur_day"])
    assert m["savings_pct"] == pytest.approx(100 * m["savings_eur_day"] / P.cost(ref, idx).mean())


def test_block_bootstrap_contains_point_and_is_wider_than_iid_for_autocorrelated_series():
    rng = np.random.default_rng(1)
    x = np.convolve(rng.normal(size=400), np.ones(10) / 10, "same") + 1.0  # strongly autocorrelated
    pt, lo, hi = block_bootstrap_ci(x, 7, 1500)
    _, lo1, hi1 = bootstrap_ci(x, 1500)
    assert lo <= pt <= hi
    assert (hi - lo) > (hi1 - lo1)


def test_holm_is_monotone_and_dominates_raw_p():
    p = np.array([0.001, 0.02, 0.04, 0.5])
    adj = holm(p)
    assert np.all(adj >= p) and np.all(adj <= 1)
    assert adj[0] == pytest.approx(0.004) and adj[1] == pytest.approx(0.06)
    assert np.all(np.diff(adj[np.argsort(p)]) >= 0)


def test_savings_sign_conventions_against_reference():
    P = FakeP()
    idx = np.arange(60)
    better = np.clip(P.L, 0, None)  # perfect-foresight purchase is cheaper than the forecast when the gap is positive
    m = agent_metrics(P, better, idx, P.Lhat)
    worse = agent_metrics(P, np.zeros_like(P.L), idx, P.Lhat)
    assert m["savings_eur_day"] > 0 > worse["savings_eur_day"]
