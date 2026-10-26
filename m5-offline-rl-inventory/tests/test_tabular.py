import numpy as np
import pytest

from candidates.ce import FittedMDP, ope_value, policy_iteration, value_iteration
from candidates.dynaq import train_dynaq


def random_mdp(nS=5, nA=3, seed=0, deterministic=False):
    rng = np.random.default_rng(seed)
    P = rng.dirichlet(np.ones(nS) * 0.5, size=(nS, nA))
    if deterministic:
        P = np.eye(nS)[rng.integers(nS, size=(nS, nA))]
    R = rng.normal(size=(nS, nA))
    return P, R


def sample_log(P, R, n, seed=0):
    rng = np.random.default_rng(seed)
    nS, nA = R.shape
    s = rng.integers(nS, size=n)
    a = rng.integers(nA, size=n)
    s2 = np.array([rng.choice(nS, p=P[i, j]) for i, j in zip(s, a)]) if n < 5000 else None
    if s2 is None:
        u = rng.random(n)
        s2 = (np.cumsum(P[s, a], 1) < u[:, None]).sum(1)
    return s, a, R[s, a], s2


def exact_dp(P, R, gamma, iters=5000):
    V = np.zeros(R.shape[0])
    for _ in range(iters):
        V = (R + gamma * P @ V).max(1)
    return V


def test_ce_estimates_converge_and_match_dp():
    P, R = random_mdp()
    s, a, r, s2 = sample_log(P, R, 400_000, seed=1)
    mdp = FittedMDP(s, a, r, s2, 5, 3)
    assert np.abs(mdp.P - P).max() < 0.02
    assert np.abs(mdp.R - R).max() < 0.03
    pi, V, _ = value_iteration(mdp, 0.9)
    Vt = exact_dp(P, R, 0.9)
    assert np.abs(V - Vt).max() < 0.2
    pol_true = (R + 0.9 * P @ Vt).argmax(1)
    assert np.array_equal(pi, pol_true)


def test_vi_and_pi_agree():
    P, R = random_mdp(nS=8, nA=4, seed=3)
    s, a, r, s2 = sample_log(P, R, 100_000, seed=2)
    mdp = FittedMDP(s, a, r, s2, 8, 4)
    for kw in [dict(rule="mask", n_min=1), dict(rule="mask", n_min=500), dict(rule="pess", lam=2.0)]:
        pv, Vv, _ = value_iteration(mdp, 0.95, **kw)
        pp, Vp, _ = policy_iteration(mdp, 0.95, **kw)
        assert np.array_equal(pv, pp)
        assert np.allclose(Vv, Vp, atol=1e-6)


def test_mask_rule_falls_back_to_most_frequent_action():
    nS, nA = 2, 3
    s = np.array([0] * 10 + [1] * 10)
    a = np.array([1] * 8 + [2] * 2 + [0] * 10)
    mdp = FittedMDP(s, a, np.ones(20), np.zeros(20, int), nS, nA)
    pi, _, _ = value_iteration(mdp, 0.9, "mask", n_min=9)          # no action of state 0 has >=9 samples... except 1 has 8
    assert pi[0] == 1                                              # most frequent logged action
    assert pi[1] == 0


def test_ope_matches_exact_policy_value():
    P, R = random_mdp(seed=5)
    s, a, r, s2 = sample_log(P, R, 300_000, seed=3)
    mdp = FittedMDP(s, a, r, s2, 5, 3)
    pi = np.full((5, 3), 1 / 3)
    mu0 = np.full(5, 0.2)
    v, _ = ope_value(mdp, pi, 0.9, mu0)
    Ppi = np.einsum("sa,san->sn", pi, P)
    Vt = np.linalg.solve(np.eye(5) - 0.9 * Ppi, (pi * R).sum(1))
    assert v == pytest.approx(mu0 @ Vt, abs=0.1)


def test_dynaq_converges_on_small_mdp():
    P, R = random_mdp(nS=4, nA=2, seed=7, deterministic=True)
    s, a, r, s2 = sample_log(P, R, 4000, seed=4)
    Q, vis = train_dynaq(s, a, r, s2, 4, 2, 0.9, lr=0.2, epochs=60, n_planning=5, seed=0)
    Qt = np.zeros((4, 2))
    for _ in range(3000):
        Qt = R + 0.9 * P @ Qt.max(1)
    assert vis.all()
    assert np.abs(Q - Qt).max() < 0.05


def test_dynaq_nplan0_equals_plain_q_learning():
    P, R = random_mdp(nS=5, nA=3, seed=8)
    s, a, r, s2 = sample_log(P, R, 500, seed=5)
    seed, lr, gamma, epochs = 11, 0.3, 0.9, 4
    Q, _ = train_dynaq(s, a, r, s2, 5, 3, gamma, lr, epochs, 0, seed=seed)
    rng = np.random.default_rng(seed)
    Qr = np.zeros((5, 3))
    for _ in range(epochs):
        for i in rng.permutation(len(s)):
            Qr[s[i], a[i]] += lr * (r[i] + gamma * Qr[s2[i]].max() - Qr[s[i], a[i]])
    assert np.allclose(Q, Qr)
