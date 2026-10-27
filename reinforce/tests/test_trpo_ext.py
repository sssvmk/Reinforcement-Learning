"""Tests for the extensions: neural TRPO families, entropy bonus, Q_max-free training."""
import numpy as np
import pytest
import policies as P
from trpo.algorithm import trpo
from trpo.kl_fisher import Gauss, Softmax, Bernoulli, NeuralGauss, NeuralBern, family_of
from synth import make_inv, make_trade

torch = pytest.importorskip("torch")


def _neural_cases():
    rng = np.random.default_rng(1)
    pi, pt = P.NeuralGaussInv(hidden=5), P.NeuralBernT(hidden=5)
    ci = (NeuralGauss(pi), pi.init(rng), rng.standard_normal((40, pi.F)), None)
    ct = (NeuralBern(pt), pt.init(rng), rng.standard_normal((40, pt.F)), None)
    return [("I4", ci), ("T3", ct)]


def _act(fam, th, phi, rng):
    return fam.sample(th, phi, rng)


@pytest.mark.parametrize("name", ["I4", "T3"])
def test_neural_kl_fisher_score(name):
    fam, th, phi, _ = dict(_neural_cases())[name]
    rng = np.random.default_rng(2)
    a = _act(fam, th, phi, rng)
    assert fam.kl(th, th, phi) == pytest.approx(0.0, abs=1e-12)
    # score = gradient of logp (finite differences, mean over batch)
    S = fam.score(th, phi, a).mean(axis=0)
    fd = np.array([(fam.logp(th + 1e-6 * e, phi, a).mean() - fam.logp(th - 1e-6 * e, phi, a).mean()) / 2e-6 for e in np.eye(fam.P)[::7]])
    np.testing.assert_allclose(S[::7], fd, rtol=1e-4, atol=1e-6)
    # KL(old, old + eps v) = 0.5 eps^2 v'Fv
    v = rng.standard_normal(fam.P)
    for eps in (1e-3,):
        k = fam.kl(th, th + eps * v, phi)
        assert k == pytest.approx(0.5 * eps ** 2 * float(v @ fam.fvp(th, phi, v)), rel=2e-2)
    # Fisher is symmetric PSD (random vectors) and the analytic diagonal matches e_i' F e_i
    for i in (0, 3, fam.P - 1):
        e = np.zeros(fam.P)
        e[i] = 1.0
        assert fam.diag(th, phi)[i] == pytest.approx(float(e @ fam.fvp(th, phi, e)), rel=1e-8)
    w = rng.standard_normal(fam.P)
    assert float(v @ fam.fvp(th, phi, w)) == pytest.approx(float(w @ fam.fvp(th, phi, v)), rel=1e-8)
    assert float(v @ fam.fvp(th, phi, v)) >= 0


def test_family_of_neural():
    assert isinstance(family_of(P.NeuralGaussInv(hidden=3)), NeuralGauss)
    assert isinstance(family_of(P.NeuralBernT(hidden=3)), NeuralBern)


def _all_fams():
    rng = np.random.default_rng(3)
    out = [(Gauss(4), 0.4 * rng.standard_normal(5), rng.standard_normal((30, 4))),
           (Softmax(3, 4), 0.5 * rng.standard_normal(12), rng.standard_normal((30, 3))),
           (Bernoulli(4), 0.5 * rng.standard_normal(4), rng.standard_normal((30, 4)))]
    for _, (fam, th, phi, _) in _neural_cases():
        out.append((fam, th, phi))
    return out


def test_entropy_gradient_matches_finite_differences():
    for fam, th, phi in _all_fams():
        g = fam.ent_grad(th, phi)
        idx = list(range(0, fam.P, max(1, fam.P // 8))) + [fam.P - 1]
        for i in idx:
            e = np.zeros(fam.P)
            e[i] = 1e-6
            fd = (fam.ent_value(th + e, phi) - fam.ent_value(th - e, phi)) / 2e-6
            assert g[i] == pytest.approx(fd, rel=1e-4, abs=1e-7)


def test_entropy_bonus_raises_sigma_and_keeps_trust_region():
    pr = make_inv(H=20)
    pol = P.GaussInv("I2")
    th0 = pol.init(np.random.default_rng(0))
    kw = dict(B=16, delta=0.02, n_eval=3, diag=False)
    a = trpo(pr, pol, th0, np.random.default_rng(1), 16 * 40, **kw)
    b = trpo(pr, pol, th0, np.random.default_rng(1), 16 * 40, ent_coef=1.0, **kw)
    assert b["theta_final"][-1] > a["theta_final"][-1]            # log sigma larger with a strong entropy bonus
    assert max(r["kl"] for r in b["it_log"]) <= 0.02 + 1e-6
    c = trpo(pr, pol, th0, np.random.default_rng(1), 16 * 40, ent_coef=1.0, **kw)
    np.testing.assert_array_equal(b["theta_final"], c["theta_final"])      # deterministic


def test_ent_coef_zero_is_unchanged():
    pr = make_trade(H=20)
    pol = P.BernoulliT()
    th0 = pol.init(np.random.default_rng(0))
    a = trpo(pr, pol, th0, np.random.default_rng(1), 8 * 20, B=8, n_eval=2, diag=False)
    b = trpo(pr, pol, th0, np.random.default_rng(1), 8 * 20, B=8, n_eval=2, diag=False, ent_coef=0.0)
    np.testing.assert_array_equal(a["theta_final"], b["theta_final"])


def test_neural_trpo_respects_kl_and_runs():
    for pr, pol in ((make_inv(H=20), P.NeuralGaussInv(hidden=4)), (make_trade(H=20), P.NeuralBernT(hidden=4))):
        th0 = pol.init(np.random.default_rng(0))
        r = trpo(pr, pol, th0, np.random.default_rng(1), 8 * 12, B=8, delta=0.01, n_eval=3, diag=True, diag_mult=1)
        assert np.all(np.isfinite(r["theta_final"]))
        acc = [x for x in r["it_log"] if x.get("accepted")]
        assert acc and all(x["kl"] <= 0.01 + 1e-6 for x in acc)
        assert r["episodes_used"] <= 8 * 12


def test_neural_trpo_exact_vs_cg_direction():
    from trpo.algorithm import natural_direction
    fam, th, phi, _ = dict(_neural_cases())["T3"]
    rng = np.random.default_rng(5)
    g = rng.standard_normal(fam.P)
    fvp = lambda v: fam.fvp(th, phi, v)
    s_cg = natural_direction(fvp, g, fam.P, 0.1, 200)[0]
    s_ex = natural_direction(fvp, g, fam.P, 0.1, None, exact=True)[0]
    assert float(s_cg @ s_ex / (np.linalg.norm(s_cg) * np.linalg.norm(s_ex))) > 0.99


def test_qmax_free_config_and_uncapped_orders(monkeypatch):
    from common import load_config
    monkeypatch.setenv("QMAX_FREE", "1")
    assert load_config()["inventory"]["q_max_mult"] is None
    pr = make_inv(H=20, qmax=None)
    pol = P.GaussInv("I1")
    th = pol.init(np.random.default_rng(0)).copy()
    th[0] = 30.0                                                        # asks for a base stock of 30 mu: must not be capped
    d = pr.run(pol, th, pr.grid("val", 20), np.random.default_rng(0), "det", detail=True)
    assert d["order"].max() > 7 * 8.0
    pr2 = make_inv(H=20, qmax=7 * 8.0)
    d2 = pr2.run(pol, th, pr2.grid("val", 20), np.random.default_rng(0), "det", detail=True)
    assert d2["order"].max() <= 7 * 8.0 + 1e-9
    r = trpo(pr, pol, pol.init(np.random.default_rng(0)), np.random.default_rng(1), 8 * 15, B=8, n_eval=2, diag=False)
    assert np.all(np.isfinite(r["theta_final"]))
