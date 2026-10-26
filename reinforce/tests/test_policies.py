"""Tests 1-6: score functions vs numerical differentiation, entropy gradients, softmax/Bernoulli identities."""
import numpy as np
import pytest
import policies as P
from synth import make_inv, make_trade


def _aux_state(pr, pol, rng, theta, N=6):
    """Draw actions at some state, then return a function theta -> mean logp for the SAME stored actions."""
    starts = pr.sample("train", N, rng)
    if pr.domain == "inventory":
        st = {"ip": rng.uniform(0, 30, N), "oh": rng.uniform(0, 20, N), "t": starts, "j": 0, "env": pr.env}
    else:
        st = {"t": starts, "prev": rng.integers(0, 2, N).astype(float), "j": 0, "env": pr.env}
    _, aux = pol.act(theta, st, rng, "sample")
    return st, aux


def _num_grad(f, th, h=1e-6):
    g = np.zeros_like(th)
    for i in range(len(th)):
        e = np.zeros_like(th)
        e[i] = h
        g[i] = (f(th + e) - f(th - e)) / (2 * h)
    return g


def _check_score(pr, pol, seed=0):
    rng = np.random.default_rng(seed)
    th = pol.init(rng) + 0.1 * rng.standard_normal(pol.n_params)
    st, aux = _aux_state(pr, pol, rng, th)
    S = pol.score(th, aux)
    for i in range(S.shape[0]):
        # logp for row i only with the stored action
        def f(t, i=i):
            return pol.logp(t, aux)[i]
        np.testing.assert_allclose(S[i], _num_grad(f, th), rtol=1e-4, atol=1e-5)


def test_gaussian_I1_score():
    _check_score(make_inv(), P.GaussInv("I1"))


def test_gaussian_I2_score_incl_logsigma():
    pol = P.GaussInv("I2")
    _check_score(make_inv(), pol, seed=3)


def test_softmax_inventory_score():
    _check_score(make_inv(), P.SoftmaxInv())


def test_bernoulli_score():
    _check_score(make_trade(), P.BernoulliT())


def test_softmax_trading_score():
    _check_score(make_trade(), P.SoftmaxT())


def test_neural_scores():
    _check_score(make_inv(), P.NeuralGaussInv(), seed=4)
    _check_score(make_trade(), P.NeuralBernT(), seed=5)


def test_neural_entropy_gradient():
    rng = np.random.default_rng(2)
    pr, pol = make_trade(), P.NeuralBernT()
    th = pol.init(rng) + 0.1 * rng.standard_normal(pol.n_params)
    st, aux = _aux_state(pr, pol, rng, th)
    G = pol.ent_grad(th, aux)

    def ent(t):
        p = P.sigmoid(pol.net.forward(t, aux[0])[0])
        return -(p * np.log(p) + (1 - p) * np.log(1 - p))
    for i in range(G.shape[0]):
        np.testing.assert_allclose(G[i], _num_grad(lambda t: ent(t)[i], th), rtol=1e-4, atol=1e-6)


def test_entropy_gradients_numerical():
    rng = np.random.default_rng(1)
    prT, prI = make_trade(), make_inv()
    # Bernoulli entropy H(p) and softmax entropy; Gaussian entropy gradient w.r.t. log sigma is 1
    for pr, pol in ((prT, P.BernoulliT()), (prT, P.SoftmaxT()), (prI, P.SoftmaxInv())):
        th = pol.init(rng) + 0.1 * rng.standard_normal(pol.n_params)
        st, aux = _aux_state(pr, pol, rng, th)
        G = pol.ent_grad(th, aux)

        def ent(t):
            if isinstance(pol, P.BernoulliT):
                p = P.sigmoid(aux[0] @ t)
                return -(p * np.log(p) + (1 - p) * np.log(1 - p))
            p = P.softmax(aux[0] @ t.reshape(pol.F, pol.A))
            return -(p * np.log(p)).sum(axis=1)
        for i in range(G.shape[0]):
            ng = _num_grad(lambda t: ent(t)[i], th)
            np.testing.assert_allclose(G[i], ng, rtol=1e-4, atol=1e-6)
    g = P.GaussInv("I1").ent_grad(np.zeros(11), (np.ones((3, 10)), np.zeros(3), np.zeros(3), 1.0))
    assert np.all(g[:, -1] == 1) and np.all(g[:, :-1] == 0)
