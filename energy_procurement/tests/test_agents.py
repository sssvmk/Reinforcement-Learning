import os
import sys
import numpy as np
import pytest
from scipy import stats
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents import newsvendor as nv, cvar_sp, mean_variance as mv, forecast as F
import costs
from risk import cvar


# ---------- test 5: newsvendor ----------
def test_saa_matches_analytic_critical_fractile():
    rng = np.random.default_rng(0)
    S, pDA, pA, rho = 200000, 50.0, 60.0, 0.0
    Ls = rng.lognormal(np.log(10), 0.1, S)
    tau = (pA - pDA) / ((pA - pDA) + (pDA - rho * pA))
    q = nv.saa_quantity(Ls[None], np.full((1, S), pA), np.array([pDA]), np.array([1e9]), rho, 0.0)[0]
    exact = np.exp(np.log(10) + 0.1 * stats.norm.ppf(tau))
    assert q == pytest.approx(exact, rel=5e-3)


def test_variant_A_returns_analytic_quantile_and_regimes():
    class P: pass
    p = P()
    p.rho, p.pi = 0.0, 0.0
    D = 1
    p.pDA = np.full((D, 24), 50.0)
    p.Lhat = np.full((D, 24), 10.0)
    p.cap = 1.5 * p.Lhat
    sc = P()
    sc.cg = np.full((D, 24), 10.0)
    sig = 0.05
    qres = np.exp(sig * stats.norm.ppf(F.LEVELS)) - 1
    sc.Q = np.tile(qres, (D, 24, 1))
    q, tau = nv.newsvendor_A(p, sc, [0])
    t = 10 / 60
    assert tau[0, 0] == pytest.approx(t)
    assert q[0, 0] == pytest.approx(10 * np.exp(sig * stats.norm.ppf(t)), rel=5e-3)
    sc.cg = np.full((D, 24), -5.0)  # Cu <= 0 -> zero
    assert (nv.newsvendor_A(p, sc, [0])[0] == 0).all()
    p.rho = 1.2  # Co <= 0 -> cap
    sc.cg = np.full((D, 24), 10.0)
    p.pDA = np.full((D, 24), 50.0)
    assert (nv.newsvendor_A(p, sc, [0])[0] == p.cap).all()


def test_saa_regimes():
    L = np.random.default_rng(0).uniform(5, 15, (1, 100))
    z = nv.saa_quantity(L, np.full((1, 100), 40.0), np.array([50.0]), np.array([20.0]), 0.0, 0.0)  # pA<pDA -> Cu<0
    assert z[0] == 0.0
    c = nv.saa_quantity(L, np.full((1, 100), 80.0), np.array([50.0]), np.array([20.0]), 1.0, 0.0)  # Co<0 -> cap
    assert c[0] == 20.0


# ---------- test 6: CVaR LP ----------
def _tiny(S=8, seed=0):
    r = np.random.default_rng(seed)
    pDA = np.full(24, 50.0)
    Ls = r.uniform(8, 12, (S, 24))
    pAs = pDA + r.uniform(-5, 20, (S, 24))
    return Ls, pAs, pDA, np.full(24, 18.0)


def _cvar_obj(q, Ls, pAs, pDA, rho, pi, lam, alpha, Cref):
    C = costs.daily_cost(q, Ls, pDA, pAs, rho, pi) / Cref
    return C.mean() + lam * cvar(C, alpha)


def test_lp_equals_bruteforce_one_hour():
    # one active hour (others have zero load and zero cap) -> brute-force grid over q
    Ls, pAs, pDA, cap = _tiny(S=10)
    cap = np.zeros(24); cap[0] = 18.0
    Ls[:, 1:] = 0.0
    Cref = 500.0
    lam, alpha = 2.0, 0.8
    res = cvar_sp.solve_day(Ls, pAs, pDA, cap, Cref, 0.0, 0.0, (lam,), alpha)[lam][0]
    grid = np.linspace(0, 18, 3601)
    vals = []
    for g in grid:
        q = np.zeros((1, 24)); q[0, 0] = g
        vals.append(_cvar_obj(q, Ls, pAs, pDA, 0.0, 0.0, lam, alpha, Cref))
    assert _cvar_obj(res[None], Ls, pAs, pDA, 0.0, 0.0, lam, alpha, Cref) <= min(vals) + 1e-6


def test_lp_lambda0_equals_risk_neutral_and_monotone():
    Ls, pAs, pDA, cap = _tiny(S=40)
    Cref = 12000.0
    lams = (0.0, 1.0, 4.0, 16.0)
    out = cvar_sp.solve_day(Ls, pAs, pDA, cap, Cref, 0.0, 0.0, lams, 0.9)
    saa = nv.saa_quantity(Ls.T[None], pAs.T[None], pDA[None], cap[None], 0.0, 0.0)[0]
    assert np.allclose(out[0.0][0], saa, atol=1e-6)
    means, cv = [], []
    for lam in lams:
        C = costs.daily_cost(out[lam][0], Ls, pDA, pAs) / Cref
        means.append(C.mean()); cv.append(cvar(C, 0.9))
    assert all(np.diff(cv) <= 1e-9) and all(np.diff(means) >= -1e-9)


# ---------- test 7: mean-variance ----------
def test_mv_numeric_matches_closed_form_gaussian():
    r = np.random.default_rng(1)
    S, H = 20000, 24
    pDA = np.full(H, 50.0)
    g = 8 + 4 * r.standard_normal((S, H)) + 3 * r.standard_normal((S, 1))
    pAs = pDA + g
    Ls = 10 + 0.5 * r.standard_normal((S, H)) + 0.3 * g * 0.1
    for kappa in (0.5, 5.0):
        qc = mv.closed_form_linear(Ls, pAs, pDA, kappa)
        a = (pAs * Ls).sum(1)
        f = lambda q: ((a - (g * q).sum(1)).mean() + kappa * (a - (g * q).sum(1)).var())
        r_ = minimize(f, np.full(H, 10.0), method="BFGS", options={"gtol": 1e-9})
        assert np.allclose(qc, r_.x, rtol=1e-2, atol=1e-2)


def test_mv_frontier_monotone():
    Ls, pAs, pDA, cap = _tiny(S=60)
    Cref = 12000.0
    q = np.full(24, 10.0)
    stds, means = [], []
    for k in (0, 5, 50, 500):
        q = mv.mv_day(Ls, pAs, pDA, cap, 0.0, 0.0, k, Cref, q)
        C = costs.daily_cost(q, Ls, pDA, pAs) / Cref
        means.append(C.mean()); stds.append(C.std())
    assert all(np.diff(stds) <= 1e-6) and all(np.diff(means) >= -1e-6)


# ---------- test 8: forecasting ----------
def test_quantiles_noncrossing_and_pinball_minimized_by_true_quantile():
    Q = np.random.default_rng(0).standard_normal((50, 24, len(F.LEVELS)))
    assert (np.diff(F.repair(Q), axis=-1) >= 0).all()
    y = np.random.default_rng(1).standard_normal(200000)
    for t in (0.1, 0.5, 0.9):
        qs = np.linspace(-2, 2, 81)
        loss = [F.pinball(y, q, t).mean() for q in qs]
        assert qs[int(np.argmin(loss))] == pytest.approx(stats.norm.ppf(t), abs=0.1)


def test_gbm_calibration_on_synthetic():
    r = np.random.default_rng(0)
    n = 6000
    X = r.uniform(-1, 1, (n, 2))
    y = X[:, 0] + (0.3 + 0.3 * np.abs(X[:, 1])) * r.standard_normal(n)
    m = F.QuantileGBM(levels=F.LEVELS, max_iter=60).fit(X[:4000], y[:4000])
    Q = m.predict(X[4000:])
    cov = [(y[4000:] <= Q[:, k]).mean() for k in range(len(F.LEVELS))]
    assert np.abs(np.array(cov) - F.LEVELS).max() < 0.06
    assert (np.diff(Q, axis=-1) >= 0).all()
