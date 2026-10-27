"""Tests 1-10, 13 of the TRPO prompt: KL, Fisher, conjugate gradient, trust-region step, surrogate gradient, GAE, value function,
convergence, known optimum, floor projection / determinism / accounting / VPG direction, learning a profitable rule."""
import numpy as np
import pytest
import policies as P
import inventory_env as ie
import problems as pb
from trpo import algorithm as alg
from trpo.algorithm import trpo, line_search, natural_direction, policy_gradient, surrogate
from trpo.conjugate_gradient import conjugate_gradient
from trpo.gae import gae
from trpo.kl_fisher import Gauss, Softmax, Bernoulli, family_of, full_fisher
from trpo.value import ValueFn, returns_to_go
from reinforce.algorithm import grad_estimate
from synth import make_inv, make_trade


def _fams():
    rng = np.random.default_rng(0)
    return [(Gauss(4), 0.4 * rng.standard_normal(5), "g"), (Softmax(3, 4), 0.5 * rng.standard_normal(12), "s"), (Bernoulli(4), 0.5 * rng.standard_normal(4), "b")]


def _phi(rng, F, M=1):
    return np.hstack([np.ones((M, 1)), rng.standard_normal((M, F - 1))])


def _numeric_kl(fam, th_o, th_n, phi):
    """Direct sum / quadrature for one state."""
    if fam.kind == "gauss":
        m0, s0 = fam.dist(th_o, phi)
        m1, s1 = fam.dist(th_n, phi)
        x = np.linspace(m0[0] - 12 * s0, m0[0] + 12 * s0, 200001)
        p0 = np.exp(-0.5 * ((x - m0[0]) / s0) ** 2) / (s0 * np.sqrt(2 * np.pi))
        p1 = np.exp(-0.5 * ((x - m1[0]) / s1) ** 2) / (s1 * np.sqrt(2 * np.pi))
        return float(np.trapezoid(p0 * np.log(p0 / np.maximum(p1, 1e-300)), x))
    if fam.kind == "softmax":
        return float(sum(np.exp(fam.logp(th_o, phi, np.array([a]))) * (fam.logp(th_o, phi, np.array([a])) - fam.logp(th_n, phi, np.array([a]))) for a in range(fam.A))[0])
    return float(sum(np.exp(fam.logp(th_o, phi, np.array([a]))) * (fam.logp(th_o, phi, np.array([a])) - fam.logp(th_n, phi, np.array([a]))) for a in (0.0, 1.0))[0])


# ---------------------------------------------------------------- 1. KL
def test_kl_matches_numerical_zero_and_nonnegative():
    rng = np.random.default_rng(1)
    for fam, th, _ in _fams():
        F = fam.F
        phi = _phi(rng, F)
        assert fam.kl(th, th, phi) == pytest.approx(0.0, abs=1e-12)
        for _ in range(5):
            th2 = th + 0.3 * rng.standard_normal(len(th))
            k = fam.kl(th, th2, phi)
            assert k >= -1e-12
            assert k == pytest.approx(_numeric_kl(fam, th, th2, phi), rel=2e-4, abs=1e-8)


def test_gaussian_kl_changes_mean_and_logsigma_separately():
    fam, th = Gauss(2), np.array([0.2, 0.1, np.log(0.5)])
    phi = np.array([[1.0, 2.0]])
    th_m = th.copy()
    th_m[0] += 0.5
    th_s = th.copy()
    th_s[2] += 0.4
    assert fam.kl(th, th_m, phi) == pytest.approx(_numeric_kl(fam, th, th_m, phi), rel=1e-4)
    assert fam.kl(th, th_s, phi) == pytest.approx(_numeric_kl(fam, th, th_s, phi), rel=1e-4)


# ---------------------------------------------------------------- 2. Fisher
def test_fisher_equals_kl_hessian_and_score_outer_product():
    rng = np.random.default_rng(2)
    for fam, th, _ in _fams():
        phi = _phi(rng, fam.F)
        Fa = fam.fisher_state(th, phi[0])
        # Hessian of KL(old || theta) at theta_old by central differences of the exact KL
        P_, h = fam.P, 1e-3
        Hh = np.zeros((P_, P_))
        f = lambda t: fam.kl(th, t, phi)
        for i in range(P_):
            for j in range(P_):
                ei, ej = np.eye(P_)[i] * h, np.eye(P_)[j] * h
                Hh[i, j] = (f(th + ei + ej) - f(th + ei - ej) - f(th - ei + ej) + f(th - ei - ej)) / (4 * h * h)
        np.testing.assert_allclose(Hh, Fa, rtol=2e-3, atol=2e-5)
        # Monte Carlo E[score score^T] with actions drawn from the policy itself
        a = fam.sample(th, np.repeat(phi, 200000, axis=0), rng)
        S = fam.score(th, np.repeat(phi, 200000, axis=0), a)
        np.testing.assert_allclose(S.T @ S / len(S), Fa, rtol=0.06, atol=0.02)
        # Fisher-vector product equals the batch mean of per-state Fisher matrices
        Pb = _phi(rng, fam.F, 7)
        v = rng.standard_normal(fam.P)
        Fm = np.mean([fam.fisher_state(th, r) for r in Pb], axis=0)
        np.testing.assert_allclose(fam.fvp(th, Pb, v), Fm @ v, rtol=1e-9, atol=1e-12)


# ---------------------------------------------------------------- 3. CG
def test_cg_solves_spd_and_matches_direct_and_damping():
    rng = np.random.default_rng(3)
    for n in (3, 8, 20):
        Q = rng.standard_normal((n, n))
        Fm = Q @ Q.T + 0.5 * np.eye(n)
        g = rng.standard_normal(n)
        x, hist = conjugate_gradient(lambda v: Fm @ v, g, iters=n + 5, tol=1e-18)
        np.testing.assert_allclose(x, np.linalg.solve(Fm, g), rtol=1e-7, atol=1e-9)
        assert hist[-1] < 1e-6
        xd, _ = conjugate_gradient(lambda v: Fm @ v, g, iters=n + 5, tol=1e-18, damping=0.7)
        np.testing.assert_allclose(xd, np.linalg.solve(Fm + 0.7 * np.eye(n), g), rtol=1e-7, atol=1e-9)
    # fewer iterations -> larger residual; damping rule d = frac * mean diag F
    n = 10
    Fm = np.diag(np.arange(1, n + 1.0))
    g = np.ones(n)
    s, d, _, sFs = natural_direction(lambda v: Fm @ v, g, n, damp_frac=0.1, cg_iters=40)
    assert d == pytest.approx(0.1 * np.mean(np.diag(Fm)))
    np.testing.assert_allclose(s, g / (np.diag(Fm) + d), rtol=1e-6)
    assert sFs == pytest.approx(s @ Fm @ s)


# ---------------------------------------------------------------- 4. trust-region step
def test_trust_region_step_kl_surrogate_and_rejection():
    rng = np.random.default_rng(4)
    fam, th = Gauss(3), np.array([0.3, -0.2, 0.1, np.log(0.6)])
    phi = _phi(rng, 3, 400)
    a = fam.sample(th, phi, rng)
    A = rng.standard_normal(400) + 0.8 * (a - fam.dist(th, phi)[0])        # advantages positively correlated with the action noise
    g = policy_gradient(fam, th, phi, a, A)
    fvp = lambda v: fam.fvp(th, phi, v)
    s, d, _, sFs = natural_direction(fvp, g, fam.P, 0.01, 20)
    delta = 0.01
    beta = np.sqrt(2 * delta / sFs)
    assert 0.5 * beta ** 2 * (s @ fvp(s)) == pytest.approx(delta, rel=1e-12)
    lp0 = fam.logp(th, phi, a)
    surr = lambda c: surrogate(fam, th, c, phi, a, A, lp0)
    L0 = float(np.mean(A))
    cand, info = line_search(fam, th, phi, s, sFs, delta, surr, lambda t: t, L0)
    assert info["accepted"] and info["kl"] <= delta + 1e-9 and info["L"] > L0
    assert fam.kl(th, cand, phi) <= delta + 1e-9
    # no candidate qualifies (surrogate cannot improve): no update
    cand2, info2 = line_search(fam, th, phi, s, sFs, delta, lambda c: L0 - 1.0, lambda t: t, L0)
    assert cand2 is None and not info2["accepted"] and info2["nbt"] == 11
    # fixed unchecked step is accepted even when the surrogate is worse
    cand3, info3 = line_search(fam, th, phi, s, sFs, delta, lambda c: L0 - 1.0, lambda t: t, L0, check=False)
    assert cand3 is not None and info3["nbt"] == 0


# ---------------------------------------------------------------- 5. surrogate gradient
def test_surrogate_gradient_equals_policy_gradient_and_ratio_one():
    rng = np.random.default_rng(5)
    for fam, th, _ in _fams():
        phi = _phi(rng, fam.F, 50)
        a = fam.sample(th, phi, rng)
        A = rng.standard_normal(50)
        lp0 = fam.logp(th, phi, a)
        assert surrogate(fam, th, th, phi, a, A, lp0) == pytest.approx(A.mean())
        g = policy_gradient(fam, th, phi, a, A)
        num = np.zeros(fam.P)
        for i in range(fam.P):
            e = np.eye(fam.P)[i] * 1e-6
            num[i] = (surrogate(fam, th, th + e, phi, a, A, lp0) - surrogate(fam, th, th - e, phi, a, A, lp0)) / 2e-6
        np.testing.assert_allclose(g, num, rtol=1e-5, atol=1e-7)


def test_family_scores_match_policy_scores():
    rng = np.random.default_rng(6)
    pr = make_inv(H=10)
    for kind in ("I1", "I2", "I3"):
        pol = P.make_stoch("inventory", kind)
        th = pol.init(rng)
        o = pr.run(pol, th, pr.grid("train", 30)[:6], rng, "sample", record=True)
        fam = family_of(pol)
        phi, act = o["phi"].reshape(-1, o["phi"].shape[2]), o["act"].reshape(-1)
        sc = o["score"].reshape(-1, o["score"].shape[2])
        np.testing.assert_allclose(fam.score(th, phi, act), sc, rtol=1e-9, atol=1e-12)
    pt = make_trade(H=10)
    for kind in ("T1", "T2"):
        pol = P.make_stoch("trading", kind)
        th = pol.init(rng)
        o = pt.run(pol, th, pt.grid("train", 30)[:6], rng, "sample", record=True)
        fam = family_of(pol)
        phi, act = o["phi"].reshape(-1, o["phi"].shape[2]), o["act"].reshape(-1)
        np.testing.assert_allclose(fam.score(th, phi, act), o["score"].reshape(-1, o["score"].shape[2]), rtol=1e-9, atol=1e-12)


# ---------------------------------------------------------------- 6. GAE
def test_gae_hand_example_and_limits():
    rew = np.array([[1.0, 2.0, 3.0]])
    V = np.array([[0.5, 1.0, 1.5]])
    vl = np.array([2.0])
    g, lam = 0.9, 0.8
    d = [1 + g * 1.0 - 0.5, 2 + g * 1.5 - 1.0, 3 + g * 2.0 - 1.5]
    A, delta = gae(rew, V, vl, g, lam)
    np.testing.assert_allclose(delta[0], d)
    exp = [d[0] + g * lam * d[1] + (g * lam) ** 2 * d[2], d[1] + g * lam * d[2], d[2]]
    np.testing.assert_allclose(A[0], exp)
    A1, _ = gae(rew, V, np.zeros(1), g, 1.0)
    np.testing.assert_allclose(A1, returns_to_go(rew, g) - V)               # lambda = 1 -> Monte Carlo return minus V (terminal V = 0)
    A0, d0 = gae(rew, V, vl, g, 0.0)
    np.testing.assert_allclose(A0, d0)
    assert d[2] == pytest.approx(3 + g * vl[0] - 1.5)                       # the last step bootstraps with V(s_H)


# ---------------------------------------------------------------- 7. value function
def test_value_recovers_linear_function_and_order_of_operations(monkeypatch):
    rng = np.random.default_rng(7)
    H, F = 12, 3
    phi = rng.standard_normal((40, H, F))
    w = np.array([0.5, -1.0, 2.0, 0.7])
    vf = ValueFn(H, ridge=1e-12)
    y = (np.concatenate([phi, (((H - np.arange(H)) / H)[None, :, None] * np.ones((40, 1, 1)))], axis=2)) @ w
    vf.fit([(phi, y)])
    np.testing.assert_allclose(vf.w, w, atol=1e-6)
    assert vf.r2(phi, y) > 0.999999

    events = []

    class Rec(ValueFn):
        def fit(self, batches):
            events.append("fit")
            super().fit(batches)

        def predict(self, phi):
            events.append("predict")
            return super().predict(phi)

    monkeypatch.setattr(alg, "ValueFn", Rec)
    pr = make_trade(H=10)
    pol = P.BernoulliT()
    alg.trpo(pr, pol, pol.init(np.random.default_rng(0)), np.random.default_rng(1), 4 * 8 + 8, B=8, diag=False, n_eval=2)
    assert events[0] == "fit"                                    # pilot batch fit
    body = [e for e in events[1:]]
    assert body[0] == "predict" and body[:3].count("predict") == 2 and body[2] == "fit"                          # the V used for a batch's advantages is fitted BEFORE that batch
    assert all(body[i:i + 3] == ["predict", "predict", "fit"] for i in range(0, len(body) - 2, 3))


# ---------------------------------------------------------------- 8. convergence
def test_exact_bandit_every_accepted_step_improves_and_converges():
    fam = Softmax(1, 3)
    phi = np.ones((1, 1))
    r = np.array([0.2, 1.0, 0.5])
    th = np.zeros(3)
    J = lambda t: float(fam.probs(t, phi)[0] @ r)
    prev = J(th)
    for _ in range(300):
        p = fam.probs(th, phi)[0]
        Aexact = r - p @ r
        g = p * Aexact                                                 # exact gradient of J w.r.t. the logits
        if np.linalg.norm(g) < 1e-10:
            break
        fvp = lambda v: fam.fvp(th, phi, v)
        s, _, _, sFs = natural_direction(fvp, g, 3, 1e-3, None, exact=True)
        surr = lambda c: float(fam.probs(c, phi)[0] @ Aexact)          # exact expected surrogate
        cand, info = line_search(fam, th, phi, s, sFs, 0.01, surr, lambda t: t, 0.0)
        if cand is None:
            break
        th = cand
        assert J(th) >= prev - 1e-12
        prev = J(th)
    assert fam.probs(th, phi)[0][1] > 0.97


def test_exact_tiny_mdp_improves_monotonically():
    """2-state, 2-action MDP with exact discounted quantities (state weights = discounted visitation of the old policy)."""
    gam = 0.9
    # s0: a0 stay (r=0), a1 -> s1 (r=0); s1: a0 -> s0 (r=0), a1 stay (r=1)
    nxt = np.array([[0, 1], [0, 1]])
    rw = np.array([[0.0, 0.0], [0.0, 1.0]])
    fam = Softmax(2, 2)
    phi_s = np.eye(2)

    def pol_probs(th):
        return fam.probs(th, phi_s)

    def values(th):
        p = pol_probs(th)
        Pm = np.zeros((2, 2))
        r = np.zeros(2)
        for s in range(2):
            for a in range(2):
                Pm[s, nxt[s, a]] += p[s, a]
                r[s] += p[s, a] * rw[s, a]
        V = np.linalg.solve(np.eye(2) - gam * Pm, r)
        d = np.linalg.solve((np.eye(2) - gam * Pm).T, np.array([1.0, 0.0]))     # discounted visitation from s0
        Q = rw + gam * V[nxt]
        return V, d / d.sum(), Q

    th = np.zeros(4)
    prev = values(th)[0][0]
    for _ in range(400):
        V, d, Q = values(th)
        p = pol_probs(th)
        A = Q - V[:, None]
        reps = np.maximum(np.round(d * 5000).astype(int), 1)
        phi = np.repeat(phi_s, reps, axis=0)
        # exact gradient with the weights d: sum_s d(s) sum_a p(s,a) A(s,a) score(s,a)
        g = np.zeros(4)
        for s in range(2):
            for a in range(2):
                e = -p[s].copy()
                e[a] += 1.0
                g += (d[s] * p[s, a] * A[s, a] * np.outer(phi_s[s], e)).ravel()
        fvp = lambda v: fam.fvp(th, phi, v)
        s_, _, _, sFs = natural_direction(fvp, g, 4, 1e-3, None, exact=True)
        surr = lambda c: float(sum(d[s] * (pol_probs(c)[s] @ A[s]) for s in range(2)))
        cand, info = line_search(fam, th, phi, s_, sFs, 0.005, surr, lambda t: t, 0.0)
        if cand is None:
            break
        th = cand
        cur = values(th)[0][0]
        assert cur >= prev - 1e-9
        prev = cur
    assert prev > 0.97 * (1.0 / (1 - gam)) * gam ** 1 * 0.9 or prev > 7.0


def test_sampled_bandit_converges_with_trpo_step():
    rng = np.random.default_rng(8)
    fam = Softmax(1, 3)
    phi1 = np.ones((1, 1))
    means = np.array([0.1, 0.9, 0.4])
    th = np.zeros(3)
    for _ in range(150):
        phi = np.ones((256, 1))
        a = fam.sample(th, phi, rng)
        rr = means[a] + 0.3 * rng.standard_normal(256)
        A = (rr - rr.mean()) / (rr.std() + 1e-8)
        g = policy_gradient(fam, th, phi, a, A)
        fvp = lambda v: fam.fvp(th, phi, v)
        s, _, _, sFs = natural_direction(fvp, g, 3, 0.01, 20)
        lp0 = fam.logp(th, phi, a)
        cand, _ = line_search(fam, th, phi, s, sFs, 0.01, lambda c: surrogate(fam, th, c, phi, a, A, lp0), lambda t: t, float(A.mean()))
        if cand is not None:
            th = cand
    assert fam.probs(th, phi1)[0][1] > 0.9


# ---------------------------------------------------------------- 9. known optimum
def _inv_const_demand(demand, L, H=30, price=5.0, unit=3.0, hold=1.0, so=1.0, fixed=0.0, n=900, seed=0, poisson=None):
    rng = np.random.default_rng(seed)
    d = np.full(n, float(demand)) if poisson is None else rng.poisson(poisson, n).astype(float)
    cfg = ie.InvCfg(L=L, unit_cost=unit, holding=hold, stockout=so, fixed=fixed, init_inv=float(demand if poisson is None else poisson) * (L + 1))
    mu = float(d.mean())
    env = ie.InvData(d, np.full(n, price), np.arange(n) % 7, np.zeros(n), np.zeros(n), np.zeros(n), mu, cfg, qmax=None)
    sp = {"train": (0, 600), "val": (600, 750), "test": (750, 900)}
    pr = pb.Problem("inventory", "syn", env, sp, H, np.ones(n, bool))
    pr.norm = 30.0
    return pr


def test_known_optimum_constant_demand():
    pr = _inv_const_demand(5, 0)
    pol = P.GaussInv("I1", sigma0=0.3)
    th0 = pol.init(np.random.default_rng(0))
    th0[0] = 2.0
    r = trpo(pr, pol, th0, np.random.default_rng(1), 12000, delta=0.01, lam=0.97, B=32, gamma=1.0, n_eval=4)
    th = r["theta_final"]
    S = pol.features({"ip": np.zeros(1), "oh": np.zeros(1), "t": np.array([100]), "env": pr.env}) @ th[:pol.F]
    assert S[0] == pytest.approx(1.0, abs=0.15)            # order-up-to multiplier -> S = mu = 5 (documented tolerance 0.15)


def test_known_optimum_poisson_critical_ratio():
    from scipy.stats import poisson
    pr = _inv_const_demand(8, 0, poisson=8, fixed=0.0)
    cu, co = 5.0 + 1.0 - 3.0, 1.0
    q = cu / (cu + co)
    s_star = poisson.ppf(q, 8) / 8.0                      # critical-ratio (newsvendor) quantile as multiplier of mu
    pol = P.GaussInv("I1", sigma0=0.3)
    th0 = pol.init(np.random.default_rng(0))
    th0[0] = 1.0
    r = trpo(pr, pol, th0, np.random.default_rng(2), 20000, delta=0.01, lam=0.97, B=32, gamma=1.0, n_eval=4)
    th = r["theta_final"]
    S = pol.features({"ip": np.zeros(1), "oh": np.zeros(1), "t": np.array([100]), "env": pr.env}) @ th[:pol.F]
    assert abs(S[0] - s_star) < 0.25                      # tolerance documented: sampled Poisson demand, finite window (carry-over), 20k episodes


# ---------------------------------------------------------------- 10. projection, determinism, accounting, VPG direction
def test_sigma_floor_projection_rechecks_kl():
    pol = P.GaussInv("I2", sigma0=0.06)
    fam = family_of(pol)
    th = pol.init(np.random.default_rng(0))
    th[fam.F] = np.log(0.0501)
    rng = np.random.default_rng(1)
    phi = _phi(rng, fam.F, 100)
    s = np.zeros(fam.P)
    s[fam.F] = -1.0                                        # push log sigma far below the floor
    fvp = lambda v: fam.fvp(th, phi, v)
    sFs = float(s @ fvp(s))
    cand, info = line_search(fam, th, phi, s, sFs, 0.01, lambda c: 1.0, pol.project, 0.0)
    assert cand is not None and cand[fam.F] >= np.log(P.SIGMA_FLOOR) - 1e-12
    assert fam.kl(th, cand, phi) <= 0.01 + 1e-9 and info["kl"] == pytest.approx(fam.kl(th, cand, phi))


def test_determinism_and_episode_accounting():
    pr = make_inv(H=15)
    pol = P.GaussInv("I2")
    calls = {"n": 0}
    orig = pr.run

    def counting(pol_, th, starts, rng=None, mode="mean", **kw):
        if kw.get("record"):
            calls["n"] += len(starts)
        return orig(pol_, th, starts, rng, mode, **kw)

    pr.run = counting
    th0 = pol.init(np.random.default_rng(0))
    r1 = trpo(pr, pol, th0, np.random.default_rng(5), 700, B=16, n_eval=3)
    n1 = calls["n"]
    r2 = trpo(pr, pol, th0, np.random.default_rng(5), 700, B=16, n_eval=3)
    np.testing.assert_array_equal(r1["theta_final"], r2["theta_final"])
    assert r1["episodes_used"] == n1 == 16 * ((700 - 16) // 16 + 1)        # pilot batch included in the budget


def test_vanilla_pg_direction_equals_reinforce_direction():
    pr = make_trade(H=12)
    pol = P.BernoulliT()
    th = 0.3 * np.random.default_rng(1).standard_normal(7)
    starts = pr.grid("train", 20)[:10]
    g_r, o, G = grad_estimate(pr, pol, th, starts, np.random.default_rng(3), "none", 1.0)
    o2 = pr.run(pol, th, starts, np.random.default_rng(3), "sample", record=True)
    fam = family_of(pol)
    g_t = policy_gradient(fam, th, o2["phi"].reshape(-1, 7), o2["act"].reshape(-1), G.reshape(-1))
    np.testing.assert_allclose(g_t * pr.H, g_r, rtol=1e-9, atol=1e-12)


# ---------------------------------------------------------------- 13. learns a profitable rule
def test_trpo_learns_profitable_rule_on_synthetic_prices():
    pr = make_trade(H=40)
    pol = P.BernoulliT()
    th0 = np.zeros(7)
    r = trpo(pr, pol, th0, np.random.default_rng(0), 6000, delta=0.01, lam=0.97, B=32, gamma=0.995, n_eval=6)
    test_starts = pr.grid("test", 5)
    tot = pr.J(pol, r["theta_best"], test_starts, None, "det")
    assert tot > 0.0                                       # flat earns 0
    assert r["best_val"] > 0.0


def test_ng_fixed_and_vpg_variants_run():
    pr = make_inv(H=15)
    pol = P.SoftmaxInv()
    for var in ("vpg", "ng_fixed", "trpo"):
        r = trpo(pr, pol, pol.init(np.random.default_rng(0)), np.random.default_rng(1), 400, variant=var, B=16, n_eval=2)
        assert np.all(np.isfinite(r["theta_final"])) and len(r["it_log"]) > 0
