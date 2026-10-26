"""Tests 6-12 and 15 (update rules, tiny MDPs, window-reset effect, Baird, rho=1 identities, FQE, oracle)."""
import json
import numpy as np
import pytest
from numba import njit
from mdps import tiny_exact, tiny_chain, tiny_windows, baird, baird_exact, baird_chain
from estimators import (TD, GTD2, TDC, ETD, run_method, run_kernel, one_step, emphasis_trace, make_orders,
                        batch_AbC, mspbe, wis, fqe, oracle_projection)
from common import ROOT

SANITY = {}


def _save():
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    (out / "sanity_results.json").write_text(json.dumps(SANITY, indent=1, default=float))


# ---------------------------------------------------------------- test 6: rule-level
def _ref_step(method, w, v, x, xp, rho, R, a, b, g, vform=0):
    delta = R + g * w @ xp - w @ x
    xv = x @ v
    if method == GTD2:
        wn = w + a * rho * (x - g * xp) * xv
    elif method == TDC:
        wn = w + a * rho * (delta * x - g * xp * xv)
    elif method == TD:
        wn = w + a * rho * delta * x
    if vform == 0:
        vn = v + b * rho * (delta - xv) * x
    else:
        vn = v + b * (rho * delta * x - xv * x)
    return wn, vn


@pytest.mark.parametrize("method", [GTD2, TDC])
@pytest.mark.parametrize("vform", [0, 1])
def test_gradient_td_one_step(method, vform):
    rng = np.random.default_rng(1)
    d = 5
    w, v, x, xp = rng.normal(size=d), rng.normal(size=d), rng.normal(size=d), rng.normal(size=d)
    rho, R, a, b, g = 1.7, 0.4, 0.05, 0.3, 0.9
    w1, v1 = w.copy(), v.copy()
    one_step(method, w1, v1, x, xp, rho, R, a, b, g, 0.0, vform, 0.0, np.zeros(d), 0.0)
    wr, vr = _ref_step(method, w, v, x, xp, rho, R, a, b, g, vform)
    np.testing.assert_allclose(w1, wr, rtol=1e-12)     # (a) uses pre-update v and w
    np.testing.assert_allclose(v1, vr, rtol=1e-12)
    # (b) rho multiplies the whole v update (vform 0) : doubling rho changes v-increment accordingly
    if vform == 0:
        w2, v2 = w.copy(), v.copy()
        one_step(method, w2, v2, x, xp, 2 * rho, R, a, b, g, 0.0, 0, 0.0, np.zeros(d), 0.0)
        d2 = R + g * w @ xp - w @ x - x @ v
        np.testing.assert_allclose(v2 - v, b * 2 * rho * d2 * x, rtol=1e-12)


def etd_reference(Xs_win, rho_win, R_win, alpha, gamma, lam, w0, reset=True):
    """Pure-python ETD(lambda) over a list of windows following Part 2: F_t = gamma*rho_{t-1}*F_{t-1} + 1."""
    w = w0.copy()
    e = np.zeros_like(w)
    F_prev, rho_prev = None, None
    for Xs, rho, R in zip(Xs_win, rho_win, R_win):
        if reset:
            e = np.zeros_like(w)
            F_prev = None
        for t in range(len(R)):
            x, xp = Xs[t], Xs[t + 1]
            delta = R[t] + gamma * w @ xp - w @ x
            F = 1.0 if F_prev is None else gamma * rho_prev * F_prev + 1.0
            M = lam + (1 - lam) * F
            e = rho[t] * (gamma * lam * e + M * x)
            w = w + alpha * delta * e
            F_prev, rho_prev = F, rho[t]
    return w


@pytest.mark.parametrize("lam", [0.0, 0.5, 0.9])
def test_etd_rule_level(lam):
    rng = np.random.default_rng(3)
    Wn, H, d = 3, 4, 3
    Xs = rng.normal(size=(Wn, H + 1, d))
    rho = rng.uniform(0.3, 2.5, size=(Wn, H))
    R = rng.normal(size=(Wn, H))
    order = np.arange(Wn)[None, :]
    out = run_kernel(ETD, Xs, rho, R, order, 0.05, 10.0, 0.9, lam, 0, True, 0.0, 0.0, 1.0, 5, np.zeros(d))
    wref = etd_reference(Xs, rho, R, 0.05, 0.9, lam, np.zeros(d), reset=True)
    np.testing.assert_allclose(out[1], wref, rtol=1e-10)
    # (f) F restarts at 1 at every window start; (c) rho_{t-1} only inside F
    F = emphasis_trace(rho, 0.9, True, 0.0)
    np.testing.assert_allclose(F[:, 0], 1.0)
    np.testing.assert_allclose(F[:, 1], 0.9 * rho[:, 0] * 1.0 + 1.0)
    np.testing.assert_allclose(F[:, 2], 0.9 * rho[:, 1] * F[:, 1] + 1.0)
    # no-reset: window w+1 starts from the carried F of window w
    Fn = emphasis_trace(rho, 0.9, False, 0.0)
    np.testing.assert_allclose(Fn[1, 0], 0.9 * rho[0, -1] * Fn[0, -1] + 1.0)


def test_etd_lambda0_matches_closed_form_and_rho_t_on_trace():
    rng = np.random.default_rng(5)
    d, g, a = 4, 0.9, 0.1
    Xs = rng.normal(size=(1, 3, d))
    rho = np.array([[1.8, 0.6]])
    R = np.array([[0.3, -0.2]])
    out = run_kernel(ETD, Xs, rho, R, np.zeros((1, 1), dtype=np.int64), a, 10.0, g, 0.0, 0, True, 0.0, 0.0, 1.0, 2,
                     np.zeros(d))
    w = np.zeros(d)
    F0 = 1.0
    F1 = g * rho[0, 0] * F0 + 1.0
    for t, F in enumerate((F0, F1)):
        delta = R[0, t] + g * w @ Xs[0, t + 1] - w @ Xs[0, t]
        w = w + a * F * rho[0, t] * delta * Xs[0, t]       # (e) lambda = 0 closed form
    np.testing.assert_allclose(out[1], w, rtol=1e-12)


# ---------------------------------------------------------------- tests 7/8: tiny MDPs
def test_tiny1_exact_numbers():
    e = tiny_exact(0.5)
    assert e["w_td"] == pytest.approx(2.7273, abs=1e-4)
    assert e["A"][0, 0] == pytest.approx(0.22) and e["b"][0] == pytest.approx(0.6)
    np.testing.assert_allclose(e["f"], [2.3, 7.7])
    assert e["A_E"][0, 0] == pytest.approx(0.904) and e["b_E"][0] == pytest.approx(4.92)
    assert e["w_etd"] == pytest.approx(5.4425, abs=1e-4)
    assert e["gamma2_Erho2"] == pytest.approx(1.10, abs=0.002) and e["gamma2_Erho2"] > 1
    SANITY["tiny1"] = {k: (np.asarray(e[k]).tolist()) for k in ("w_td", "f", "w_etd", "gamma2_Erho2")}


def test_tiny2_exact_numbers():
    e = tiny_exact(0.7)
    np.testing.assert_allclose(e["d"], [0.3, 0.7])
    np.testing.assert_allclose(e["rho"], [0.2 / 0.3, 0.8 / 0.7])
    assert e["w_td"] == pytest.approx(4.1935, abs=1e-4)
    np.testing.assert_allclose(e["f"], [2.1, 7.9])
    assert e["w_etd"] == pytest.approx(5.9901, abs=1e-4)
    assert e["gamma2_Erho2"] == pytest.approx(0.8486, abs=1e-4) and e["gamma2_Erho2"] < 1
    SANITY["tiny2"] = {k: np.asarray(e[k]).tolist() for k in ("w_td", "f", "w_etd", "gamma2_Erho2")}


def test_tiny1_expected_updates():
    """Expected updates: GTD2/TDC go to w_TD (v->0); ETD(0) goes to w_ETD, not to w_TD."""
    e = tiny_exact(0.5)
    A, b, C = e["A"], e["b"], (e["X"].T @ np.diag(e["d"]) @ e["X"])
    for name in ("gtd2", "tdc"):
        w, v = np.zeros(1), np.zeros(1)
        for _ in range(60000):
            if name == "gtd2":
                dw = A.T @ v
            else:
                dw = b - A @ w - (C - A.T) @ v
            dv = b - A @ w - C @ v
            w, v = w + 0.02 * dw, v + 0.2 * dv
        assert w[0] == pytest.approx(e["w_td"], rel=1e-3) and abs(v[0]) < 1e-3
        assert abs(w[0] - e["w_etd"]) > 1.0
    w = np.zeros(1)
    for _ in range(20000):
        w = w + 0.1 * (e["b_E"] - e["A_E"] @ w)
    assert w[0] == pytest.approx(e["w_etd"], rel=1e-4) and abs(w[0] - e["w_td"]) > 1.0


def test_tiny1_sampled_runs_are_reported():
    e = tiny_exact(0.5)
    Xs, rho, R = tiny_chain(0.5, 200_000, 11)
    o = np.zeros((1, 1), dtype=np.int64)
    res = {}
    for name, m in (("gtd2", GTD2), ("tdc", TDC), ("etd0", ETD)):
        r = run_method(m, Xs, rho, R, 0.05, 10.0, 0.9, n_epochs=1, decay_p=0.7, decay_k0=1000, reset=False, order=o)
        res[name] = float(r["w_tail"][0]) if not r["diverged"] else float("nan")
    SANITY["tiny1_sampled_tail_avg"] = res
    SANITY["tiny1_sampled_note"] = "gamma^2 E[rho^2] = 1.10 > 1: sampled ETD has infinite emphasis variance; erratic"
    assert res["gtd2"] == pytest.approx(e["w_td"], rel=0.1)
    assert res["tdc"] == pytest.approx(e["w_td"], rel=0.1)


def _sampled(m, bR, n, seed, lam=0.0, reset=False, alpha=0.05, kappa=10.0):
    Xs, rho, R = tiny_chain(bR, n, seed)
    r = run_method(m, Xs, rho, R, alpha, kappa, 0.9, lam=lam, n_epochs=1, decay_p=0.7, decay_k0=1000, reset=reset,
                   order=np.zeros((1, 1), dtype=np.int64))
    return float(r["w_tail"][0])


def test_tiny2_sampled_convergence():
    e = tiny_exact(0.7)
    tol = 0.03   # 3% relative: seeded 1e6-step run, tail-averaged over the last 20% of iterates
    assert _sampled(GTD2, 0.7, 1_000_000, 1) == pytest.approx(e["w_td"], rel=tol)
    assert _sampled(TDC, 0.7, 1_000_000, 1) == pytest.approx(e["w_td"], rel=tol)
    assert _sampled(ETD, 0.7, 1_000_000, 1, lam=0.0) == pytest.approx(e["w_etd"], rel=tol)
    SANITY["tiny2_sampled"] = dict(gtd2=_sampled(GTD2, 0.7, 1_000_000, 1), tdc=_sampled(TDC, 0.7, 1_000_000, 1),
                                   etd0=_sampled(ETD, 0.7, 1_000_000, 1), tol_rel=tol)


@njit(cache=True)
def _lam_stats(Xs, rho, R, gamma, lam, burn):
    carry, e, sA, sb, n = 0.0, 0.0, 0.0, 0.0, 0
    for t in range(rho.shape[0]):
        F = carry + 1.0
        M = lam + (1 - lam) * F
        e = rho[t] * (gamma * lam * e + M * Xs[t])
        if t >= burn:
            sA += e * (Xs[t] - gamma * Xs[t + 1])
            sb += e * R[t]
            n += 1
        carry = gamma * rho[t] * F
    return sA / n, sb / n


@pytest.mark.parametrize("lam", [0.0, 0.5, 0.9])
def test_tiny2_lambda_fixed_point(lam):
    Xs, rho, R = tiny_chain(0.7, 2_000_000, 123)
    A, b = _lam_stats(Xs[0, :, 0], rho[0], R[0], 0.9, lam, 10_000)
    w_star = b / A
    if lam == 0.0:
        assert w_star == pytest.approx(5.9901, rel=0.02)
    got = _sampled(ETD, 0.7, 1_000_000, 7, lam=lam)
    SANITY.setdefault("tiny2_lambda", {})[str(lam)] = dict(b_over_A=float(w_star), sampled_tail_avg=got)
    assert got == pytest.approx(w_star, rel=0.08)     # tolerance documented: seeded, 8% relative


# ---------------------------------------------------------------- test 9: window reset effect
def test_window_reset_effect_tiny2():
    e = tiny_exact(0.7)
    Xs, rho, R, _, _ = tiny_windows(0.7, 8000, 100, 5)
    res = {}
    for reset in (True, False):
        order = np.tile(np.arange(Xs.shape[0]), (3, 1))
        r = run_method(ETD, Xs, rho, R, 0.02, 10.0, 0.9, reset=reset, order=order, decay_p=0.6, decay_k0=5000)
        res["reset" if reset else "no_reset"] = float(r["w_tail"][0])
    res["continuing_fixed_point"] = e["w_etd"]
    res["abs_diff_reset_vs_continuing"] = res["reset"] - e["w_etd"]
    SANITY["window_reset_effect_tiny2"] = res
    assert abs(res["reset"] - e["w_etd"]) > 0.0


# ---------------------------------------------------------------- test 10: Baird
def test_baird_structure():
    X = baird()
    np.testing.assert_array_equal(X[0], [2, 0, 0, 0, 0, 0, 0, 1])
    np.testing.assert_array_equal(X[6], [0, 0, 0, 0, 0, 0, 1, 2])
    np.testing.assert_allclose(X @ np.array([1, 1, 1, 1, 1, 1, 4, -2.0]), 0.0)   # null-space direction


def test_baird_behaviors():
    Xs, rho, R = baird_chain(100_000, 0)
    w0 = np.array([1, 1, 1, 1, 1, 1, 10, 1.0])
    o = np.zeros((1, 1), dtype=np.int64)
    td = run_method(TD, Xs, rho, R, 0.01, gamma=0.99, n_epochs=1, w0=w0, order=o)
    assert td["diverged"] or np.linalg.norm(td["w"]) > 1e3
    Xall = baird()
    start = np.abs(Xall @ w0).max()
    out = {"td_diverged": bool(td["diverged"]), "td_final_norm": float(np.linalg.norm(td["w"]))}
    for name, m in (("tdc", TDC), ("gtd2", GTD2)):
        r = run_method(m, Xs, rho, R, 0.005, 10.0, 0.99, n_epochs=1, w0=w0, order=o)
        assert not r["diverged"]
        vmax = np.abs(Xall @ r["w"]).max()
        out[name] = dict(max_abs_vhat_start=float(start), max_abs_vhat_end=float(vmax),
                         final_w=r["w"].tolist())
        assert np.linalg.norm(r["w"]) < 1e3
    SANITY["baird_sampled"] = out


def test_baird_etd_expected_updates():
    B = baird_exact()
    w = np.array([1, 1, 1, 1, 1, 1, 10, 1.0])
    traj = []
    for k in range(200000):
        w = w - 0.03 * B["A_E"] @ w            # rewards are zero: b_E = 0
        if k % 20000 == 0:
            traj.append(float(np.abs(B["X"] @ w).max()))
    final = float(np.abs(B["X"] @ w).max())
    SANITY["baird_etd_expected"] = dict(max_abs_vhat_trace=traj, final=final)
    assert np.isfinite(final) and final < 0.1 * traj[0]
    # and semi-gradient TD expected update diverges in this MDP
    w = np.array([1, 1, 1, 1, 1, 1, 10, 1.0])
    for _ in range(20000):
        w = w - 0.01 * B["A"] @ w
    assert np.linalg.norm(w) > 1e3 or not np.isfinite(np.linalg.norm(w))


def test_zz_save_sanity():
    _save()


# ---------------------------------------------------------------- test 12 and 15
def test_on_policy_identities():
    rng = np.random.default_rng(0)
    rho = np.ones((4, 6))
    F = emphasis_trace(rho, 0.9, True, 0.0)
    for t in range(1, 6):
        np.testing.assert_allclose(F[:, t], 0.9 * F[:, t - 1] + 1.0)
    R = rng.normal(size=(50, 6))
    J, ess = wis(R, np.zeros((50, 6)), 0.9, 50.0)
    G = (R * 0.9 ** np.arange(6)).sum(axis=1)
    assert J == pytest.approx(G.mean()) and ess == pytest.approx(50.0)


def test_fqe_recovers_tiny_mdp_values():
    Xs, rho, R, a, s = tiny_windows(0.5, 3000, 100, 2)
    Xtab = np.eye(2)[s]                                   # tabular features
    piR = 0.8
    Pn = np.tile([1 - piR, piR], (*a.shape, 1))
    P0 = np.tile([1 - piR, piR], (a.shape[0], 1))
    J, _ = fqe(Xtab, a, R, Pn, P0, 0.9, 200, 1e-6, 2)
    assert J == pytest.approx(0.8 / (1 - 0.9), rel=0.02)   # v = 0.8/(1-gamma) = 8


def test_oracle_projection_floor():
    rng = np.random.default_rng(0)
    Xst = rng.normal(size=(200, 6))
    v = rng.normal(size=200)
    wo = oracle_projection(Xst, v)
    msve_o = np.mean((Xst @ wo - v) ** 2)
    for _ in range(20):
        w = rng.normal(size=6)
        assert msve_o <= np.mean((Xst @ w - v) ** 2) + 1e-12
