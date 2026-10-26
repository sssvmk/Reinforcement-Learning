"""Tests 12-20: REINFORCE machinery, baselines, optimiser, comparison methods, statistics, FDPG isolation."""
import glob
import hashlib
import os
import numpy as np
import pytest
import policies as P
import stats
from reinforce.algorithm import grad_estimate, reinforce
from reinforce.baselines import rewards_to_go, advantages, BaselineState
from reinforce.optimizer import Adam
from reinforce import comparison_methods as cm
from synth import make_inv, make_trade

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_rewards_to_go():
    r = np.array([[1.0, 2.0, 3.0]])
    np.testing.assert_allclose(rewards_to_go(r, 1.0), [[6, 5, 3]])
    np.testing.assert_allclose(rewards_to_go(r, 0.5), [[1 + 1 + 0.75, 2 + 1.5, 3]])


def test_baselines_leave_one_out_independent_of_own_return():
    G = np.random.default_rng(0).normal(size=(8, 5))
    for kind in ("loo_const", "loo_time"):
        A = advantages(kind, G, None, None, 5)
        G2 = G.copy()
        G2[3] += 100.0
        A2 = advantages(kind, G2, None, None, 5)
        # the baseline of episode 3 is unchanged when only its own return changes
        np.testing.assert_allclose((G - A)[3], (G2 - A2)[3])


def test_baseline_unbiasedness_bernoulli():
    """Mean gradient with each baseline equals the no-baseline mean gradient (up to Monte-Carlo error)."""
    pr = make_trade(H=15)
    pol = P.BernoulliT()
    th = 0.3 * np.random.default_rng(0).standard_normal(7)
    starts = pr.grid("train", 20)[:16]
    res = {}
    for bl in ("none", "loo_time", "loo_const"):
        rng = np.random.default_rng(5)
        gs = np.array([grad_estimate(pr, pol, th, starts, rng, bl, 1.0)[0] for _ in range(400)])
        res[bl] = (gs.mean(0), gs.std(0) / np.sqrt(len(gs)))
    for bl in ("loo_time", "loo_const"):
        diff = np.abs(res[bl][0] - res["none"][0])
        se = np.sqrt(res[bl][1] ** 2 + res["none"][1] ** 2)
        assert (diff < 4 * se + 1e-9).all()


def test_baseline_reduces_variance():
    pr = make_inv(H=20)           # returns have a large common mean, so a baseline must help
    pol = P.GaussInv("I1")
    th = pol.init(np.random.default_rng(0))
    v = {}
    for bl in ("none", "loo_time"):
        rng = np.random.default_rng(9)
        gs = np.array([grad_estimate(pr, pol, th, pr.sample("train", 32, rng), rng, bl, 1.0)[0] for _ in range(200)])
        v[bl] = gs.var(0).sum()
    assert v["loo_time"] < v["none"]


def test_gradient_matches_finite_difference_of_expected_return():
    """Large-batch REINFORCE gradient aligns with a finite-difference gradient of E[J] (same windows, many samples)."""
    pr = make_trade(H=8)
    pol = P.BernoulliT()
    th = 0.3 * np.random.default_rng(1).standard_normal(7)
    starts = np.repeat(pr.grid("train", 25)[:12], 600)
    rng = np.random.default_rng(2)
    g = grad_estimate(pr, pol, th, starts, rng, "loo_time", 1.0)[0]

    def J(t):
        return pr.J(pol, t, starts, np.random.default_rng(3), "sample")
    fd = np.array([(J(th + 0.05 * e) - J(th - 0.05 * e)) / 0.1 for e in np.eye(7)])
    cos = g @ fd / (np.linalg.norm(g) * np.linalg.norm(fd))
    assert cos > 0.9


def test_training_improves_validation_and_is_reproducible():
    pr = make_trade(H=30)
    pol = P.BernoulliT()
    out = []
    for _ in range(2):
        rng = np.random.default_rng(0)
        th0 = pol.init(rng)
        out.append(reinforce(pr, pol, th0, rng, 3200, lr=0.1, B=32, baseline="loo_time", gamma=0.99, n_eval=8))
    np.testing.assert_allclose(out[0]["theta_best"], out[1]["theta_best"])
    assert out[0]["best_val"] >= out[0]["log"][0]["val_det"]
    assert out[0]["best_val"] > -1.0


def test_inventory_training_runs_all_policies_and_baselines():
    pr = make_inv(H=20)
    for k in ("I1", "I2", "I3"):
        pol = P.make_stoch("inventory", k)
        for bl in ("none", "loo_const", "loo_time", "linear", "loo_time_std"):
            rng = np.random.default_rng(0)
            r = reinforce(pr, pol, pol.init(rng), rng, 320, lr=0.05, B=16, baseline=bl, gamma=0.99, n_eval=4)
            assert np.isfinite(r["best_val"]) and np.all(np.isfinite(r["theta_best"]))


def test_sigma_floor_and_adam_clipping():
    pol = P.GaussInv("I1")
    th = np.zeros(11)
    th[10] = -20.0
    assert np.exp(pol.project(th)[10]) >= P.SIGMA_FLOOR - 1e-12
    opt = Adam(3, lr=1.0, clip=5.0)
    _, n = opt.step(np.zeros(3), np.array([300.0, 400.0, 0.0]))
    assert n == pytest.approx(500.0)
    x, _ = Adam(3, lr=0.1, clip=5.0).step(np.zeros(3), np.array([300.0, 0.0, 0.0]))
    assert abs(x[0]) <= 0.1 + 1e-9


def test_entropy_bonus_increases_sigma_direction():
    pr = make_inv(H=10)
    pol = P.GaussInv("I1")
    th = pol.init(np.random.default_rng(0))
    starts = pr.grid("train", 9)[:8]
    g0 = grad_estimate(pr, pol, th, starts, np.random.default_rng(1), "loo_time", 1.0, ent_coef=0.0)[0]
    g1 = grad_estimate(pr, pol, th, starts, np.random.default_rng(1), "loo_time", 1.0, ent_coef=1.0)[0]
    assert g1[-1] - g0[-1] == pytest.approx(10.0)       # per-step +1 on log sigma over H=10 steps


def test_comparison_methods_respect_budget_and_run():
    pr = make_inv(H=20)
    pol = P.GaussInv("I1")
    for name, f in cm.METHODS.items():
        rng = np.random.default_rng(0)
        r = f(pr, pol, pol.init(rng), rng, 640, B=16)
        assert r["log"][-1]["episodes"] <= 640 + 2 * 16 * pol.mean_dim + 64
        assert np.isfinite(r["best_val"])


def test_statistics_helpers():
    assert stats.holm([0.01, 0.04, 0.03]).tolist() == pytest.approx([0.03, 0.06, 0.06])
    assert stats.cliffs_delta([3, 4, 5], [0, 1, 2]) == 1.0
    lo, hi = stats.bootstrap_ci(np.arange(100.0))
    assert lo < 49.5 < hi
    m, lo, hi, p = stats.paired_bootstrap(np.full(30, 1.0) + np.random.default_rng(0).normal(0, 0.1, 30))
    assert lo > 0 and p < 0.01


def _fdpg_dir():
    from common import load_config, resolve_fdpg_dir
    try:
        return resolve_fdpg_dir(load_config())
    except Exception:
        return None


def test_no_fdpg_imports_or_pickles_in_source():
    bad = ("import pickle", "pickle.load", "allow_pickle=True", "joblib.load", "from fdpg", "import fdpg")
    for f in glob.glob(os.path.join(ROOT, "*.py")) + glob.glob(os.path.join(ROOT, "reinforce", "*.py")):
        txt = open(f, encoding="utf8").read()
        for b in bad:
            assert b not in txt, (f, b)


def test_fdpg_folder_hash_unchanged_by_reading():
    d = _fdpg_dir()
    if not d or not os.path.isdir(d):
        pytest.skip("FDPG folder absent")
    import compare_with_fdpg as cw
    h1 = cw.hash_tree(d)
    cw.read_fdpg_texts(d)
    assert cw.hash_tree(d) == h1
