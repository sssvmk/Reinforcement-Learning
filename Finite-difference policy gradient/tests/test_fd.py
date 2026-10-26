import numpy as np
import pytest
from fdpg.crn import Objective, PathProvider, PathSampler
from fdpg.estimators import ESTIMATORS, central, forward, spsa, regression
from fdpg.optimizer import HParams, Scaler, project, run_fd

STAR = np.array([1.0, -2.0, 0.5])


class QuadJ:
    """J(u) = -||u - STAR||^2 + noise * (mean of per-path scalar offsets); offsets are theta-independent."""

    def __init__(self, noise=0.0, n_paths=8):
        self.noise, self.n_paths, self.rollouts = noise, n_paths, 0
        self.fixed_paths = np.arange(4)
        self.log = []

    def draw(self, rng):
        return rng.integers(0, 10 ** 9, self.n_paths)

    def __call__(self, u, paths):
        self.rollouts += len(paths)
        self.log.append(np.asarray(paths).copy())
        off = np.mean([np.random.default_rng(int(p)).normal() for p in paths]) * self.noise
        return -np.sum((u - STAR) ** 2) + off


U0 = np.array([0.0, 0.0, 0.0])
TRUE_G = -2 * (U0 - STAR)


def test_central_exact_forward_biased():
    gc = central(QuadJ(), U0, np.random.default_rng(0), eps=0.3)
    np.testing.assert_allclose(gc, TRUE_G, atol=1e-9)
    gf = forward(QuadJ(), U0, np.random.default_rng(0), eps=0.3)
    np.testing.assert_allclose(gf, TRUE_G - 0.3, atol=1e-9)   # O(eps) bias exactly -eps for a quadratic
    gf2 = forward(QuadJ(), U0, np.random.default_rng(0), eps=0.03)
    assert np.abs(gf2 - TRUE_G).max() < np.abs(gf - TRUE_G).max() / 5


def test_spsa_and_regression_converge_to_true_gradient():
    g = spsa(QuadJ(), U0, np.random.default_rng(1), eps=0.05, m=6000)
    assert np.abs(g - TRUE_G).max() < 0.25
    g = regression(QuadJ(), U0, np.random.default_rng(2), eps=0.05, K=600)
    assert np.abs(g - TRUE_G).max() < 0.1


def test_regression_axis_aligned_equals_central():
    eps, n = 0.2, 3
    D = np.vstack([eps * np.eye(n), -eps * np.eye(n)])
    g = regression(QuadJ(), U0, np.random.default_rng(0), eps=eps, D=D, ridge_rel=0.0)
    np.testing.assert_allclose(g, central(QuadJ(), U0, np.random.default_rng(0), eps=eps), atol=1e-9)


@pytest.mark.parametrize("est", list(ESTIMATORS))
@pytest.mark.parametrize("noise", [0.0, 1.0])
def test_gradient_ascent_converges(est, noise):
    J = QuadJ(noise)
    hp = HParams(est=est, eps=0.1, a=1.0, A=10, budget=30000, clip=10.0)
    out = run_fd(J, QuadJ(), U0, -5 * np.ones(3), 5 * np.ones(3), hp, seed=0)
    assert np.abs(out["u_best"] - STAR).max() < (0.25 if est in ("spsa", "gaussian") else 0.15)


def test_crn_identical_paths_and_lower_variance():
    J = QuadJ(noise=1.0)
    central(J, U0, np.random.default_rng(0), eps=0.1, crn=True)
    assert len(J.log) == 6 and all(np.array_equal(J.log[0], p) for p in J.log)
    J2 = QuadJ(noise=1.0)
    central(J2, U0, np.random.default_rng(0), eps=0.1, crn=False)
    assert not all(np.array_equal(J2.log[0], p) for p in J2.log)
    v = lambda crn: np.var([central(QuadJ(1.0), U0, np.random.default_rng(s), eps=0.1, crn=crn) for s in range(40)], axis=0).sum()
    assert v(True) < v(False) / 100


def test_projection_clipping_scaling_determinism():
    lo, hi = -np.ones(3), np.ones(3)
    assert np.all(project(np.array([-5, 0.2, 9.0]), lo, hi) == np.array([-1, 0.2, 1.0]))
    sc = Scaler([2.0, 10.0, 0.5])
    p = np.array([1.0, -3.0, 0.7])
    np.testing.assert_allclose(sc.to_phys(sc.to_internal(p)), p)
    hp = HParams(est="central", eps=0.1, a=1.0, A=1, clip=0.5, budget=200)
    out = run_fd(QuadJ(), QuadJ(), U0 + 100, -1e3 * np.ones(3), 1e3 * np.ones(3), hp, seed=3)
    u = [c[4] for c in out["curve"]]
    assert np.linalg.norm(u[1] - u[0]) <= 1.0 / 1 ** 0.602 * 0.5 + 1e-9
    o1 = run_fd(QuadJ(1.0), QuadJ(), U0, -5 * np.ones(3), 5 * np.ones(3), HParams(est="spsa", budget=2000), seed=7)
    o2 = run_fd(QuadJ(1.0), QuadJ(), U0, -5 * np.ones(3), 5 * np.ones(3), HParams(est="spsa", budget=2000), seed=7)
    np.testing.assert_array_equal(o1["u_last"], o2["u_last"])


def test_rollout_accounting():
    count = [0]

    def roll(th, starts):
        count[0] += len(starts)
        return -np.sum((th - STAR) ** 2) * np.ones(len(starts))

    J = Objective(roll, PathSampler(0, 1000, 10), 8)
    Jv = Objective(roll, PathSampler(0, 1000, 10), 8, fixed_paths=np.arange(5))
    out = run_fd(J, Jv, U0, -5 * np.ones(3), 5 * np.ones(3), HParams(est="central", budget=480), seed=0)
    assert J.rollouts == out["rollouts"] == 480 == out["iters"] * 6 * 8
    assert count[0] == J.rollouts + Jv.rollouts   # validation rollouts are counted separately, never in the budget
