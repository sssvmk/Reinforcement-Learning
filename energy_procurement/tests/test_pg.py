import os
import sys
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pg import policies
from pg.env import Risk
from pg.fdpg import FDPG
from pg.reinforce import Reinforce
from pg.actor_critic import A2C
from pg.trpo import TRPO, conjugate_gradient
from pg.ppo import PPO
from pg.dpg import DPG


class SynthEnv:
    """Known-optimum stand-in for DayEnv: a*(z) = W.z, step cost k (a - a*)^2, so the optimal linear policy is W."""
    n_in = 3
    W = np.array([0.04, -0.03])

    def __init__(self, D=300, seed=0, k=100.0):
        rng = np.random.default_rng(seed)
        self.Z = torch.tensor(rng.normal(size=(D, 24, 2)), dtype=torch.float32)
        self.k = k
        self.Lhat = np.ones((D, 24)); self.cap = np.full((D, 24), 5.0)

        class P: pass
        self.P = P()
        self.P.idx = lambda split: np.arange(D) if split != "val" else np.arange(D // 2)

    def astar(self, idx):
        return (self.Z[idx] * torch.tensor(self.W, dtype=torch.float32)).sum(-1)

    def rollout(self, idx, act_fn):
        idx = np.asarray(idx)
        S = torch.cat([self.Z[idx], torch.zeros(len(idx), 24, 1)], -1)
        A = torch.stack([act_fn(S[:, h]) for h in range(24)], 1)
        cost = self.k * (A - self.astar(idx)) ** 2 / 24
        q = (1 + A.detach().numpy()).clip(0, 5)
        return {"S": S, "A": A, "R": -cost, "q": q, "Ct": cost.sum(1).detach().numpy(), "idx": idx}


# ---------- test 9: score-function gradient and baselines ----------
def _grad_estimate(n, baseline, seed=0, mu=0.2, c=0.5, sg=0.3):
    g = torch.Generator().manual_seed(seed)
    a = mu + sg * torch.randn(n, generator=g)
    cost = (a - c) ** 2
    score = (a - mu) / sg ** 2  # d log pi / d mu
    if baseline == "loo":
        b = (cost.sum() - cost) / (n - 1)
    elif baseline == "mean":
        b = cost.mean()
    else:
        b = 0.0
    return float((score * (cost - b)).mean()), float((score * (cost - b)).var())


def test_score_function_gradient_matches_analytic():
    mu, c = 0.2, 0.5
    g, _ = _grad_estimate(2_000_000, "none")
    assert g == pytest.approx(2 * (mu - c), abs=0.03)  # E[(a-c)^2] = (mu-c)^2 + sg^2


def test_baselines_unbiased_and_reduce_variance():
    g0, v0 = _grad_estimate(1_000_000, "none")
    gl, vl = _grad_estimate(1_000_000, "loo")
    gm, vm = _grad_estimate(1_000_000, "mean")
    assert gl == pytest.approx(g0, abs=0.05) and gm == pytest.approx(g0, abs=0.05)
    assert vl < v0 and vm < v0


def test_gaussian_logprob_gradient_matches_numeric():
    pol = policies.GaussianPolicy(3, "linear", 0.1)
    s, a = torch.randn(5, 3), torch.randn(5)
    pol.mu.weight.data.normal_()
    g = torch.autograd.grad(pol.log_prob(s, a).sum(), pol.mu.weight)[0]
    eps, w0 = 1e-3, pol.mu.weight.data.clone()
    num = torch.zeros_like(w0)
    for j in range(3):
        for sgn in (1, -1):
            pol.mu.weight.data = w0.clone(); pol.mu.weight.data[0, j] += sgn * eps
            num[0, j] += sgn * pol.log_prob(s, a).sum().item() / (2 * eps)
    pol.mu.weight.data = w0
    assert torch.allclose(g, num, rtol=2e-2, atol=2e-2)


# ---------- test 10: GAE ----------
def test_gae_hand_example():
    r = torch.tensor([[1.0, 2.0, 3.0]])
    v = torch.tensor([[0.5, 1.0, 1.5, 0.0]])
    gamma, lam = 0.9, 0.8
    d = [1 + .9 * 1.0 - .5, 2 + .9 * 1.5 - 1.0, 3 + 0 - 1.5]
    a2 = d[2]; a1 = d[1] + gamma * lam * a2; a0 = d[0] + gamma * lam * a1
    adv, ret = policies.gae(r, v, gamma, lam)
    assert adv[0].tolist() == pytest.approx([a0, a1, a2], rel=1e-5)
    assert ret[0].tolist() == pytest.approx([a0 + .5, a1 + 1.0, a2 + 1.5], rel=1e-5)
    adv1, _ = policies.gae(r, v, gamma, 1.0)  # lambda = 1: discounted return minus value
    G = [1 + .9 * 2 + .81 * 3, 2 + .9 * 3, 3.0]
    assert adv1[0].tolist() == pytest.approx([G[0] - .5, G[1] - 1.0, G[2] - 1.5], rel=1e-5)


# ---------- test 11: finite differences, SPSA, common random numbers ----------
def _fd_algo(method="central", **kw):
    env = SynthEnv()
    cfg = {"fd": method, "N": 64, "lr": 0.003, "eps": 0.01, "budget_days": 1, **kw}
    return FDPG(env, Risk("neutral"), cfg, 0), env


def test_central_difference_matches_numerical_gradient_and_accounting():
    algo, env = _fd_algo()
    th = algo.theta().astype(float)
    idx = np.arange(64)
    jf = lambda t: algo._J(t, idx)[0]
    d = len(th)
    g_fd = np.array([(jf(th + 0.01 * e) - jf(th - 0.01 * e)) / 0.02 for e in np.eye(d)])
    g_ref = np.array([(jf(th + 1e-3 * e) - jf(th - 1e-3 * e)) / 2e-3 for e in np.eye(d)])
    assert np.allclose(g_fd, g_ref, rtol=5e-2, atol=2e-3)
    m = algo.iterate()
    assert m["days"] == algo.days_per_iter() == 64 * (2 * d + 1)


def test_spsa_unbiased_and_crn_reduces_variance():
    algo, env = _fd_algo("spsa", n_dir=1)
    th = algo.theta().astype(float)
    idx = np.arange(64)
    d = len(th)
    jf = lambda t, i=idx: algo._J(t, i)[0]
    g_true = np.array([(jf(th + 1e-3 * e) - jf(th - 1e-3 * e)) / 2e-3 for e in np.eye(d)])
    rng = np.random.default_rng(0)
    ests = []
    for _ in range(600):
        dl = rng.choice([-1.0, 1.0], d)
        ests.append((jf(th + .01 * dl) - jf(th - .01 * dl)) / .02 * dl)
    ests = np.array(ests)
    assert np.abs(ests.mean(0) - g_true).max() < 4 * ests.std(0).max() / np.sqrt(600) + 1e-3
    dl = np.ones(d)
    crn = [jf(th + .01 * dl, i) - jf(th - .01 * dl, i) for i in (rng.choice(300, 64) for _ in range(100))]
    ind = [jf(th + .01 * dl, rng.choice(300, 64)) - jf(th - .01 * dl, rng.choice(300, 64)) for _ in range(100)]
    assert np.var(crn) < np.var(ind)


# ---------- test 12: TRPO, PPO, DPG components ----------
def test_conjugate_gradient_solves_spd_system():
    rng = np.random.default_rng(0)
    M = torch.tensor(rng.normal(size=(6, 6)), dtype=torch.float32)
    A = M @ M.T + 6 * torch.eye(6)
    b = torch.tensor(rng.normal(size=6), dtype=torch.float32)
    x = conjugate_gradient(lambda v: A @ v, b, iters=30)
    assert torch.allclose(A @ x, b, atol=1e-3)


def test_kl_gauss_closed_form():
    kl = policies.kl_gauss(torch.zeros(4), torch.tensor(1.0), torch.zeros(4), torch.tensor(1.0))
    assert float(kl) == pytest.approx(0.0, abs=1e-7)
    kl = policies.kl_gauss(torch.zeros(1), torch.tensor(1.0), torch.ones(1), torch.tensor(1.0))
    assert float(kl) == pytest.approx(0.5, rel=1e-5)


def test_trpo_step_respects_kl_and_accepts():
    env = SynthEnv()
    t = TRPO(env, Risk("neutral"), {"B": 128, "delta": 0.01, "lam_gae": 0.0}, 0)
    kls, acc = [], 0
    for _ in range(15):
        m = t.iterate()
        kls.append(m["kl"]); acc += m["accepted"]
        assert m["kl"] <= 1.5 * 0.01 + 1e-9
    assert acc > 0 and max(kls) > 0


def test_ppo_clipped_objective_zeroes_gradient_outside_clip():
    ratio = torch.tensor([1.5, 0.5], requires_grad=True)
    adv = torch.tensor([1.0, -1.0])
    surr = torch.min(ratio * adv, torch.clamp(ratio, 0.8, 1.2) * adv).sum()
    g = torch.autograd.grad(surr, ratio)[0]
    assert g.tolist() == [0.0, 0.0]
    ratio2 = torch.tensor([1.1, 0.9], requires_grad=True)
    surr2 = torch.min(ratio2 * adv, torch.clamp(ratio2, 0.8, 1.2) * adv).sum()
    assert torch.autograd.grad(surr2, ratio2)[0].tolist() == [1.0, -1.0]


def test_dpg_critic_prefers_optimal_action():
    env = SynthEnv()
    d = DPG(env, Risk("neutral"), {"B": 32, "q_lr": 3e-3, "lr": 1e-9, "updates": 60, "warmup_iters": 0,
                                   "replay": 20000, "minibatch": 256, "init_sigma": 0.05}, 0)
    for _ in range(40):
        d.iterate()
    idx = np.arange(100)
    S = torch.cat([env.Z[idx], torch.zeros(100, 24, 1)], -1)[:, 0]
    a = env.astar(idx)[:, 0]
    with torch.no_grad():
        assert d.q1(S, a).mean() > d.q1(S, a + 0.1).mean()


# ---------- test 13: known optimum for every algorithm ----------
def _train(cls, cfg, iters, seed=0):
    torch.manual_seed(seed)
    env = SynthEnv()
    a = cls(env, Risk("neutral"), cfg, seed)
    for _ in range(iters):
        a.iterate()
    return env, a


@pytest.mark.parametrize("name,cls,cfg,iters", [
    ("fdpg", FDPG, {"fd": "central", "N": 64, "lr": 0.004, "eps": 0.01, "budget_days": 1}, 100),
    ("fdpg_spsa", FDPG, {"fd": "spsa", "N": 64, "n_dir": 4, "lr": 0.004, "eps": 0.01, "budget_days": 1}, 250),
    ("reinforce", Reinforce, {"B": 128, "lr": 0.004, "baseline": "loo"}, 300),
    ("a2c", A2C, {"B": 128, "lr": 0.004, "lam_gae": 0.0}, 300),
    ("trpo", TRPO, {"B": 128, "delta": 0.01, "lam_gae": 0.0}, 120),
    ("ppo", PPO, {"B": 128, "lr": 0.003, "lam_gae": 0.0, "epochs": 5}, 100),
])
def test_algorithms_recover_known_optimum(name, cls, cfg, iters):
    env, a = _train(cls, cfg, iters)
    th = a.theta()
    err = np.abs(th[:2] - SynthEnv.W).max()
    assert err < 0.015, (name, err)  # initial error is 0.04


def test_seed_reproducibility():
    r = []
    for _ in range(2):
        env, a = _train(Reinforce, {"B": 32, "lr": 0.004}, 5, seed=3)
        r.append(a.theta().copy())
    assert np.allclose(r[0], r[1])
    env, a = _train(Reinforce, {"B": 32, "lr": 0.004}, 5, seed=4)
    assert not np.allclose(r[0], a.theta())


# ---------- risk state updates ----------
def test_cvar_eta_tracks_alpha_quantile():
    rk = Risk("cvar", alpha=0.9, lr_eta=0.05, eta0=0.0)
    rng = np.random.default_rng(0)
    for _ in range(3000):
        rk.update(rng.normal(size=256))
    from scipy.stats import norm
    assert rk.eta == pytest.approx(norm.ppf(0.9), abs=0.15)


def test_meanvar_m_is_ema_and_penalty():
    rk = Risk("meanvar", kappa=2.0, ema=0.5, m0=0.0)
    rk.update(np.array([1.0, 3.0]))
    assert rk.m == pytest.approx(1.0)
    assert rk.penalty(np.array([3.0]))[0] == pytest.approx(2.0 * (3.0 - 1.0) ** 2)
