import numpy as np
import pytest

from candidates.base import Policy
from conftest import make_series
from data import split_bounds
from discretize import Discretizer
from env import DemandModel, make_params
from estimators import dr_estimate, wis_estimate
from fqe import fqe_fit, fqe_evaluate
from log_data import BehaviorPolicy, generate_log
from ranking_metrics import (metrics_for, pairwise_accuracy, precision_at_k, regret_at_1, spearman, kendall,
                             tie_matrix)

# ---- toy MDP with continuous (one-hot) features and exact policy values
nS, nA, GAMMA = 3, 2, 0.9
P_T = np.zeros((nS, nA, nS))
P_T[0, 0, 1] = P_T[0, 1, 2] = 1.0
P_T[1, 0, 0] = P_T[1, 1, 2] = 1.0
P_T[2, 0, 0] = P_T[2, 1, 2] = 1.0
R_T = np.array([[0.0, 0.5], [1.0, 0.0], [0.2, 1.0]])


def toy_log(n=20000, seed=0, trunc_frac=0.0):
    rng = np.random.default_rng(seed)
    s = rng.integers(nS, size=n)
    a = rng.integers(nA, size=n)
    s2 = P_T[s, a].argmax(1)
    eye = np.eye(nS, dtype=np.float32)
    return {"s_cont": eye[s], "a": a, "r": R_T[s, a].astype(np.float32), "s2_cont": eye[s2],
            "done": np.zeros(n, bool), "truncated": rng.random(n) < trunc_frac, "s0": eye[np.zeros(50, int)]}


def exact_value(pi, s0=0):
    Ppi = np.einsum("sa,san->sn", pi, P_T)
    return np.linalg.solve(np.eye(nS) - GAMMA * Ppi, (pi * R_T).sum(1))[s0]


def det_probs(table):
    M = np.zeros((nS, nA)); M[np.arange(nS), table] = 1.0
    return M, (lambda S: M[S.argmax(1)])


def test_fqe_recovers_exact_values_and_ranks():
    log = toy_log(trunc_frac=0.2)           # truncated flags must NOT stop bootstrapping
    pis = [det_probs(np.array([1, 0, 1])), det_probs(np.array([0, 1, 0])), det_probs(np.array([1, 1, 1]))]
    truths, ests = [], []
    for M, fn in pis:
        truths.append(exact_value(M))
        r = fqe_evaluate(fn, log, 1.0, GAMMA, seeds=2, rounds=80, steps=150, hidden=32, batch=512)
        ests.append(r["value"])
        assert r["value"] == pytest.approx(exact_value(M), abs=0.35)
    assert np.argsort(ests).tolist() == np.argsort(truths).tolist()


def test_fqe_target_uses_evaluated_policy_action_not_max():
    log = toy_log()
    M_opt = det_probs(np.array([1, 0, 1]))   # near-optimal in this toy
    M_bad = det_probs(np.array([0, 1, 0]))
    vo = fqe_evaluate(M_opt[1], log, 1.0, GAMMA, 1, rounds=80, steps=150, hidden=32)["value"]
    vb = fqe_evaluate(M_bad[1], log, 1.0, GAMMA, 1, rounds=80, steps=150, hidden=32)["value"]
    assert vb < vo - 0.5
    assert vb == pytest.approx(exact_value(M_bad[0]), abs=0.4)       # a max-target would give ~ the optimal value


def test_fqe_stochastic_policy_exact_expectation():
    log = toy_log()
    pi = np.full((nS, nA), 0.5)
    fn = lambda S: pi[S.argmax(1)]
    v = fqe_evaluate(fn, log, 1.0, GAMMA, 1, rounds=80, steps=150, hidden=32)["value"]
    assert v == pytest.approx(exact_value(pi), abs=0.35)


def _setup(cfg):
    ser = make_series()
    p = make_params(ser, cfg)
    lo, hi = split_bounds(len(ser), cfg)["train"]
    dm = DemandModel(ser.sales[lo:hi], ser.wday[lo:hi], ser.snap[lo:hi])
    return ser, p, dm


def test_is_with_target_equal_behavior_gives_unit_weights(cfg):
    ser, p, dm = _setup(cfg)
    disc = Discretizer(p)
    beh = BehaviorPolicy(p, [5 * p.mean_demand], 0.3)
    log = generate_log(ser, cfg, p, dm, disc, beh, "train", np.arange(40), 1)
    v, ess = wis_estimate(beh.probs, log, p.gamma, clip=100.0)
    G = np.array([(log["r"][log["episode_id"] == i] * p.gamma ** np.arange(cfg["env"]["horizon"])).sum()
                  for i in np.unique(log["episode_id"])])
    assert ess == pytest.approx(40.0)
    assert v == pytest.approx(G.mean())


# ---- ranking metrics
def test_ranking_metrics_basic():
    t = np.array([1.0, 2, 3, 4, 5])
    assert spearman(t, t) == pytest.approx(1) and spearman(t[::-1], t) == pytest.approx(-1)
    assert kendall(t, t) == pytest.approx(1) and kendall(t[::-1], t) == pytest.approx(-1)
    assert regret_at_1(t, t) == (0.0, 0.0)
    r, nr = regret_at_1(t[::-1], t)
    assert r == 4 and nr == 1.0
    assert precision_at_k(t, t, 3) == 1.0 and precision_at_k(t[::-1], t, 3) == pytest.approx(1 / 3)
    assert pairwise_accuracy(t, t) == 1.0 and pairwise_accuracy(t[::-1], t) == 0.0
    tie = np.zeros((5, 5), bool); tie[0, 1] = tie[1, 0] = True
    est = np.array([2.0, 1, 3, 4, 5])        # swaps tied pair only
    assert pairwise_accuracy(est, t, tie) == 1.0
    assert pairwise_accuracy(est, t) < 1.0
    m = metrics_for(t, t, ci=False)
    assert m["regret@1"] == 0 and m["spearman"] == pytest.approx(1)


def test_tie_matrix_detects_ties():
    rng = np.random.default_rng(0)
    base = rng.normal(size=200)
    ret = {"a": base, "b": base + rng.normal(scale=0.01, size=200), "c": base + 5}
    tie = tie_matrix(ret, ["a", "b", "c"])
    assert tie[0, 1] and not tie[0, 2]
