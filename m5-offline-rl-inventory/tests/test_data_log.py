import numpy as np
import pytest

from candidates.references import BaseStock
from conftest import make_series
from data import load_config, split_bounds
from discretize import Discretizer
from env import DemandModel, make_params, make_episodes
from log_data import BehaviorPolicy, generate_log, split_log_ids, subset_log, tune_base_stock


def test_splits_chronological_disjoint(cfg):
    for n in [500, 1913, 1200]:
        b = split_bounds(n, cfg)
        assert b["train"][0] == 0 and b["train"][1] == b["val"][0] and b["val"][1] == b["test"][0] and b["test"][1] == n
        assert b["train"][1] - b["train"][0] > b["val"][1] - b["val"][0]


def test_train_and_fqe_logs_share_no_episode_ids():
    tr, fq = split_log_ids(200, 0.4, seed=1)
    assert not set(tr) & set(fq) and len(tr) + len(fq) == 200


def _build(series, cfg):
    p = make_params(series, cfg)
    lo, hi = split_bounds(len(series), cfg)["train"]
    dm = DemandModel(series.sales[lo:hi], series.wday[lo:hi], series.snap[lo:hi])
    return p, dm


def test_no_test_period_information_leaks(cfg):
    s1 = make_series()
    s2 = make_series()
    t0 = split_bounds(len(s1), cfg)["test"][0]
    s2.sales[t0:] = s2.sales[t0:] * 50 + 7           # corrupt only the test period (demand and prices)
    s2.price[t0:] = 99.0
    (p1, dm1), (p2, dm2) = _build(s1, cfg), _build(s2, cfg)
    assert np.array_equal(p1.actions, p2.actions) and p1.mean_demand == p2.mean_demand and p1.mean_price == p2.mean_price
    for k in dm1.table:
        assert dm1.table[k] == dm2.table[k]
    assert tune_base_stock(s1, cfg, p1, dm1, n_ep=10) == tune_base_stock(s2, cfg, p2, dm2, n_ep=10)
    disc = Discretizer(p1)
    beh = BehaviorPolicy(p1, [50.0], 0.2)
    l1 = generate_log(s1, cfg, p1, dm1, disc, beh, "train", np.arange(20), 0)
    l2 = generate_log(s2, cfg, p2, dm2, disc, beh, "train", np.arange(20), 0)
    for k in l1:
        assert np.array_equal(l1[k], l2[k]), k


def test_log_shapes_and_behavior_probs(cfg, series):
    p, dm = _build(series, cfg)
    disc = Discretizer(p)
    eps = 0.3
    beh = BehaviorPolicy(p, [5 * p.mean_demand], eps)
    log = generate_log(series, cfg, p, dm, disc, beh, "train", np.arange(300), 4)
    M, H = 300 * cfg["env"]["horizon"], cfg["env"]["horizon"]
    assert log["s_cont"].shape == (M, p.state_dim) and log["s2_cont"].shape == (M, p.state_dim)
    assert len(log["a"]) == len(log["r"]) == len(log["behavior_prob"]) == M
    bp = log["behavior_prob"]
    assert bp.min() > 0 and bp.max() <= 1
    assert log["truncated"].sum() == 300 and not log["done"].any()
    assert (log["t"][log["truncated"]] == H - 1).all()
    greedy = BaseStock(p, 5 * p.mean_demand).greedy(log["s_cont"])
    assert (log["a"] == greedy).mean() == pytest.approx(1 - eps + eps / p.n_actions, abs=0.01)
    P = beh.probs(log["s_cont"][:2000])
    assert np.allclose(P.sum(1), 1)
    assert bp.mean() == pytest.approx((beh.probs(log["s_cont"]) ** 2).sum(1).mean(), abs=0.01)
    assert np.array_equal(log["s_discrete"], disc(log["s_cont"]))


def test_discretizer_range(cfg, series):
    p, dm = _build(series, cfg)
    disc = Discretizer(p)
    beh = BehaviorPolicy(p, [30.0], 0.5)
    log = generate_log(series, cfg, p, dm, disc, beh, "train", np.arange(50), 0)
    assert log["s_discrete"].min() >= 0 and log["s_discrete"].max() < disc.n_states
    assert 100 <= disc.n_states <= 5000


def test_common_random_numbers(cfg, series):
    p, dm = _build(series, cfg)
    e1 = make_episodes(series, cfg, dm, "test", np.arange(5), 9)
    e2 = make_episodes(series, cfg, dm, "test", np.arange(5), 9)
    e3 = make_episodes(series, cfg, dm, "test", np.arange(5), 10)
    assert np.array_equal(e1.demand, e2.demand) and not np.array_equal(e1.demand, e3.demand)
    # an episode's demand does not depend on which other ids are requested
    e4 = make_episodes(series, cfg, dm, "test", np.array([3, 4]), 9)
    assert np.array_equal(e4.demand, e1.demand[3:5])
