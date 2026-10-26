import numpy as np
import pytest
import torch

from candidates.bc import BCPolicy, train_bc
from candidates.cql import train_cql
from candidates.dqn import conservative_penalty, train_dqn
from candidates.mcts_policy import MCTSPolicy
from conftest import make_series
from env import DemandModel, make_params, unpack


def test_bc_learns_deterministic_policy():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, 4)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int) + 2 * (X[:, 2] > 0.5)
    net = train_bc(X, y, 4, hidden=64, epochs=60, seed=0)
    pol = BCPolicy("bc", net, 4, False, {})
    assert (pol.greedy(X) == y).mean() > 0.97


def toy_transitions(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    S = rng.normal(size=(n, 3)).astype(np.float32)
    A = rng.integers(0, 3, n)
    R = (S[:, 0] * (A == 1) + 0.1 * rng.normal(size=n)).astype(np.float32)
    S2 = rng.normal(size=(n, 3)).astype(np.float32)
    return S, A, R, S2, np.zeros(n, np.float32)


def _flat(sd):
    return torch.cat([v.flatten() for v in sd.values()])


def test_target_network_frozen_between_syncs():
    S, A, R, S2, D = toy_transitions()
    kw = dict(n_actions=3, r_scale=1.0, gamma=0.9, seed=1, sync_every=100, return_net=True)
    a = train_dqn(S, A, R, S2, D, steps=10, **kw)
    b = train_dqn(S, A, R, S2, D, steps=60, **kw)
    assert torch.equal(_flat(a["tgt"].state_dict()), _flat(b["tgt"].state_dict()))      # no sync yet
    assert not torch.equal(_flat(a["q"].state_dict()), _flat(b["q"].state_dict()))
    c = train_dqn(S, A, R, S2, D, steps=100, **kw)
    assert torch.equal(_flat(c["tgt"].state_dict()), _flat(c["q"].state_dict()))        # synced at step 100


def test_cql_alpha0_reproduces_dqn_and_penalty_nonnegative():
    S, A, R, S2, D = toy_transitions()
    kw = dict(n_actions=3, r_scale=1.0, gamma=0.9, seed=2)
    d = train_dqn(S, A, R, S2, D, steps=200, **kw)["snapshots"][200]
    c0 = train_cql(S, A, R, S2, D, n_actions=3, r_scale=1.0, gamma=0.9, steps=200, alpha=0.0, seed=2)["snapshots"][200]
    c_eps = train_cql(S, A, R, S2, D, n_actions=3, r_scale=1.0, gamma=0.9, steps=200, alpha=1e-12, seed=2)["snapshots"][200]
    c1 = train_cql(S, A, R, S2, D, n_actions=3, r_scale=1.0, gamma=0.9, steps=200, alpha=1.0, seed=2)["snapshots"][200]
    assert torch.equal(_flat(d), _flat(c0))
    assert torch.allclose(_flat(d), _flat(c_eps), atol=1e-5)
    assert not torch.allclose(_flat(d), _flat(c1), atol=1e-3)
    Q = torch.randn(64, 5) * 3
    a = torch.randint(0, 5, (64,))
    assert conservative_penalty(Q, a).item() >= 0


def test_dqn_loss_decreases():
    S, A, R, S2, D = toy_transitions(500)
    r = train_dqn(S, A, R, S2, D, n_actions=3, r_scale=1.0, gamma=0.0, steps=600, seed=0)
    td = [x[1] for x in r["log"]]
    assert np.mean(td[-3:]) < 0.8 * np.mean(td[:3])


def _const_setup(cfg, kind="mean", S_roll=None):
    ser = make_series(constant=5)
    p = make_params(ser, cfg)
    dm = DemandModel(ser.sales[:600], ser.wday[:600], ser.snap[:600], kind=kind)
    return p, dm


def _state(p, on, pipe, wd=0):
    from env import features
    return features(np.array([on] + list(pipe), float), wd, 0, 0, p.mean_price, p.mean_demand, p)


def test_mcts_legal_action_and_constant_demand(cfg):
    p, dm = _const_setup(cfg)
    pol = MCTSPolicy("m", p, dm, budget=300, c=1.0, h_plan=10, S_rollout=3 * p.mean_demand, max_decisions=100, seed=0)
    s = _state(p, 5, [5, 0])
    a = pol.act(s)
    assert 0 <= a < p.n_actions
    q = p.actions[a]
    assert 3 <= q <= 10, q                                  # about demand (5/day)
    assert p.actions[pol.act(_state(p, 100, [100, 100]))] == 0        # plenty of stock: no order
    assert p.actions[pol.act(_state(p, 0, [0, 0]))] > 0               # empty: order


def test_mcts_budget_fallback_and_cache(cfg):
    p, dm = _const_setup(cfg)
    pol = MCTSPolicy("m", p, dm, budget=20, h_plan=5, S_rollout=15.0, max_decisions=1, seed=0)
    pol.act(_state(p, 5, [5, 5])); pol.act(_state(p, 1, [0, 0]))
    assert pol.n_planned == 1 and pol.n_fallback == 1
    pol.act(_state(p, 5, [5, 5]))
    assert pol.n_planned == 1                                        # served from cache
