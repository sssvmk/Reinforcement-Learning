"""Sanity checks #3 and #4: replay buffer correctness and target-network sync behavior."""
import numpy as np
import torch

from dqn import ReplayBuffer, DQNAgent, DQNConfig


def test_replay_buffer_capacity_and_overwrite():
    cap = 10
    buf = ReplayBuffer(capacity=cap, state_dim=3, seed=0)
    for i in range(25):
        s = np.full(3, i, dtype=np.float32)
        buf.push(s, i % 4, float(i), s + 1, done=(i % 5 == 0))

    assert len(buf) == cap  # capped at capacity, not 25
    # After 25 pushes into a capacity-10 buffer, the buffer should hold the
    # last 10 entries (indices 15..24), in circular order.
    expected_last_state_values = set(range(15, 25))
    actual_values = set(buf.states[:, 0].astype(int).tolist())
    assert actual_values == expected_last_state_values


def test_replay_buffer_sample_shapes():
    buf = ReplayBuffer(capacity=100, state_dim=5, seed=0)
    for i in range(50):
        buf.push(np.random.randn(5), i % 3, float(i), np.random.randn(5), False)

    batch_size = 16
    s, a, r, s2, d = buf.sample(batch_size)
    assert s.shape == (batch_size, 5)
    assert a.shape == (batch_size,)
    assert r.shape == (batch_size,)
    assert s2.shape == (batch_size, 5)
    assert d.shape == (batch_size,)


def test_target_network_hard_sync():
    cfg = DQNConfig(state_dim=4, n_actions=3, target_update_mode="hard",
                     target_sync_every=5, warmup_steps=0, batch_size=2, buffer_capacity=50, seed=0)
    agent = DQNAgent(cfg)

    # Right after construction, target == online.
    for p_online, p_target in zip(agent.q.parameters(), agent.target_q.parameters()):
        assert torch.allclose(p_online, p_target)

    # Fill buffer and take a few learn steps (fewer than sync_every): target
    # network parameters must stay fixed while the online net changes.
    for i in range(3):
        s = np.random.randn(4).astype(np.float32)
        s2 = np.random.randn(4).astype(np.float32)
        agent.store(s, i % 3, 1.0, s2, False)

    target_before = [p.clone() for p in agent.target_q.parameters()]
    for step in range(1, 4):  # fewer steps than target_sync_every=5
        agent.total_steps = step
        agent.learn_step()
    for p_before, p_after in zip(target_before, agent.target_q.parameters()):
        assert torch.allclose(p_before, p_after), "target network changed before a sync point"

    online_before_sync = [p.clone() for p in agent.q.parameters()]
    agent.total_steps = 5  # hits target_sync_every -> hard sync happens inside learn_step
    agent.learn_step()
    for p_online, p_target in zip(agent.q.parameters(), agent.target_q.parameters()):
        assert torch.allclose(p_online, p_target), "hard sync did not copy online weights to target"
