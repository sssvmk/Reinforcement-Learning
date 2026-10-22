"""Sanity check #5: DQN on a toy deterministic environment (constant demand of
5 units/day) should learn to order about 5/day, matching the optimal policy."""
import numpy as np
import torch

from env import InventoryEnv, CostConfig
from dqn import DQNAgent, DQNConfig


def test_dqn_converges_on_constant_demand(constant_demand_bundle):
    torch.manual_seed(0)
    np.random.seed(0)

    cost = CostConfig(unit_cost_frac=0.6, holding_cost_frac=0.005,
                       stockout_penalty_frac=0.2, fixed_order_cost=0.1)
    env = InventoryEnv(
        constant_demand_bundle, split="train", window_len=30, lead_time=1,
        init_inventory_days=3.0, cost_config=cost, seed=0,
    )

    cfg = DQNConfig(
        state_dim=env.observation_space.shape[0],
        n_actions=env.action_space.n,
        hidden_sizes=(64, 64),
        gamma=0.95,
        lr=1e-3,
        batch_size=32,
        buffer_capacity=5000,
        warmup_steps=200,
        target_update_mode="hard",
        target_sync_every=100,
        double_dqn=True,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay_steps=3000,
        seed=0,
    )
    agent = DQNAgent(cfg)

    n_episodes = 150
    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep)
        done = False
        while not done:
            a = agent.act(obs)
            next_obs, r, term, trunc, step_info = env.step(a)
            done = term or trunc
            agent.store(obs, a, r, next_obs, done)
            agent.learn_step()
            agent.step_counter()
            obs = next_obs

    # Evaluate greedily, skipping the first few transient days, and check the
    # mean chosen order quantity is close to the true mean demand (5/day).
    chosen_orders = []
    obs, info = env.reset(seed=999)
    done = False
    while not done:
        a = agent.act(obs, greedy=True)
        order_qty = env.order_grid[a]
        obs, r, term, trunc, step_info = env.step(a)
        done = term or trunc
        chosen_orders.append(order_qty)

    mean_order = np.mean(chosen_orders[5:])  # skip initial transient
    assert abs(mean_order - 5.0) <= 2.5, f"expected ~5 units/day, got {mean_order}"
