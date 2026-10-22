"""
train.py -- train a DQN agent per series (train split), with periodic
validation, logging, checkpointing and training-curve plots.

Usage:
    python train.py --config config.yaml --series FOODS_3_090_CA_3_evaluation
    python train.py --config config.yaml --all     # train on every series in config
"""
from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import load_all
from env import InventoryEnv, CostConfig
from dqn import DQNAgent, DQNConfig


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_env(bundle, split, cfg, seed=None):
    ecfg = cfg["env"]
    cost = CostConfig(
        unit_cost_frac=ecfg["unit_cost_frac"],
        holding_cost_frac=ecfg["holding_cost_frac"],
        stockout_penalty_frac=ecfg["stockout_penalty_frac"],
        fixed_order_cost=ecfg["fixed_order_cost"],
    )
    return InventoryEnv(
        bundle,
        split=split,
        window_len=ecfg["window_len"],
        lead_time=ecfg["lead_time"],
        init_inventory_days=ecfg["init_inventory_days"],
        order_grid_multiples=ecfg["order_grid_multiples"],
        cost_config=cost,
        seed=seed,
    )


def evaluate_greedy(agent: DQNAgent, env: InventoryEnv, n_episodes: int, seed: int):
    rng = np.random.default_rng(seed)
    returns, fills, stockouts = [], [], []
    for _ in range(n_episodes):
        obs, info = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        done = False
        ep_return = 0.0
        ep_fill, ep_stock = [], []
        while not done:
            a = agent.act(obs, greedy=True)
            obs, r, term, trunc, step_info = env.step(a)
            done = term or trunc
            ep_return += r
            ep_fill.append(step_info["fill_rate"])
            ep_stock.append(step_info["stockout"])
        returns.append(ep_return)
        fills.append(np.mean(ep_fill))
        stockouts.append(np.mean(ep_stock))
    return float(np.mean(returns)), float(np.mean(fills)), float(np.mean(stockouts))


def train_one_series(series_id: str, bundle, cfg: dict, out_dir: str, ablation_tag: str = "full"):
    seed = cfg.get("seed", 42)
    set_seed(seed)

    train_env = make_env(bundle, "train", cfg, seed=seed)
    val_env = make_env(bundle, "val", cfg, seed=seed + 1)

    dcfg = cfg["dqn"]
    use_replay = dcfg.get("use_replay", True) and (ablation_tag != "no_replay")
    use_target = dcfg.get("use_target_network", True) and (ablation_tag != "no_target")

    agent_cfg = DQNConfig(
        state_dim=train_env.observation_space.shape[0],
        n_actions=train_env.action_space.n,
        hidden_sizes=tuple(dcfg["hidden_sizes"]),
        gamma=dcfg["gamma"],
        lr=dcfg["lr"],
        batch_size=dcfg["batch_size"],
        buffer_capacity=dcfg["buffer_capacity"],
        warmup_steps=dcfg["warmup_steps"],
        target_update_mode=dcfg["target_update_mode"],
        target_sync_every=dcfg["target_sync_every"],
        tau=dcfg["tau"],
        double_dqn=dcfg["double_dqn"],
        grad_clip_norm=dcfg["grad_clip_norm"],
        eps_start=dcfg["eps_start"],
        eps_end=dcfg["eps_end"],
        eps_decay_steps=dcfg["eps_decay_steps"],
        use_replay=use_replay,
        use_target_network=use_target,
        seed=seed,
    )
    agent = DQNAgent(agent_cfg)

    history = {"episode": [], "return": [], "avg_loss": [], "avg_max_q": [],
               "epsilon": [], "fill_rate": [], "stockout_rate": [],
               "val_return": []}

    num_episodes = dcfg["num_episodes"]
    for ep in range(num_episodes):
        obs, info = train_env.reset(seed=seed + ep)
        done = False
        ep_return = 0.0
        losses, max_qs, fills, stockouts = [], [], [], []
        while not done:
            a = agent.act(obs)
            next_obs, r, term, trunc, step_info = train_env.step(a)
            done = term or trunc
            agent.store(obs, a, r, next_obs, done)
            loss = agent.learn_step()
            agent.step_counter()
            if loss is not None:
                losses.append(loss)
                max_qs.append(agent._last_max_q)
            obs = next_obs
            ep_return += r
            fills.append(step_info["fill_rate"])
            stockouts.append(step_info["stockout"])

        history["episode"].append(ep)
        history["return"].append(ep_return)
        history["avg_loss"].append(float(np.mean(losses)) if losses else np.nan)
        history["avg_max_q"].append(float(np.mean(max_qs)) if max_qs else np.nan)
        history["epsilon"].append(agent.epsilon())
        history["fill_rate"].append(float(np.mean(fills)))
        history["stockout_rate"].append(float(np.mean(stockouts)))

        if ep % dcfg["eval_every"] == 0 or ep == num_episodes - 1:
            val_return, val_fill, val_stock = evaluate_greedy(agent, val_env, n_episodes=5, seed=123)
            history["val_return"].append((ep, val_return))
            print(
                f"[{series_id}][{ablation_tag}] ep {ep:4d} return={ep_return:9.2f} "
                f"val_return={val_return:9.2f} eps={agent.epsilon():.3f} "
                f"fill={history['fill_rate'][-1]:.2f} stockout={history['stockout_rate'][-1]:.2f}"
            )

    os.makedirs(out_dir, exist_ok=True)
    safe_id = series_id.replace("/", "_")
    ckpt_path = os.path.join(out_dir, f"{safe_id}_{ablation_tag}.pt")
    agent.save(ckpt_path)
    plot_training_curves(history, os.path.join(out_dir, f"{safe_id}_{ablation_tag}_curves.png"), series_id, ablation_tag)
    np.savez(os.path.join(out_dir, f"{safe_id}_{ablation_tag}_history.npz"),
             **{k: np.array(v, dtype=object) for k, v in history.items()})
    return agent, history


def plot_training_curves(history, path, series_id, ablation_tag):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].plot(history["episode"], history["return"])
    axes[0, 0].set_title("Episode return")
    axes[0, 1].plot(history["episode"], history["avg_loss"])
    axes[0, 1].set_title("Avg Huber loss")
    axes[1, 0].plot(history["episode"], history["avg_max_q"])
    axes[1, 0].set_title("Avg max-Q")
    axes[1, 1].plot(history["episode"], history["fill_rate"], label="fill rate")
    axes[1, 1].plot(history["episode"], history["stockout_rate"], label="stockout rate")
    axes[1, 1].legend()
    axes[1, 1].set_title("Fill / stockout rate")
    fig.suptitle(f"{series_id} ({ablation_tag})")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--series", default=None, help="single series id to train (default: all in config)")
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--ablation", default="full", choices=["full", "no_replay", "no_target"])
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    bundles = load_all(cfg)
    series_ids = [args.series] if args.series else list(bundles.keys())

    for sid in series_ids:
        print(f"=== Training on {sid} (ablation={args.ablation}) ===")
        train_one_series(sid, bundles[sid], cfg, args.out_dir, ablation_tag=args.ablation)


if __name__ == "__main__":
    main()
