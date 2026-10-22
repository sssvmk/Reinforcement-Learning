"""
evaluate.py -- evaluate DQN checkpoints and classical baselines on the
held-out test split, over multiple windows and seeds, and produce a results
table + plots.

Usage:
    python evaluate.py --config config.yaml --out_dir outputs
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import yaml

from data import load_all
from env import InventoryEnv, CostConfig
from dqn import DQNAgent, DQNConfig
from baselines import (
    never_order_policy,
    fixed_order_policy,
    base_stock_policy,
    s_S_policy,
    newsvendor_policy,
    simulate_policy,
    tune_base_stock,
    tune_s_S,
    tune_newsvendor,
)
from train import make_env, set_seed


def dqn_policy_from_agent(agent: DQNAgent):
    def policy(env: InventoryEnv) -> int:
        # obs is reconstructed by env internally; agent needs obs, so this
        # wrapper is only used via the custom loop in evaluate_full, not
        # simulate_policy (kept here for API symmetry / potential reuse).
        raise NotImplementedError("use evaluate_dqn() for the DQN policy")

    return policy


def evaluate_dqn(env: InventoryEnv, agent: DQNAgent, n_episodes: int, seed: int):
    rng = np.random.default_rng(seed)
    returns, fills, stockouts, invs, n_orders = [], [], [], [], []
    for _ in range(n_episodes):
        obs, info = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        done = False
        ep_return = 0.0
        fs, ss, invl = [], [], []
        orders = 0
        while not done:
            a = agent.act(obs, greedy=True)
            obs, r, term, trunc, step_info = env.step(a)
            done = term or trunc
            ep_return += r
            fs.append(step_info["fill_rate"])
            ss.append(step_info["stockout"])
            invl.append(step_info["on_hand"])
            if step_info["order_qty"] > 0:
                orders += 1
        returns.append(ep_return)
        fills.append(np.mean(fs))
        stockouts.append(np.mean(ss))
        invs.append(np.mean(invl))
        n_orders.append(orders)
    return {
        "mean_return": float(np.mean(returns)), "std_return": float(np.std(returns)),
        "mean_fill_rate": float(np.mean(fills)), "mean_stockout_rate": float(np.mean(stockouts)),
        "mean_inventory": float(np.mean(invs)), "mean_n_orders": float(np.mean(n_orders)),
    }


def build_agent_for_env(env: InventoryEnv, cfg: dict, ckpt_path: str) -> DQNAgent:
    dcfg = cfg["dqn"]
    agent_cfg = DQNConfig(
        state_dim=env.observation_space.shape[0],
        n_actions=env.action_space.n,
        hidden_sizes=tuple(dcfg["hidden_sizes"]),
        gamma=dcfg["gamma"], lr=dcfg["lr"], batch_size=dcfg["batch_size"],
        buffer_capacity=dcfg["buffer_capacity"], warmup_steps=dcfg["warmup_steps"],
        target_update_mode=dcfg["target_update_mode"], target_sync_every=dcfg["target_sync_every"],
        tau=dcfg["tau"], double_dqn=dcfg["double_dqn"], grad_clip_norm=dcfg["grad_clip_norm"],
        eps_start=dcfg["eps_start"], eps_end=dcfg["eps_end"], eps_decay_steps=dcfg["eps_decay_steps"],
        seed=cfg.get("seed", 42),
    )
    agent = DQNAgent(agent_cfg)
    agent.load(ckpt_path)
    return agent


def evaluate_series(series_id: str, bundle, cfg: dict, out_dir: str) -> pd.DataFrame:
    n_windows = cfg["evaluation"]["n_test_windows"]
    n_seeds = cfg["evaluation"]["n_seeds"]
    base_seed = cfg.get("seed", 42)

    train_env = make_env(bundle, "train", cfg, seed=base_seed)
    test_env = make_env(bundle, "test", cfg, seed=base_seed + 999)
    mean_d = bundle.mean_demand
    L = cfg["env"]["lead_time"]
    rp = cfg["baselines"]["base_stock_review_period"]

    rows = []

    # ---- DQN across seeds ----
    safe_id = series_id.replace("/", "_")
    ckpt_path = os.path.join(out_dir, f"{safe_id}_full.pt")
    if os.path.exists(ckpt_path):
        for seed_i in range(n_seeds):
            agent = build_agent_for_env(test_env, cfg, ckpt_path)
            m = evaluate_dqn(test_env, agent, n_episodes=n_windows, seed=base_seed + 100 * seed_i)
            m.update({"series_id": series_id, "policy": "DQN", "seed": seed_i})
            rows.append(m)
    else:
        print(f"[warn] no checkpoint found at {ckpt_path}, skipping DQN for {series_id}")

    # ---- Baselines (tuned on train split, evaluated on test split) ----
    _, never_metrics = (never_order_policy(), None)
    never_m = simulate_policy(test_env, never_order_policy(), n_episodes=n_windows, seed=base_seed)
    never_m.update({"series_id": series_id, "policy": "NeverOrder", "seed": 0})
    rows.append(never_m)

    fixed_m = simulate_policy(test_env, fixed_order_policy(mean_d), n_episodes=n_windows, seed=base_seed)
    fixed_m.update({"series_id": series_id, "policy": "FixedOrder(mean_d)", "seed": 0})
    rows.append(fixed_m)

    best_S, _ = tune_base_stock(train_env, mean_d, L, review_period=rp, n_episodes=10, seed=base_seed)
    bs_m = simulate_policy(test_env, base_stock_policy(best_S), n_episodes=n_windows, seed=base_seed)
    bs_m.update({"series_id": series_id, "policy": f"BaseStock(S={best_S:.1f})", "seed": 0})
    rows.append(bs_m)

    (best_s, best_S2), _ = tune_s_S(
        train_env, mean_d,
        cfg["baselines"]["s_S_grid_s_multiples"], cfg["baselines"]["s_S_grid_S_multiples"],
        n_episodes=10, seed=base_seed,
    )
    sS_m = simulate_policy(test_env, s_S_policy(best_s, best_S2), n_episodes=n_windows, seed=base_seed)
    sS_m.update({"series_id": series_id, "policy": f"(s={best_s:.1f},S={best_S2:.1f})", "seed": 0})
    rows.append(sS_m)

    train_demand_hist = bundle.demand[bundle.train_range[0]:bundle.train_range[1]]
    best_q, _ = tune_newsvendor(
        train_env, train_demand_hist, cfg["baselines"]["newsvendor_quantiles"], L, review_period=rp,
        n_episodes=10, seed=base_seed,
    )
    level = float(np.quantile(train_demand_hist, best_q)) * (L + rp)
    nv_m = simulate_policy(test_env, newsvendor_policy(level), n_episodes=n_windows, seed=base_seed)
    nv_m.update({"series_id": series_id, "policy": f"Newsvendor(q={best_q})", "seed": 0})
    rows.append(nv_m)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out_dir", default="outputs")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg.get("seed", 42))

    bundles = load_all(cfg)
    all_rows = []
    for sid, bundle in bundles.items():
        print(f"=== Evaluating {sid} ===")
        df = evaluate_series(sid, bundle, cfg, args.out_dir)
        all_rows.append(df)

    results = pd.concat(all_rows, ignore_index=True)
    summary = results.groupby(["series_id", "policy"]).agg(
        mean_return=("mean_return", "mean"), std_return=("mean_return", "std"),
        mean_fill_rate=("mean_fill_rate", "mean"), mean_stockout_rate=("mean_stockout_rate", "mean"),
        mean_inventory=("mean_inventory", "mean"), mean_n_orders=("mean_n_orders", "mean"),
    ).reset_index()

    os.makedirs(args.out_dir, exist_ok=True)
    results.to_csv(os.path.join(args.out_dir, "results_raw.csv"), index=False)
    summary.to_csv(os.path.join(args.out_dir, "results_summary.csv"), index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
