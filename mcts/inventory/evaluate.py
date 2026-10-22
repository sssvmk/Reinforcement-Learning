"""Experiment runner for Part B: MCTS vs. classical inventory baselines,
evaluated in the SAME InventoryEnv on the SAME test windows, for one or
more M5 series.

Data split is chronological (data_split(): earliest 70% = train, next 15%
= validation, last 15% = test -- see config.yaml `inventory.split`).
Baselines are tuned on TRAIN only (baselines.py). The demand model used by
MCTS is fit on TRAIN only (demand_model.py). Both are then evaluated,
unmodified, on TEST windows, with the real historical test-period demand
(never the demand model) driving InventoryEnv.step().

Usage:
    python evaluate.py --quick                         # fast smoke run
    python evaluate.py                                 # full run (5 series, 5 seeds)
    python evaluate.py --series FOODS_3_477_CA_3_evaluation
    python evaluate.py --ablation budget                # sim-budget ablation (1st series)
    python evaluate.py --ablation horizon|c|demand_model|rollout|lead_time
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from data import load_series
from inventory_env import InventoryConfig, InventoryEnv, InventoryState, make_action_grid, data_split
from demand_model import make_demand_model
from baselines import build_all_baselines
from inventory_mcts import MCTSInventoryConfig, ReceedingHorizonPlanner

ROOT = Path(__file__).resolve().parent.parent
OUT_RESULTS = ROOT / "outputs" / "results"
OUT_PLOTS = ROOT / "outputs" / "plots"


def load_config() -> dict:
    with open(ROOT / "config.yaml", "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Episode runner: drives InventoryEnv with REAL historical test-period
# demand for a stateless policy fn(state)->action_idx, or an MCTS planner
# with .act(state)->action_idx. Both are driven identically through
# env.step(), so they are compared under exactly the same dynamics.
# ---------------------------------------------------------------------------

def run_episode(env: InventoryEnv, act_fn: Callable[[InventoryState], int], start_idx: int) -> Dict:
    obs, info = env.reset(start_idx=start_idx)
    total_reward = 0.0
    on_hand_trace = []
    n_orders = 0
    done = False
    while not done:
        a = act_fn(env.state)
        order_qty = env.action_grid[a]
        if order_qty > 0:
            n_orders += 1
        obs, r, term, trunc, step_info = env.step(a)
        total_reward += r
        on_hand_trace.append(step_info["on_hand_end"])
        done = term or trunc
    final_state = env.state
    total_demand = final_state.cum_sales + final_state.cum_lost_sales
    fill_rate = final_state.cum_sales / total_demand if total_demand > 0 else 1.0
    stockout_rate = final_state.cum_lost_sales / total_demand if total_demand > 0 else 0.0
    return {
        "profit": total_reward,
        "fill_rate": fill_rate,
        "stockout_rate": stockout_rate,
        "avg_inventory": float(np.mean(on_hand_trace)) if on_hand_trace else 0.0,
        "n_orders": n_orders,
    }


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    if len(values) == 0:
        return (float("nan"), float("nan"))
    boots = np.array([rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def _test_window_starts(test_idx: np.ndarray, episode_days: int, lead_time_L: int, n_windows: int,
                         seed: int) -> List[int]:
    """Deterministic set of `n_windows` start POSITIONS within the test
    split's day_indices array (i.e. values directly usable as the
    `start_idx` argument to InventoryEnv.reset() when that env was built
    with day_indices=test_idx -- reset() indexes day_indices[start_pos]
    itself, so these must be positions, NOT absolute day indices).
    Reused IDENTICALLY across every policy (fair comparison on the 'same
    test windows' per spec). Windows are spread as evenly as possible
    across the test period and may overlap if the test period is short
    relative to episode_days * n_windows (documented tradeoff for short
    series)."""
    max_start = len(test_idx) - episode_days - lead_time_L - 1
    if max_start < 0:
        raise ValueError(
            f"test split too short ({len(test_idx)} days) for episode_days={episode_days} "
            f"+ lead_time_L={lead_time_L}; use --quick or a shorter episode_days."
        )
    rng = np.random.default_rng(seed)
    if n_windows <= max_start + 1:
        starts = np.linspace(0, max_start, n_windows).round().astype(int)
        starts = sorted(set(starts.tolist()))
        while len(starts) < n_windows:
            starts.append(int(rng.integers(0, max_start + 1)))
    else:
        starts = sorted(rng.integers(0, max_start + 1, size=n_windows).tolist())
    return [int(s) for s in starts]


# ---------------------------------------------------------------------------
# Per-series setup: data, split, grid, demand model, baselines, planner.
# ---------------------------------------------------------------------------

class SeriesSetup:
    def __init__(self, series_id: str, cfg: dict, quick: bool = False):
        self.series_id = series_id
        inv_cfg_d = cfg["inventory"]
        self.df = load_series(series_id)
        split_cfg = inv_cfg_d["split"]
        self.train_idx, self.val_idx, self.test_idx = data_split(
            self.df, split_cfg["train_frac"], split_cfg["val_frac"], split_cfg["test_frac"]
        )
        train_df = self.df.iloc[self.train_idx]
        self.mean_price = float(train_df["sell_price"].mean())
        self.mean_demand = float(train_df["sales"].mean())

        episode_days = cfg["quick"]["episode_days"] if quick else inv_cfg_d["episode_days"]
        self.env_cfg = InventoryConfig(
            lead_time_L=inv_cfg_d["lead_time_L"],
            episode_days=episode_days,
            unit_cost_frac_of_price=inv_cfg_d["unit_cost_frac_of_price"],
            holding_cost_frac_of_unit_cost_per_day=inv_cfg_d["holding_cost_frac_of_unit_cost_per_day"],
            stockout_penalty_frac_of_price=inv_cfg_d["stockout_penalty_frac_of_price"],
            fixed_order_cost=inv_cfg_d["fixed_order_cost"],
            initial_inventory_days_of_mean_demand=inv_cfg_d["initial_inventory_days_of_mean_demand"],
            action_grid_days_of_mean_demand=tuple(inv_cfg_d["action_grid_days_of_mean_demand"]),
        )
        self.grid = make_action_grid(self.mean_demand, self.env_cfg)

    def train_env_factory(self):
        return InventoryEnv(self.df, cfg=self.env_cfg, day_indices=self.train_idx,
                             mean_price=self.mean_price, mean_demand=self.mean_demand)

    def test_env_factory(self):
        return InventoryEnv(self.df, cfg=self.env_cfg, day_indices=self.test_idx,
                             mean_price=self.mean_price, mean_demand=self.mean_demand)

    def build_baselines(self, n_tune_episodes: int, seed: int) -> Dict:
        train_df = self.df.iloc[self.train_idx]
        policies, tuned = build_all_baselines(
            self.train_env_factory, self.grid, self.mean_demand, self.mean_price, self.env_cfg,
            train_demand_samples=train_df["sales"].to_numpy(dtype=float),
            n_tune_episodes=n_tune_episodes, seed=seed,
        )
        return policies, tuned

    def build_planner(self, mcts_cfg: MCTSInventoryConfig, demand_model_kind: str) -> ReceedingHorizonPlanner:
        train_df = self.df.iloc[self.train_idx]
        demand_model = make_demand_model(demand_model_kind, train_df)
        return ReceedingHorizonPlanner(self.df, self.grid, self.env_cfg, mcts_cfg, demand_model, self.mean_price)


# ---------------------------------------------------------------------------
# Evaluation of one policy/planner over n_seeds test windows.
# ---------------------------------------------------------------------------

def evaluate_policy(setup: SeriesSetup, act_fn_factory: Callable[[], Callable[[InventoryState], int]],
                     starts: List[int], n_seeds: int) -> pd.DataFrame:
    rows = []
    for seed in range(n_seeds):
        env = setup.test_env_factory()
        act_fn = act_fn_factory()
        for w, start in enumerate(starts):
            metrics = run_episode(env, act_fn, start_idx=start)
            metrics.update({"seed": seed, "window": w})
            rows.append(metrics)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, policy_name: str) -> Dict:
    out = {"policy": policy_name}
    for metric in ["profit", "fill_rate", "stockout_rate", "avg_inventory", "n_orders"]:
        vals = df[metric].to_numpy(dtype=float)
        lo, hi = bootstrap_ci(vals)
        out[f"{metric}_mean"] = float(np.mean(vals))
        out[f"{metric}_std"] = float(np.std(vals))
        out[f"{metric}_ci_lo"] = lo
        out[f"{metric}_ci_hi"] = hi
    return out


def evaluate_series(series_id: str, cfg: dict, quick: bool = False, n_seeds: Optional[int] = None,
                     mcts_budget: Optional[int] = None) -> pd.DataFrame:
    setup = SeriesSetup(series_id, cfg, quick=quick)
    inv_cfg_d = cfg["inventory"]
    n_seeds = n_seeds or (cfg["quick"]["n_seeds"] if quick else inv_cfg_d["mcts"]["n_seeds"])
    n_tune_episodes = 5 if quick else 10
    n_windows = n_seeds  # one window per seed for the bootstrap (seeds reuse different windows too)

    starts = _test_window_starts(setup.test_idx, setup.env_cfg.episode_days,
                                  setup.env_cfg.lead_time_L, n_windows, seed=cfg["seed"])

    print(f"[{series_id}] mean_demand={setup.mean_demand:.2f} mean_price={setup.mean_price:.2f} "
          f"grid={setup.grid} train/val/test days={len(setup.train_idx)}/{len(setup.val_idx)}/{len(setup.test_idx)}")

    baselines, tuned = setup.build_baselines(n_tune_episodes=n_tune_episodes, seed=cfg["seed"])
    print(f"[{series_id}] tuned baseline params: {tuned}")

    rows = []
    for name, policy in baselines.items():
        df_runs = evaluate_policy(setup, lambda policy=policy: policy, starts, n_seeds=1)
        # baselines are deterministic given state -> one "seed" suffices,
        # but we still replicate n_seeds times for a like-for-like CI with MCTS
        # (demand realization is identical across replicas for a deterministic
        # policy, so we instead just evaluate once per window).
        rows.append(summarize(df_runs, name))

    mcts_cfg = MCTSInventoryConfig(
        horizon_H=inv_cfg_d["mcts"]["horizon_H"],
        c=inv_cfg_d["mcts"]["c"],
        n_sims=mcts_budget or (cfg["quick"]["budgets"][-1] if quick else inv_cfg_d["mcts"]["budgets"][-1]),
        rollout_policy=inv_cfg_d["mcts"]["rollout_policy"],
        rollout_base_stock_S=tuned.get("base_stock_S"),
        reward_scale=max(setup.mean_price * setup.mean_demand, 1.0),
        seed=cfg["seed"],
    )
    demand_model_kind = inv_cfg_d["mcts"]["demand_model"]

    t0 = time.time()
    mcts_rows = []
    for seed in range(n_seeds):
        env = setup.test_env_factory()
        planner = setup.build_planner(
            MCTSInventoryConfig(**{**mcts_cfg.__dict__, "seed": cfg["seed"] + seed}), demand_model_kind
        )
        start = starts[seed % len(starts)]
        metrics = run_episode(env, planner.act, start_idx=start)
        metrics.update({"seed": seed, "window": seed % len(starts)})
        mcts_rows.append(metrics)
    mcts_df = pd.DataFrame(mcts_rows)
    print(f"[{series_id}] MCTS: {n_seeds} test windows in {time.time()-t0:.1f}s "
          f"(~{(time.time()-t0)/(n_seeds*setup.env_cfg.episode_days)*1000:.1f} ms/decision, "
          f"n_sims={mcts_cfg.n_sims})")
    rows.append(summarize(mcts_df, "mcts"))

    result = pd.DataFrame(rows)
    result.insert(0, "series_id", series_id)
    return result


# ---------------------------------------------------------------------------
# Ablations (run on a single series by default -- full 5-series x
# multi-hyperparameter sweep is not CPU-laptop-feasible; see README).
# ---------------------------------------------------------------------------

def run_ablation(kind: str, cfg: dict, series_id: str, quick: bool) -> pd.DataFrame:
    setup = SeriesSetup(series_id, cfg, quick=quick)
    inv_cfg_d = cfg["inventory"]
    n_seeds = cfg["quick"]["n_seeds"] if quick else 3
    starts = _test_window_starts(setup.test_idx, setup.env_cfg.episode_days,
                                  setup.env_cfg.lead_time_L, n_seeds, seed=cfg["seed"])
    baselines, tuned = setup.build_baselines(n_tune_episodes=5, seed=cfg["seed"])
    base_S = tuned.get("base_stock_S")

    base_mcts_kwargs = dict(
        horizon_H=inv_cfg_d["mcts"]["horizon_H"], c=inv_cfg_d["mcts"]["c"],
        n_sims=(cfg["quick"]["budgets"][-1] if quick else inv_cfg_d["mcts"]["budgets"][-1]),
        rollout_policy=inv_cfg_d["mcts"]["rollout_policy"], rollout_base_stock_S=base_S,
        reward_scale=max(setup.mean_price * setup.mean_demand, 1.0),
    )
    demand_model_kind_default = inv_cfg_d["mcts"]["demand_model"]
    lead_time_default = setup.env_cfg.lead_time_L

    if kind == "budget":
        values = cfg["quick"]["budgets"] if quick else inv_cfg_d["mcts"]["budgets"]
        param_name, param_key = "n_sims", "n_sims"
    elif kind == "horizon":
        values = [7, 14, 21] if not quick else [7, 14]
        param_name, param_key = "horizon_H", "horizon_H"
    elif kind == "c":
        values = [0.5, 1.0, 1.41, 2.0]
        param_name, param_key = "c", "c"
    elif kind == "demand_model":
        values = ["bootstrap", "negbin", "deterministic_mean"]
        param_name, param_key = "demand_model", None
    elif kind == "rollout":
        values = ["base_stock", "random"]
        param_name, param_key = "rollout_policy", "rollout_policy"
    elif kind == "lead_time":
        values = [1, 2, 5]
        param_name, param_key = "lead_time_L", None
    else:
        raise ValueError(f"unknown ablation kind: {kind}")

    rows = []
    for v in values:
        env_cfg = setup.env_cfg
        demand_model_kind = demand_model_kind_default
        mcts_kwargs = dict(base_mcts_kwargs)
        if kind == "lead_time":
            env_cfg = InventoryConfig(**{**setup.env_cfg.__dict__, "lead_time_L": v})
            setup.grid = make_action_grid(setup.mean_demand, env_cfg)
        elif kind == "demand_model":
            demand_model_kind = v
        else:
            mcts_kwargs[param_key] = v

        t0 = time.time()
        mcts_rows = []
        for seed in range(n_seeds):
            local_env_cfg = env_cfg
            env = InventoryEnv(setup.df, cfg=local_env_cfg, day_indices=setup.test_idx,
                                mean_price=setup.mean_price, mean_demand=setup.mean_demand)
            mcts_cfg = MCTSInventoryConfig(seed=cfg["seed"] + seed, **mcts_kwargs)
            planner = setup.build_planner(mcts_cfg, demand_model_kind)
            start = starts[seed % len(starts)]
            metrics = run_episode(env, planner.act, start_idx=start)
            mcts_rows.append(metrics)
        mcts_df = pd.DataFrame(mcts_rows)
        seconds = time.time() - t0
        summary = summarize(mcts_df, f"mcts_{param_name}={v}")
        summary[param_name] = v
        summary["seconds_total"] = seconds
        summary["seconds_per_decision"] = seconds / (n_seeds * local_env_cfg.episode_days)
        rows.append(summary)
        print(f"  ablation[{kind}] {param_name}={v}: profit={summary['profit_mean']:.2f} "
              f"fill={summary['fill_rate_mean']:.3f} ({seconds:.1f}s)")

    result = pd.DataFrame(rows)
    result.insert(0, "series_id", series_id)
    result.insert(1, "ablation", kind)
    return result


def plot_policy_comparison(results: pd.DataFrame, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series_list = results["series_id"].unique()
    fig, axes = plt.subplots(1, len(series_list), figsize=(5 * len(series_list), 4), squeeze=False)
    for i, sid in enumerate(series_list):
        ax = axes[0][i]
        sub = results[results["series_id"] == sid]
        ax.bar(sub["policy"], sub["profit_mean"],
               yerr=[sub["profit_mean"] - sub["profit_ci_lo"], sub["profit_ci_hi"] - sub["profit_mean"]],
               capsize=4)
        ax.set_title(sid, fontsize=9)
        ax.set_ylabel("mean test profit")
        ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_ablation(results: pd.DataFrame, kind: str, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    param_col = [c for c in results.columns if c not in (
        "series_id", "ablation", "policy", "seconds_total", "seconds_per_decision"
    ) and not c.startswith(("profit", "fill_rate", "stockout_rate", "avg_inventory", "n_orders"))]
    param_name = param_col[0] if param_col else "value"
    fig, ax = plt.subplots(figsize=(5, 4))
    x = results[param_name].astype(str)
    ax.errorbar(x, results["profit_mean"],
                yerr=[results["profit_mean"] - results["profit_ci_lo"], results["profit_ci_hi"] - results["profit_mean"]],
                marker="o", capsize=4)
    ax.set_xlabel(param_name)
    ax.set_ylabel("mean test profit")
    ax.set_title(f"Ablation: {kind}")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="fast smoke run: fewer series/seeds/sims")
    parser.add_argument("--series", nargs="*", default=None, help="subset of series ids to run")
    parser.add_argument("--n-seeds", type=int, default=None)
    parser.add_argument("--mcts-budget", type=int, default=None)
    parser.add_argument("--ablation", choices=["budget", "horizon", "c", "demand_model", "rollout", "lead_time"],
                         default=None, help="run one ablation sweep (on the first/--series series) instead of the main comparison")
    args = parser.parse_args()

    cfg = load_config()
    np.random.seed(cfg["seed"])

    series_ids = args.series or (cfg["inventory"]["series_ids"][:2] if args.quick else cfg["inventory"]["series_ids"])

    OUT_RESULTS.mkdir(parents=True, exist_ok=True)
    OUT_PLOTS.mkdir(parents=True, exist_ok=True)

    if args.ablation:
        sid = series_ids[0]
        print(f"Running ablation '{args.ablation}' on {sid} (quick={args.quick})")
        result = run_ablation(args.ablation, cfg, sid, quick=args.quick)
        result.to_csv(OUT_RESULTS / f"ablation_{args.ablation}.csv", index=False)
        plot_ablation(result, args.ablation, OUT_PLOTS / f"ablation_{args.ablation}.png")
        print(result)
        return

    all_results = []
    for sid in series_ids:
        print(f"\n=== Evaluating {sid} ===")
        res = evaluate_series(sid, cfg, quick=args.quick, n_seeds=args.n_seeds, mcts_budget=args.mcts_budget)
        all_results.append(res)
        print(res[["policy", "profit_mean", "profit_std", "fill_rate_mean", "stockout_rate_mean",
                    "avg_inventory_mean", "n_orders_mean"]].to_string(index=False))

    combined = pd.concat(all_results, ignore_index=True)
    out_csv = OUT_RESULTS / ("policy_comparison_quick.csv" if args.quick else "policy_comparison.csv")
    combined.to_csv(out_csv, index=False)
    print(f"\nSaved results to {out_csv}")

    plot_policy_comparison(combined, OUT_PLOTS / ("policy_comparison_quick.png" if args.quick else "policy_comparison.png"))
    print(f"Saved plot to {OUT_PLOTS}")


if __name__ == "__main__":
    main()
