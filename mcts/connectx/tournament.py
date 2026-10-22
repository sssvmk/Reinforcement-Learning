"""Round-robin ConnectX tournament harness: plays agents against each other
in both seats, computes win rates with Wilson 95% confidence intervals, and
produces the ablation plots requested in the task (win rate vs. simulation
budget, vs. exploration constant c, random vs. heuristic rollout, and
seconds-per-move vs. budget).

Does NOT depend on kaggle_environments -- this is "your own harness" as the
task requires. If kaggle_environments is importable, `cross_check_with_kaggle_runtime`
additionally plays a few games through the real runtime for sanity.
"""
from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from connectx_state import ConnectXState
from agents import (
    random_policy, greedy_policy, make_negamax, make_mcts_agent,
)


def wilson_ci(wins: float, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    """Wilson score interval for a (possibly fractional, due to draws=0.5)
    win count out of n games. Returns (point_estimate, low, high)."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = wins / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def play_game(policy1: Callable, policy2: Callable, rows=6, columns=7, inarow=4,
              seed: int = 0) -> Tuple[float, float, float]:
    """Plays one game, policy1 as mark 1, policy2 as mark 2. Returns
    (score1, score2, seconds_per_move_avg) with score in {0, 0.5, 1}."""
    s = ConnectXState(rows=rows, columns=columns, inarow=inarow, mark=1)
    rng1 = random.Random(seed * 2)
    rng2 = random.Random(seed * 2 + 1)
    move_times = []
    while not s.is_terminal():
        policy, rng = (policy1, rng1) if s.current_player() == 1 else (policy2, rng2)
        t0 = time.time()
        a = policy(s, rng)
        move_times.append(time.time() - t0)
        s = s.apply(a)
    winner = s.winner()
    if winner is None:
        score1, score2 = 0.5, 0.5
    elif winner == 1:
        score1, score2 = 1.0, 0.0
    else:
        score1, score2 = 0.0, 1.0
    avg_time = sum(move_times) / len(move_times) if move_times else 0.0
    return score1, score2, avg_time


@dataclass
class PairingResult:
    name_a: str
    name_b: str
    games: int
    score_a: float
    win_rate_a: float
    ci_low: float
    ci_high: float
    avg_seconds_per_move: float


def round_robin(agents: Dict[str, Callable], n_games_per_pairing: int = 200,
                 seed: int = 0) -> List[PairingResult]:
    """Plays every pairing in BOTH seats, `n_games_per_pairing` total games
    split evenly across seats."""
    names = list(agents.keys())
    results = []
    half = max(1, n_games_per_pairing // 2)
    for i in range(len(names)):
        for j in range(len(names)):
            if i == j:
                continue
            name_a, name_b = names[i], names[j]
            pa, pb = agents[name_a], agents[name_b]
            total_score_a = 0.0
            n = 0
            times = []
            for g in range(half):
                s1, s2, t = play_game(pa, pb, seed=seed * 10_000 + g)
                total_score_a += s1
                n += 1
                times.append(t)
            for g in range(half):
                s2b, s1b, t = play_game(pb, pa, seed=seed * 10_000 + 5000 + g)
                total_score_a += s1b
                n += 1
                times.append(t)
            p, lo, hi = wilson_ci(total_score_a, n)
            results.append(PairingResult(
                name_a=name_a, name_b=name_b, games=n, score_a=total_score_a,
                win_rate_a=p, ci_low=lo, ci_high=hi,
                avg_seconds_per_move=sum(times) / len(times) if times else 0.0,
            ))
    return results


def build_standard_agents(negamax_depths=(2, 4, 6), mcts_budgets=(50, 200, 800, 3200),
                           c=1.41, rollout="heuristic", seed=0) -> Dict[str, Callable]:
    agents: Dict[str, Callable] = {
        "random": random_policy,
        "greedy": greedy_policy,
    }
    for d in negamax_depths:
        agents[f"negamax_d{d}"] = make_negamax(d)
    for b in mcts_budgets:
        agents[f"mcts_{b}"] = make_mcts_agent(n_sims=b, c=c, rollout=rollout, seed=seed)
    return agents


def cross_check_with_kaggle_runtime(our_agent_name: str, our_agent, n_games: int = 20, seed: int = 0):
    """Only runs if kaggle_environments is importable. Plays our wrapped
    agent against the package's own built-in 'random' and 'negamax' agents
    through the REAL runtime (env.run), both seats, confirming the wrapped
    agent behaves sanely outside our own harness."""
    try:
        from kaggle_environments import make
    except ImportError:
        print("kaggle_environments not importable -- skipping runtime cross-check.")
        return None
    config = {"rows": 6, "columns": 7, "inarow": 4}
    rows_out = []
    for opp_name in ("random", "negamax"):
        wins = losses = draws = 0
        for g in range(n_games):
            env = make("connectx", configuration=config, debug=False)
            if g % 2 == 0:
                res = env.run([our_agent, opp_name])
                my_reward = res[-1][0]["reward"]
            else:
                res = env.run([opp_name, our_agent])
                my_reward = res[-1][1]["reward"]
            if my_reward == 1:
                wins += 1
            elif my_reward == -1 or my_reward is None:
                losses += 1
            else:
                draws += 1
        n = wins + losses + draws
        score = wins + 0.5 * draws
        p, lo, hi = wilson_ci(score, n)
        rows_out.append((our_agent_name, opp_name, n, wins, losses, draws, p, lo, hi))
        print(f"[kaggle runtime] {our_agent_name} vs {opp_name}: "
              f"{wins}W/{losses}L/{draws}D over {n} games, win_rate={p:.3f} "
              f"[{lo:.3f},{hi:.3f}]")
    return rows_out


def win_rate_vs_fixed(agent_fn: Callable, opponent: Callable, n_games: int, seed: int) -> PairingResult:
    """Plays agent_fn (as both seats) against a fixed opponent and returns
    a PairingResult from agent_fn's perspective."""
    half = max(1, n_games // 2)
    total_score = 0.0
    times = []
    for g in range(half):
        s1, s2, t = play_game(agent_fn, opponent, seed=seed * 10_000 + g)
        total_score += s1
        times.append(t)
    for g in range(half):
        s2, s1, t = play_game(opponent, agent_fn, seed=seed * 10_000 + 5000 + g)
        total_score += s1
        times.append(t)
    n = 2 * half
    p, lo, hi = wilson_ci(total_score, n)
    return PairingResult("agent", "opponent", n, total_score, p, lo, hi,
                          sum(times) / len(times) if times else 0.0)


def ablation_budget(opponent_name: str, opponent: Callable, budgets: List[int], c: float,
                     rollout: str, n_games: int, seed: int = 0) -> List[dict]:
    rows = []
    for b in budgets:
        agent = make_mcts_agent(n_sims=b, c=c, rollout=rollout, seed=seed)
        r = win_rate_vs_fixed(agent, opponent, n_games, seed)
        rows.append({"budget": b, "opponent": opponent_name, "win_rate": r.win_rate_a,
                     "ci_low": r.ci_low, "ci_high": r.ci_high, "seconds_per_move": r.avg_seconds_per_move})
        print(f"  budget={b} vs {opponent_name}: win_rate={r.win_rate_a:.3f} "
              f"[{r.ci_low:.3f},{r.ci_high:.3f}] ({r.avg_seconds_per_move*1000:.1f} ms/move)")
    return rows


def ablation_c(opponent_name: str, opponent: Callable, budget: int, c_values: List[float],
               rollout: str, n_games: int, seed: int = 0) -> List[dict]:
    rows = []
    for c in c_values:
        agent = make_mcts_agent(n_sims=budget, c=c, rollout=rollout, seed=seed)
        r = win_rate_vs_fixed(agent, opponent, n_games, seed)
        rows.append({"c": c, "opponent": opponent_name, "win_rate": r.win_rate_a,
                     "ci_low": r.ci_low, "ci_high": r.ci_high})
        print(f"  c={c} vs {opponent_name}: win_rate={r.win_rate_a:.3f} [{r.ci_low:.3f},{r.ci_high:.3f}]")
    return rows


def ablation_rollout(opponent_name: str, opponent: Callable, budget: int, c: float,
                      n_games: int, seed: int = 0) -> List[dict]:
    rows = []
    for rollout in ("random", "heuristic"):
        agent = make_mcts_agent(n_sims=budget, c=c, rollout=rollout, seed=seed)
        r = win_rate_vs_fixed(agent, opponent, n_games, seed)
        rows.append({"rollout": rollout, "opponent": opponent_name, "win_rate": r.win_rate_a,
                     "ci_low": r.ci_low, "ci_high": r.ci_high})
        print(f"  rollout={rollout} vs {opponent_name}: win_rate={r.win_rate_a:.3f} "
              f"[{r.ci_low:.3f},{r.ci_high:.3f}]")
    return rows


def ablation_robust_child(opponent_name: str, opponent: Callable, budget: int, c: float,
                           rollout: str, n_games: int, seed: int = 0) -> List[dict]:
    rows = []
    for robust_child in (True, False):
        agent = make_mcts_agent(n_sims=budget, c=c, rollout=rollout, robust_child=robust_child, seed=seed)
        r = win_rate_vs_fixed(agent, opponent, n_games, seed)
        label = "robust_child" if robust_child else "max_value_child"
        rows.append({"selection_rule": label, "opponent": opponent_name, "win_rate": r.win_rate_a,
                     "ci_low": r.ci_low, "ci_high": r.ci_high})
        print(f"  selection_rule={label} vs {opponent_name}: win_rate={r.win_rate_a:.3f} "
              f"[{r.ci_low:.3f},{r.ci_high:.3f}]")
    return rows


def _plot_series(rows: List[dict], x_key: str, out_path: Path, title: str, xlabel: str,
                  group_key: Optional[str] = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 4))
    if group_key:
        groups = sorted(set(r[group_key] for r in rows), key=str)
        for g in groups:
            sub = [r for r in rows if r[group_key] == g]
            xs = [r[x_key] for r in sub]
            ys = [r["win_rate"] for r in sub]
            yerr_lo = [r["win_rate"] - r["ci_low"] for r in sub]
            yerr_hi = [r["ci_high"] - r["win_rate"] for r in sub]
            ax.errorbar(xs, ys, yerr=[yerr_lo, yerr_hi], marker="o", capsize=4, label=str(g))
        ax.legend(fontsize=8)
    else:
        xs = [r[x_key] for r in rows]
        ys = [r["win_rate"] for r in rows]
        yerr_lo = [r["win_rate"] - r["ci_low"] for r in rows]
        yerr_hi = [r["ci_high"] - r["win_rate"] for r in rows]
        ax.errorbar(xs, ys, yerr=[yerr_lo, yerr_hi], marker="o", capsize=4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("win rate (draws=0.5)")
    ax.set_title(title)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def run_all_ablations(out_dir: Path, quick: bool = False, seed: int = 0):
    import csv
    opponent_name, opponent = "negamax_d4", make_negamax(4)
    budgets = (20, 50) if quick else (50, 200, 800, 3200)
    c_values = (0.5, 1.41) if quick else (0.5, 1.0, 1.41, 2.0)
    n_games = 20 if quick else 100
    fixed_budget = budgets[1] if len(budgets) > 1 else budgets[0]

    print(f"\n--- Ablation: win rate vs. simulation budget (opponent={opponent_name}) ---")
    rows_budget = ablation_budget(opponent_name, opponent, list(budgets), c=1.41, rollout="heuristic",
                                   n_games=n_games, seed=seed)
    _plot_series(rows_budget, "budget", out_dir.parent / "plots" / "ablation_connectx_budget.png",
                 "MCTS win rate vs. simulation budget", "simulations/move")
    _plot_series(
        [{"budget": r["budget"], "win_rate": r["seconds_per_move"], "ci_low": r["seconds_per_move"],
          "ci_high": r["seconds_per_move"]} for r in rows_budget],
        "budget", out_dir.parent / "plots" / "ablation_connectx_seconds_per_move.png",
        "Seconds per move vs. simulation budget", "simulations/move",
    )

    print(f"\n--- Ablation: win rate vs. exploration constant c (opponent={opponent_name}) ---")
    rows_c = ablation_c(opponent_name, opponent, budget=fixed_budget, c_values=list(c_values),
                         rollout="heuristic", n_games=n_games, seed=seed)
    _plot_series(rows_c, "c", out_dir.parent / "plots" / "ablation_connectx_c.png",
                 "MCTS win rate vs. exploration constant c", "c")

    print(f"\n--- Ablation: random vs. heuristic rollout (opponent={opponent_name}) ---")
    rows_rollout = ablation_rollout(opponent_name, opponent, budget=fixed_budget, c=1.41,
                                     n_games=n_games, seed=seed)
    print(rows_rollout)

    print(f"\n--- Ablation: robust-child vs. max-value-child selection (opponent={opponent_name}) ---")
    rows_robust = ablation_robust_child(opponent_name, opponent, budget=fixed_budget, c=1.41,
                                         rollout="heuristic", n_games=n_games, seed=seed)
    print(rows_robust)

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in [("budget", rows_budget), ("c", rows_c), ("rollout", rows_rollout),
                        ("robust_child", rows_robust)]:
        out_csv = out_dir / f"connectx_ablation_{name}.csv"
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {out_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--games-per-pairing", type=int, default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--kaggle-cross-check", action="store_true",
                         help="Also play a few real matches through kaggle_environments, if importable.")
    parser.add_argument("--ablations", action="store_true",
                         help="Run win-rate/seconds-per-move ablation sweeps and plots instead of the full round robin.")
    parser.add_argument("--depths", type=str, default=None,
                         help="Comma-separated negamax depths to include, e.g. '2,4'.")
    parser.add_argument("--budgets", type=str, default=None,
                         help="Comma-separated MCTS simulation budgets to include, e.g. '50,200,800'.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    out_dir = Path(args.out) if args.out else root / "outputs" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.ablations:
        run_all_ablations(out_dir, quick=args.quick, seed=0)
        return

    if args.quick:
        depths, budgets, n_games = (2, 4), (50, 200), 20
    else:
        depths, budgets, n_games = (2, 4, 6), (50, 200, 800, 3200), 200
    if args.games_per_pairing:
        n_games = args.games_per_pairing
    if args.depths:
        depths = tuple(int(x) for x in args.depths.split(","))
    if args.budgets:
        budgets = tuple(int(x) for x in args.budgets.split(","))

    agents = build_standard_agents(negamax_depths=depths, mcts_budgets=budgets)
    print(f"Running round robin with {len(agents)} agents, {n_games} games/pairing...")
    results = round_robin(agents, n_games_per_pairing=n_games, seed=0)

    import csv
    out_csv = out_dir / "connectx_tournament.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["agent_a", "agent_b", "games", "win_rate_a", "ci_low", "ci_high", "avg_sec_per_move"])
        for r in results:
            writer.writerow([r.name_a, r.name_b, r.games, f"{r.win_rate_a:.4f}", f"{r.ci_low:.4f}",
                              f"{r.ci_high:.4f}", f"{r.avg_seconds_per_move:.5f}"])
    print(f"Wrote {out_csv}")

    if args.kaggle_cross_check:
        from agents import mcts_agent
        cc_rows = cross_check_with_kaggle_runtime("mcts_200", mcts_agent(n_sims=200, seed=0), n_games=20)
        if cc_rows:
            out_cc = out_dir / "connectx_kaggle_cross_check.csv"
            with open(out_cc, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["agent", "opponent", "n", "wins", "losses", "draws", "win_rate", "ci_low", "ci_high"])
                for r in cc_rows:
                    writer.writerow(r)
            print(f"Wrote {out_cc}")


if __name__ == "__main__":
    main()
