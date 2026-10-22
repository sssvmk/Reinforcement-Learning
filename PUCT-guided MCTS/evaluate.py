"""Tournament harness, benchmarks, ablations and plots.

    python evaluate.py --quick                 # small smoke benchmark
    python evaluate.py --full                  # main benchmark (config eval.games_per_pairing games per pairing)
    python evaluate.py --ablations [--quick]   # trains small variants (outputs/runs/<tag>) and compares them
    python evaluate.py --plots                 # regenerate plots from saved results

All games are played from random 4-ply openings (config eval.opening_plies); each opening is played in BOTH seats.
Score = (wins + 0.5*draws)/games from the first-named agent's point of view, with Wilson 95% intervals.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from az_mcts import AZPlayer
from baselines import GreedyAgent, MCTSAgent, NegamaxAgent, RandomAgent
from config import ROOT, load_config, output_dir
from game import ReferenceGame, State, random_opening

_EV_CACHE: dict = {}
_SUB_CACHE: dict = {}


@dataclass(frozen=True)
class AgentSpec:
    type: str                         # random | greedy | negamax | mcts | az | submission | hybrid
    depth: int = 4
    time_limit: float | None = None
    sims: int | None = None
    c: float = 1.4                    # UCT exploration constant
    rollout: str = "random"
    weights: str | None = None
    c_puct: float = 1.5
    reuse: bool = False
    path: str | None = None           # submission.py path

    def label(self) -> str:
        t = self.type
        if t in ("random", "greedy"):
            return t
        if t == "negamax":
            return f"negamax-d{self.depth}" + (f"-t{self.time_limit}" if self.depth >= 20 else "")
        if t == "mcts":
            b = f"t{self.time_limit:.3f}s" if self.time_limit and not self.sims else f"{self.sims}sims"
            return f"mcts-{self.rollout[:4]}-{b}"
        if t == "az":
            tag = "policy-only" if self.sims == 0 else f"az-{self.sims}sims"
            return tag + (f"-c{self.c_puct}" if self.c_puct != 1.5 else "")
        if t == "submission":
            return "submission-np"
        if t == "hybrid":
            return f"hybrid-t{self.time_limit}"
        return t


def _evaluator(path: str):
    from submission.np_net import NumpyEvaluator
    key = (path, Path(path).stat().st_mtime_ns)
    if key not in _EV_CACHE:
        _EV_CACHE[key] = NumpyEvaluator.from_file(path)
    return _EV_CACHE[key]


class SubmissionAgent:
    """Calls the generated submission.py through a Kaggle-style (observation, configuration) interface."""

    def __init__(self, path, time_limit, rows, cols, inarow):
        key = str(path)
        if key not in _SUB_CACHE:
            sp = importlib.util.spec_from_file_location(f"submission_{len(_SUB_CACHE)}", key)
            mod = importlib.util.module_from_spec(sp)
            sp.loader.exec_module(mod)
            _SUB_CACHE[key] = mod
        self.mod = _SUB_CACHE[key]
        lim = time_limit or 2
        self.cfg = {"rows": rows, "columns": cols, "inarow": inarow, "timeout": lim, "actTimeout": lim}
        self.name = "submission-np"

    def select(self, state: State) -> int:
        obs = {"board": state.to_board(), "mark": state.current_player, "step": state.moves}
        return int(self.mod.agent(obs, dict(self.cfg)))


def make_agent(spec: AgentSpec, seed: int, game=(6, 7, 4)):
    t = spec.type
    if t == "random":
        return RandomAgent(seed)
    if t == "greedy":
        return GreedyAgent(seed)
    if t == "negamax":
        return NegamaxAgent(spec.depth, spec.time_limit, seed)
    if t == "mcts":
        return MCTSAgent(spec.sims, spec.time_limit, spec.c, spec.rollout, seed)
    if t == "az":
        return AZPlayer(_evaluator(spec.weights), sims=spec.sims, time_limit=spec.time_limit, c_puct=spec.c_puct,
                        reuse=spec.reuse, seed=seed)
    if t == "submission":
        return SubmissionAgent(spec.path, spec.time_limit, *game)
    if t == "hybrid":
        from hybrid import HybridAgent
        return HybridAgent(_evaluator(spec.weights), depth=spec.depth, time_limit=spec.time_limit, seed=seed)
    raise ValueError(f"unknown agent type {t}")


def wilson(score_sum: float, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0, 1.0
    p = score_sum / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, centre - half), min(1.0, centre + half)


def _play_game(task):
    sa, sb, opening, a_first, seed, game = task
    A, B = make_agent(sa, seed * 2 + 1, game), make_agent(sb, seed * 2 + 2, game)
    s = State(*game)
    for m in opening:
        s.play(m)
    agents = (A, B) if a_first else (B, A)
    tm = [0.0, 0.0]
    nm = [0, 0]
    sims0 = getattr(A, "total_sims", 0)
    illegal = None
    while not s.is_terminal():
        i = s.moves % 2
        t0 = time.perf_counter()
        col = agents[i].select(s)
        who = i if a_first else 1 - i      # 0 -> A, 1 -> B
        tm[who] += time.perf_counter() - t0
        nm[who] += 1
        if not s.is_legal(col):
            illegal = who
            break
        s.play(col)
    if illegal is not None:
        a_score = 0.0 if illegal == 0 else 1.0
    elif s.won:
        winner = 3 - s.current_player
        a_score = 1.0 if (winner == 1) == a_first else 0.0
    else:
        a_score = 0.5
    return {"a_score": a_score, "t_a": tm[0], "n_a": nm[0], "t_b": tm[1], "n_b": nm[1], "illegal": illegal,
            "sims_a": getattr(A, "total_sims", 0) - sims0, "plies": s.moves}


_POOL = {}


def get_pool(workers: int):
    if workers not in _POOL:
        _POOL[workers] = ProcessPoolExecutor(max_workers=workers)
    return _POOL[workers]


def play_match(sa: AgentSpec, sb: AgentSpec, n_games: int, seed: int, cfg: dict, workers: int = 1) -> dict:
    g = cfg["game"]
    game = (g["rows"], g["columns"], g["inarow"])
    rng = np.random.default_rng(seed)
    tasks = []
    for i in range((n_games + 1) // 2):
        o = random_opening(rng, cfg["eval"]["opening_plies"], *game)
        for first in (True, False):
            tasks.append((sa, sb, o, first, seed * 100003 + len(tasks), game))
    t0 = time.perf_counter()
    if workers > 1:
        res = list(get_pool(workers).map(_play_game, tasks, chunksize=1))
    else:
        res = [_play_game(t) for t in tasks]
    n = len(res)
    score = sum(r["a_score"] for r in res)
    w = sum(r["a_score"] == 1 for r in res)
    d = sum(r["a_score"] == 0.5 for r in res)
    p, lo, hi = wilson(score, n)
    na, nb = sum(r["n_a"] for r in res), sum(r["n_b"] for r in res)
    return {"a": sa.label(), "b": sb.label(), "a_spec": asdict(sa), "b_spec": asdict(sb), "n": n, "wins": int(w),
            "draws": int(d), "losses": int(n - w - d), "score": p, "lo": lo, "hi": hi,
            "sec_per_move_a": sum(r["t_a"] for r in res) / max(na, 1),
            "sec_per_move_b": sum(r["t_b"] for r in res) / max(nb, 1),
            "sims_per_move_a": sum(r["sims_a"] for r in res) / max(na, 1),
            "illegal_a": sum(r["illegal"] == 0 for r in res), "illegal_b": sum(r["illegal"] == 1 for r in res),
            "wall_seconds": time.perf_counter() - t0}


# --------------------------------------------------------------------------------------------------
# Kaggle-style harness (independent rules via ReferenceGame; agents get dict observation/configuration)
# --------------------------------------------------------------------------------------------------
def play_kaggle_style(agent1, agent2, configuration=None, opening=(), timeout=None):
    """Plays one game through the public `agent(observation, configuration)` interface only.
    Returns (winner 0/1/2, info). Illegal move or exceeding `timeout` seconds forfeits."""
    cfgd = dict(configuration or {"rows": 6, "columns": 7, "inarow": 4, "timeout": 2, "actTimeout": 2})
    ref = ReferenceGame(cfgd["rows"], cfgd["columns"], cfgd["inarow"])
    for m in opening:
        ref.play(m)
    agents = {1: agent1, 2: agent2}
    max_t = {1: 0.0, 2: 0.0}
    limit = timeout if timeout is not None else cfgd.get("actTimeout", cfgd.get("timeout", 2))
    while not ref.winner and not ref.is_draw():
        p = ref.player
        obs = {"board": ref.flat_board(), "mark": p, "step": sum(1 for x in ref.flat_board() if x)}
        t0 = time.perf_counter()
        col = agents[p](obs, dict(cfgd))
        dt = time.perf_counter() - t0
        max_t[p] = max(max_t[p], dt)
        if not isinstance(col, (int, np.integer)) or col not in ref.legal_moves():
            return 3 - p, {"forfeit": p, "reason": f"illegal move {col!r}", "max_time": max_t}
        if dt > limit:
            return 3 - p, {"forfeit": p, "reason": f"timeout {dt:.2f}s > {limit}s", "max_time": max_t}
        ref.play(int(col))
    return ref.winner, {"max_time": max_t}


# --------------------------------------------------------------------------------------------------
# Benchmark drivers
# --------------------------------------------------------------------------------------------------
class ResultStore:
    def __init__(self, path: Path):
        self.path = path
        self.data = json.loads(path.read_text()) if path.exists() else {}

    def run(self, sa, sb, n, seed, cfg, workers, tag=""):
        key = f"{tag}|{sa.label()} vs {sb.label()}|n={n}"
        if key in self.data:
            return self.data[key]
        r = play_match(sa, sb, n, seed, cfg, workers)
        r["tag"] = tag
        self.data[key] = r
        self.path.write_text(json.dumps(self.data, indent=1))
        print(f"  {sa.label():>22} vs {sb.label():<22} n={r['n']:>3} score={r['score']:.3f} "
              f"[{r['lo']:.2f},{r['hi']:.2f}] W{r['wins']}/D{r['draws']}/L{r['losses']} "
              f"{r['sec_per_move_a']*1000:.0f}ms/move  ({r['wall_seconds']:.0f}s)", flush=True)
        return r


def run_main_benchmark(cfg, weights: Path, quick: bool, workers: int, sub_path: Path | None):
    e = cfg["eval"]
    n = e["quick_games"] if quick else e["games_per_pairing"]
    n_sub = max(10, n // 5)
    res_dir = output_dir(cfg, "results")
    store = ResultStore(res_dir / ("eval_quick.json" if quick else "eval_results.json"))
    w = str(weights)
    sims_list = e["az_sims"] if not quick else e["az_sims"][:2]
    depths = e["negamax_depths"] if not quick else e["negamax_depths"][:2]
    az = [AgentSpec("az", sims=0, weights=w)] + [AgentSpec("az", sims=s, weights=w, c_puct=cfg["mcts"]["c_puct"])
                                                  for s in sims_list]
    nm = [AgentSpec("negamax", depth=d, time_limit=e["negamax_time_limit"]) for d in depths]
    if not quick:
        nm.append(AgentSpec("negamax", depth=8, time_limit=e["negamax_time_limit"]))
    seed = cfg["seed"] + 1000
    for sa in az:
        print(f"== {sa.label()}")
        for sb in [AgentSpec("random"), AgentSpec("greedy")] + nm:
            store.run(sa, sb, n, seed, cfg, workers)
        if sa.sims:
            for roll in ("random", "heuristic"):
                eq = store.run(sa, AgentSpec("mcts", sims=sa.sims, c=e["mcts_c"], rollout=roll), n, seed, cfg, workers)
                # equal wall-clock: give MCTS the AZ agent's measured seconds per move
                t = max(eq["sec_per_move_a"], 0.005)
                store.run(sa, AgentSpec("mcts", sims=None, time_limit=round(t, 3), c=e["mcts_c"], rollout=roll),
                          n, seed, cfg, workers)
    if sub_path and Path(sub_path).exists():
        sub = AgentSpec("submission", path=str(sub_path))
        print("== submission (NumPy agent, real time budget)")
        for sb in [AgentSpec("random"), AgentSpec("greedy")] + nm[:3]:
            store.run(sub, sb, n_sub, seed, cfg, min(workers, 8))
    write_tables(cfg, store, quick)
    make_eval_plots(cfg, store, quick)
    return store


def write_tables(cfg, store, quick):
    rows = {}
    opps = []
    for r in store.data.values():
        if r.get("tag"):
            continue
        rows.setdefault(r["a"], {})[r["b"]] = r
        if r["b"] not in opps:
            opps.append(r["b"])
    lines = ["| AZ agent \\ opponent | " + " | ".join(opps) + " |", "|---|" + "---|" * len(opps)]
    for a, d in rows.items():
        cells = []
        for o in opps:
            r = d.get(o)
            cells.append("-" if r is None else f"{r['score']:.2f} [{r['lo']:.2f}-{r['hi']:.2f}] n={r['n']}")
        lines.append(f"| {a} | " + " | ".join(cells) + " |")
    lines += ["", "Seconds per move / simulations per move of each AZ agent (mean over its games):", "",
              "| agent | ms/move | sims/move |", "|---|---|---|"]
    for a, d in rows.items():
        rs = list(d.values())
        lines.append(f"| {a} | {np.mean([r['sec_per_move_a'] for r in rs])*1000:.1f} | "
                     f"{np.mean([r['sims_per_move_a'] for r in rs]):.0f} |")
    out = output_dir(cfg, "results") / ("eval_table_quick.md" if quick else "eval_table.md")
    out.write_text("\n".join(lines))
    print(f"table -> {out}")


def make_eval_plots(cfg, store, quick):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pd = output_dir(cfg, "plots")
    budgets = {}
    for r in store.data.values():
        if r.get("tag") or not r["a"].startswith(("az-", "policy")):
            continue
        sims = 0 if r["a"].startswith("policy") else int(r["a"].split("-")[1].replace("sims", "").split("-c")[0])
        budgets.setdefault(r["b"], {})[sims] = r
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for opp, d in budgets.items():
        if opp.startswith("mcts") and "t0" in opp:
            continue
        xs = sorted(d)
        ax.plot([max(x, 1) for x in xs], [d[x]["score"] for x in xs], "o-", label=opp)
    ax.set_xscale("log"); ax.set_xlabel("AZ simulations per move (policy-only plotted at 1)")
    ax.set_ylabel("score vs opponent (draw=0.5)"); ax.set_ylim(0, 1.02); ax.legend(fontsize=6, ncol=2)
    ax.set_title("Win rate vs simulation budget")
    fig.tight_layout(); fig.savefig(pd / "winrate_vs_budget.png", dpi=120); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    secs = {}
    for d in budgets.values():
        for s, r in d.items():
            secs.setdefault(s, []).append(r["sec_per_move_a"])
    xs = sorted(secs)
    ax.plot([max(x, 1) for x in xs], [np.mean(secs[x]) for x in xs], "o-")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("simulations per move (policy-only at 1)")
    ax.set_ylabel("seconds per move"); ax.set_title("Compute per move (one CPU process)")
    fig.tight_layout(); fig.savefig(pd / "seconds_per_move.png", dpi=120); plt.close(fig)


def run_ablations(cfg0, quick: bool, workers: int):
    """Trains small variants with the `medium` preset (quick: `quick`), then evaluates them identically."""
    import train_az
    preset = "quick" if quick else "medium"
    variants = {
        "base": {},
        "no_mirror": {"train.mirror_augment": False},
        "no_gating": {"train.gating": False},
        "mlp": {"network.type": "mlp"},
    }
    res_dir = output_dir(cfg0, "results")
    store = ResultStore(res_dir / f"ablations_{preset}.json")
    n = 20 if quick else 60
    opps = [AgentSpec("random"), AgentSpec("greedy"), AgentSpec("negamax", depth=2), AgentSpec("negamax", depth=4)]
    weights = {}
    for name, ov in variants.items():
        cfg = load_config(preset)
        for k, v in ov.items():
            s, key = k.split(".")
            cfg[s][key] = v
        run_dir = output_dir(cfg, "runs", f"abl_{preset}_{name}")
        if not (run_dir / "best_weights.npz").exists():
            print(f"== training ablation variant {name} ({preset})")
            train_az.train(cfg, run_dir)
        weights[name] = run_dir / "best_weights.npz"
    print("== evaluating ablations")
    for name, wp in weights.items():
        for sims in (0, 50):
            sa = AgentSpec("az", sims=sims, weights=str(wp))
            for sb in opps:
                store.run(sa, sb, n, cfg0["seed"] + 77, cfg0, workers, tag=f"abl:{name}")
    # c_puct sweep on the base net
    for c in (1.0, 1.5, 2.5):
        sa = AgentSpec("az", sims=200 if not quick else 50, weights=str(weights["base"]), c_puct=c)
        for sb in opps[2:]:
            store.run(sa, sb, n, cfg0["seed"] + 78, cfg0, workers, tag="abl:cpuct")
    lines = [f"Ablation runs: preset `{preset}`, one seed, {n} games per cell (Wilson 95% CI in brackets).", "",
             "| variant | az agent | " + " | ".join(o.label() for o in opps) + " |", "|---|---|" + "---|" * len(opps)]
    for r_tag in [f"abl:{n_}" for n_ in variants] + ["abl:cpuct"]:
        agents = []
        for r in store.data.values():
            if r.get("tag") == r_tag and r["a"] not in agents:
                agents.append(r["a"])
        for a in agents:
            cells = []
            for o in opps:
                r = next((x for x in store.data.values() if x.get("tag") == r_tag and x["a"] == a and x["b"] == o.label()), None)
                cells.append("-" if r is None else f"{r['score']:.2f} [{r['lo']:.2f}-{r['hi']:.2f}]")
            lines.append(f"| {r_tag[4:]} | {a} | " + " | ".join(cells) + " |")
    (res_dir / f"ablations_{preset}.md").write_text("\n".join(lines))
    print("\n".join(lines))


def run_hybrid_benchmark(cfg, weights: Path, quick: bool, workers: int):
    """STRETCH: equal wall-clock comparison of plain negamax, hybrid (value-head leaves) and PUCT."""
    res_dir = output_dir(cfg, "results")
    store = ResultStore(res_dir / "hybrid_results.json")
    n = 20 if quick else 100
    w = str(weights)
    lines = [f"Equal time per move, {n} games per pairing (both seats, random 4-ply openings), score of the first-named agent.", "",
             "| time/move | hybrid vs negamax | hybrid vs AZ-PUCT | AZ-PUCT vs negamax |", "|---|---|---|---|"]
    for t in ([0.05, 0.2] if quick else [0.05, 0.2, 0.5]):
        hy = AgentSpec("hybrid", depth=42, time_limit=t, weights=w)
        nm = AgentSpec("negamax", depth=42, time_limit=t)
        az = AgentSpec("az", sims=None, time_limit=t, weights=w, c_puct=cfg["mcts"]["c_puct"])
        cells = []
        for a, b in ((hy, nm), (hy, az), (az, nm)):
            r = store.run(a, b, n, cfg["seed"] + 5, cfg, workers, tag="hybrid")
            cells.append(f"{r['score']:.2f} [{r['lo']:.2f}-{r['hi']:.2f}]")
        lines.append(f"| {t}s | " + " | ".join(cells) + " |")
    (res_dir / "hybrid_table.md").write_text("\n".join(lines))
    print("\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--hybrid", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--ablations", action="store_true")
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--submission", default=None)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args(argv)
    cfg = load_config()
    workers = a.workers or cfg["eval"]["workers"]
    weights = Path(a.weights) if a.weights else output_dir(cfg) / "best_weights.npz"
    sub = Path(a.submission) if a.submission else ROOT / "submission" / "submission.py"
    if a.hybrid:
        run_hybrid_benchmark(cfg, weights, a.quick, workers)
    elif a.ablations:
        run_ablations(cfg, a.quick, workers)
    elif a.plots:
        for q in (False, True):
            p = output_dir(cfg, "results") / ("eval_quick.json" if q else "eval_results.json")
            if p.exists():
                make_eval_plots(cfg, ResultStore(p), q)
    else:
        if not weights.exists():
            raise FileNotFoundError(f"weights not found: {weights} (run train_az.py first)")
        run_main_benchmark(cfg, weights, a.quick or not a.full, workers, sub)


if __name__ == "__main__":
    main()
