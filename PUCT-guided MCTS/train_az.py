"""AlphaZero-style self-play training for ConnectX.   python train_az.py --quick | --medium | --full [--resume]"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from az_mcts import PlayerSpec, run_games_batched
from config import ROOT, load_config, output_dir, verify_kaggle_defaults
from game import random_opening
from network import TorchEvaluator, build_net, count_params, export_weights, masked_log_softmax


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ReplayBuffer:
    def __init__(self, capacity, planes, rows, cols):
        self.cap = capacity
        self.x = np.zeros((capacity, planes, rows, cols), dtype=np.float32)
        self.pi = np.zeros((capacity, cols), dtype=np.float32)
        self.z = np.zeros(capacity, dtype=np.float32)
        self.n = 0
        self.ptr = 0

    def add(self, x, pi, z):
        for i in range(len(x)):
            self.x[self.ptr], self.pi[self.ptr], self.z[self.ptr] = x[i], pi[i], z[i]
            self.ptr = (self.ptr + 1) % self.cap
            self.n = min(self.n + 1, self.cap)

    def sample(self, rng, bs):
        idx = rng.integers(0, self.n, size=bs)
        return self.x[idx], self.pi[idx], self.z[idx]

    def state_dict(self):
        return {"x": self.x[:self.n].copy(), "pi": self.pi[:self.n].copy(), "z": self.z[:self.n].copy(),
                "ptr": self.ptr, "n": self.n}

    def load_state_dict(self, d):
        n = min(d["n"], self.cap)
        self.x[:n], self.pi[:n], self.z[:n] = d["x"][:n], d["pi"][:n], d["z"][:n]
        self.n, self.ptr = n, d["ptr"] % self.cap


def augment_mirror(x, pi, z):
    """Left-right mirrored copies: planes flipped along columns, policy reversed, same value."""
    return x[..., ::-1].copy(), pi[:, ::-1].copy(), z.copy()


def selfplay(net, cfg, seed, n_games):
    """Generate games with the given network; returns (x, pi, z, stats). Noise ON, temperature for early plies."""
    m, t = cfg["mcts"], cfg["train"]
    ev = TorchEvaluator(net)
    spec = PlayerSpec(ev, t["sims_per_move"], m["c_puct"], noise=True, alpha=m["dirichlet_alpha"],
                      eps=m["dirichlet_eps"], temp_moves=m["temp_moves"], cache_max=m["cache_max"])
    g = cfg["game"]
    xs, pis, zs, plies, sims = [], [], [], [], 0
    done = 0
    while done < n_games:
        k = min(t["parallel_games"], n_games - done)
        res = run_games_batched([(spec, spec)] * k, [seed * 1_000_003 + done + i for i in range(k)],
                                rows=g["rows"], cols=g["columns"], inarow=g["inarow"])
        for r in res:
            xs += r.planes
            pis += r.pis
            zs += r.values()
            plies.append(len(r.moves))
            sims += r.sims
        done += k
    return (np.array(xs, dtype=np.float32), np.array(pis, dtype=np.float32), np.array(zs, dtype=np.float32),
            {"games": n_games, "positions": len(zs), "avg_plies": float(np.mean(plies)), "sims": sims,
             "nn_evals": ev.n_evals, "nn_seconds": ev.eval_seconds})


def train_steps(net, opt, buf, steps, batch_size, rng):
    """Fixed number of Adam steps. loss = CE(policy target, masked logits) + MSE(value). L2 via optimizer weight_decay."""
    net.train()
    lp_sum = lv_sum = 0.0
    for _ in range(steps):
        x, pi, z = buf.sample(rng, batch_size)
        x, pi, z = torch.from_numpy(x), torch.from_numpy(pi), torch.from_numpy(z)
        legal = (x[:, 0, 0, :] + x[:, 1, 0, :]) == 0          # top row empty <=> column playable
        logits, v = net(x)
        logp = masked_log_softmax(logits, legal).masked_fill(~legal, 0.0)
        lp = -(pi * logp).sum(1).mean()
        lv = torch.mean((v - z) ** 2)
        loss = lp + lv
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        lp_sum += lp.item()
        lv_sum += lv.item()
    return lp_sum / steps, lv_sum / steps


def arena(ev_a, ev_b, cfg, n_games, sims, seed, c_puct=None):
    """A vs B, argmax moves, no noise, both seats from the same random openings. Returns (score_a, w, d, l)."""
    g, t, m = cfg["game"], cfg["train"], cfg["mcts"]
    c = c_puct or m["c_puct"]
    sa = PlayerSpec(ev_a, sims, c, cache_max=m["cache_max"])
    sb = PlayerSpec(ev_b, sims, c, cache_max=m["cache_max"])
    rng = np.random.default_rng(seed)
    pairs, seeds, openings, a_is_first = [], [], [], []
    for i in range((n_games + 1) // 2):
        o = random_opening(rng, t["opening_plies"], g["rows"], g["columns"], g["inarow"])
        for first in (True, False):
            pairs.append((sa, sb) if first else (sb, sa))
            openings.append(o)
            seeds.append(seed * 7919 + len(seeds))
            a_is_first.append(first)
    res = run_games_batched(pairs, seeds, openings, g["rows"], g["columns"], g["inarow"], record=False)
    w = d = l = 0
    for r, first in zip(res, a_is_first):
        if r.winner == 0:
            d += 1
        elif (r.winner == 1) == first:
            w += 1
        else:
            l += 1
    return (w + 0.5 * d) / len(res), w, d, l


def estimate_runtime(cfg, net):
    """Rough runtime estimate from a short calibration (printed up front)."""
    t = cfg["train"]
    t0 = time.perf_counter()
    n = min(8, t["games_per_iter"])
    cal = copy.deepcopy(cfg)
    cal["train"].update(sims_per_move=10, parallel_games=n)
    x, pi, z, st = selfplay(net, cal, seed=12345, n_games=n)
    dt = max(time.perf_counter() - t0, 1e-6)
    sims_per_s = st["sims"] / dt
    plies = st["avg_plies"]
    sp = t["games_per_iter"] * plies * t["sims_per_move"] / sims_per_s
    ar = t["arena_games"] * plies * t["arena_sims"] / sims_per_s if t["gating"] else 0.0
    buf = ReplayBuffer(512, cfg["network"]["input_planes"], cfg["game"]["rows"], cfg["game"]["columns"])
    buf.add(x, pi, z)
    opt = torch.optim.Adam(net.parameters(), lr=t["lr"])
    t1 = time.perf_counter()
    train_steps(net, opt, buf, 5, min(t["batch_size"], buf.n), np.random.default_rng(0))
    tr = (time.perf_counter() - t1) / 5 * t["steps_per_iter"]
    prog = (t["progress_games"] * 4 * plies * t["progress_sims"] / sims_per_s) / max(t["progress_every"], 1)
    per_iter = sp + ar + tr + prog
    return per_iter * t["iterations"], per_iter, sims_per_s


def make_training_plots(run_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    log = run_dir / "logs" / "train_log.jsonl"
    if not log.exists():
        return
    rows = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
    pd = run_dir / "plots"
    pd.mkdir(parents=True, exist_ok=True)
    it = [r["iteration"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(it, [r["policy_loss"] for r in rows], label="policy CE")
    ax.plot(it, [r["value_loss"] for r in rows], label="value MSE")
    ax.plot(it, [r["policy_loss"] + r["value_loss"] for r in rows], "--", label="total")
    ax.set_xlabel("iteration"); ax.set_ylabel("mean loss"); ax.set_title("Training losses"); ax.legend()
    fig.tight_layout(); fig.savefig(pd / "training_losses.png", dpi=120); plt.close(fig)
    ar = [r for r in rows if r.get("arena_score") is not None]
    if ar:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([r["iteration"] for r in ar], [r["arena_score"] for r in ar], "o-")
        ax.axhline(rows[0].get("gate_threshold", 0.55), color="r", ls="--", label="promotion threshold")
        for r in ar:
            if r["promoted"]:
                ax.plot(r["iteration"], r["arena_score"], "g*", ms=12)
        ax.set_xlabel("iteration"); ax.set_ylabel("new-vs-best score (green star = promoted)")
        ax.set_title("Gating arena score"); ax.legend(); fig.tight_layout()
        fig.savefig(pd / "arena_score.png", dpi=120); plt.close(fig)
    pr = [r for r in rows if r.get("progress")]
    if pr:
        fig, ax = plt.subplots(figsize=(6, 4))
        for opp in pr[0]["progress"]:
            ax.plot([r["iteration"] for r in pr], [r["progress"][opp] for r in pr], "o-", label=opp)
        ax.set_ylim(0, 1.02); ax.set_xlabel("iteration"); ax.set_ylabel("score of best net (draw=0.5)")
        ax.set_title("Strength vs iteration (fixed opponents)"); ax.legend(); fig.tight_layout()
        fig.savefig(pd / "strength_vs_iteration.png", dpi=120); plt.close(fig)


def progress_eval(weights_path: Path, cfg, seed):
    """Best net (NumPy evaluator, PUCT with progress_sims) vs fixed opponents; returns {opponent: score}."""
    from evaluate import AgentSpec, play_match
    t = cfg["train"]
    out = {}
    for opp in ({"type": "random"}, {"type": "greedy"}, {"type": "negamax", "depth": 2}, {"type": "negamax", "depth": 4}):
        a = AgentSpec("az", sims=t["progress_sims"], weights=str(weights_path), c_puct=cfg["mcts"]["c_puct"])
        r = play_match(a, AgentSpec(**opp), t["progress_games"], seed, cfg, workers=1)
        out[AgentSpec(**opp).label()] = r["score"]
    return out


def train(cfg: dict, run_dir: Path, resume: bool = False, quiet: bool = False):
    t, g, n = cfg["train"], cfg["game"], cfg["network"]
    seed = cfg["seed"]
    set_seeds(seed)
    ck_dir = run_dir / "checkpoints"
    log_dir = run_dir / "logs"
    ck_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train_log.jsonl"
    net = build_net(cfg)
    best = copy.deepcopy(net)
    opt = torch.optim.Adam(net.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    buf = ReplayBuffer(t["buffer_size"], n["input_planes"], g["rows"], g["columns"])
    start_it = 0
    if resume:
        latest = ck_dir / "latest.pt"
        if not latest.exists():
            raise FileNotFoundError(f"--resume requested but no checkpoint at {latest}")
        ck = torch.load(latest, weights_only=False)
        net.load_state_dict(ck["net"]); best.load_state_dict(ck["best"]); opt.load_state_dict(ck["opt"])
        buf.load_state_dict(ck["buffer"]); start_it = ck["iteration"]
        print(f"[resume] from {latest} at iteration {start_it}")
    elif log_path.exists():
        log_path.unlink()

    print(f"seeds: python/numpy/torch = {seed}   torch threads = {torch.get_num_threads()}")
    print(f"preset={cfg['preset']}  network={n['type']} ({count_params(net):,} params)  run_dir={run_dir}")
    print("train config:", json.dumps(t))
    if not resume:
        est, per_it, sps = estimate_runtime(cfg, copy.deepcopy(net))
        print(f"estimated runtime: ~{est / 60:.1f} min  (~{per_it:.0f} s/iteration; calibration {sps:,.0f} sims/s)")
        set_seeds(seed)

    t_start = time.perf_counter()
    for it in range(start_it + 1, t["iterations"] + 1):
        set_seeds(seed * 1000 + it)
        t0 = time.perf_counter()
        x, pi, z, st = selfplay(best, cfg, seed * 1000 + it, t["games_per_iter"])
        t_sp = time.perf_counter() - t0
        if t["mirror_augment"]:
            mx, mpi, mz = augment_mirror(x, pi, z)
            x, pi, z = np.concatenate([x, mx]), np.concatenate([pi, mpi]), np.concatenate([z, mz])
        buf.add(x, pi, z)
        t0 = time.perf_counter()
        lp, lv = train_steps(net, opt, buf, t["steps_per_iter"], min(t["batch_size"], buf.n),
                             np.random.default_rng(seed * 1000 + it))
        t_tr = time.perf_counter() - t0
        score = w = d = l = None
        promoted = True
        t0 = time.perf_counter()
        if t["gating"]:
            score, w, d, l = arena(TorchEvaluator(net), TorchEvaluator(best), cfg, t["arena_games"], t["arena_sims"],
                                   seed * 1000 + it)
            promoted = score >= t["gate_threshold"]
        t_ar = time.perf_counter() - t0
        if promoted:
            best.load_state_dict(net.state_dict())
        else:
            pass
        wpath = export_weights(best, run_dir / "best_weights.npz")
        row = {"iteration": it, "policy_loss": lp, "value_loss": lv, "arena_score": score, "arena_w": w,
               "arena_d": d, "arena_l": l, "promoted": bool(promoted), "gate_threshold": t["gate_threshold"],
               "buffer": buf.n, "selfplay": st, "sec_selfplay": t_sp, "sec_train": t_tr, "sec_arena": t_ar,
               "pos_per_sec_nn": st["nn_evals"] / max(st["nn_seconds"], 1e-9), "progress": None}
        if t["progress_every"] and (it % t["progress_every"] == 0 or it == t["iterations"]):
            row["progress"] = progress_eval(wpath, cfg, seed * 1000 + it)
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        ck = {"net": net.state_dict(), "best": best.state_dict(), "opt": opt.state_dict(), "buffer": buf.state_dict(),
              "iteration": it, "cfg": cfg}
        torch.save(ck, ck_dir / "latest.tmp")
        os.replace(ck_dir / "latest.tmp", ck_dir / "latest.pt")
        torch.save({"net": net.state_dict(), "best": best.state_dict(), "iteration": it}, ck_dir / f"iter_{it:03d}.pt")
        arena_txt = "n/a" if score is None else f"{score:.3f} (W{w}/D{d}/L{l}) promoted={promoted}"
        prog_txt = "" if not row["progress"] else "  progress=" + ", ".join(f"{k}:{v:.2f}" for k, v in row["progress"].items())
        print(f"[it {it:3d}/{t['iterations']}] pos={st['positions']} buf={buf.n} Lp={lp:.3f} Lv={lv:.3f} arena={arena_txt} "
              f"t(sp/train/arena)={t_sp:.0f}/{t_tr:.0f}/{t_ar:.0f}s  elapsed={(time.perf_counter() - t_start) / 60:.1f}m{prog_txt}",
              flush=True)
    make_training_plots(run_dir)
    return best


def main(argv=None):
    ap = argparse.ArgumentParser()
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--quick", action="store_true")
    grp.add_argument("--medium", action="store_true")
    grp.add_argument("--full", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--tag", default=None, help="write to outputs/runs/<tag>/ instead of outputs/ (ablations)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--set", nargs="*", default=[], help="overrides like train.mirror_augment=false network.type=mlp")
    a = ap.parse_args(argv)
    preset = "quick" if a.quick else "medium" if a.medium else "full"
    cfg = load_config(preset)
    for kv in a.set:
        k, v = kv.split("=", 1)
        sect, key = k.split(".", 1)
        try:
            val = json.loads(v)
        except ValueError:
            val = v
        cfg[sect][key] = val
    if a.seed is not None:
        cfg["seed"] = a.seed
    try:
        defaults = verify_kaggle_defaults(cfg)
        print(f"KAGGLE_DIR={cfg['kaggle_dir']} (read-only); connectx.json defaults: {defaults}")
        g = cfg["game"]
        if (defaults["rows"], defaults["columns"], defaults["inarow"]) != (g["rows"], g["columns"], g["inarow"]):
            print("NOTE: config game size differs from the competition defaults.")
    except FileNotFoundError as e:
        print(f"WARNING: {e}", file=sys.stderr)
    run_dir = output_dir(cfg, *(("runs", a.tag) if a.tag else ()))
    train(cfg, run_dir, a.resume)


if __name__ == "__main__":
    main()
