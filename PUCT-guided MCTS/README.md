# AlphaZero-style ConnectX (PUCT-MCTS + small policy/value net), CPU only

Self-play-trained PUCT-MCTS agent for ConnectX, with a custom bitboard engine, baselines (random, greedy, negamax, UCT-MCTS),
a PyTorch training loop, a NumPy-only single-file submission, an evaluation harness and tests.
No Kaggle package is imported; nothing is downloaded or submitted.

## Local folder
`kaggle_dir` is defined once in `config.yaml` and can be overridden with the env var `KAGGLE_DIR`
(PowerShell: `$env:KAGGLE_DIR="C:\path\to\kaggle-environments-0.1.4"`). It is read in place, read-only (a test checks nothing is written there).
Everything generated goes under `outputs/`.

## Step 0 findings (details and the full UNVERIFIED list in `ENV_NOTES.md`)
- `KAGGLE_DIR` holds a `kaggle-environments` **v0.1.4 source copy** (28 files, incl. `connectx.json/.py/.js`, tests, notebook). No competition notes, no data.
- Confirmed from `connectx.json`: 6 rows x 7 columns, `inarow` 4. Board = flat list, **top row first**, 0/1/2; observation has `board` and `mark` (no `step` in this version).
  Time limit in this version is `configuration.timeout` (default 2 s); no `actTimeout`/overage. An invalid column loses.
- **UNVERIFIED** (check the competition page yourself): the real time limit/overage, available packages for the submission, the submission file format, and the current observation fields.
- Environment used: Python 3.13, torch 2.14.1+cpu (8 threads), numpy 2.2.6. Nothing was installed.

## Encoding
- Bitboard: bit `c*(rows+1)+r`, r=0 bottom, one sentinel bit on top of each column. `to_key()` = `(mask << (cols*(rows+1)+1)) | cur`.
- `to_planes()` -> float32 `[2, rows, cols]`, **top row first** (Kaggle orientation), from the perspective of the **player to move**:
  plane 0 = mover's checkers, plane 1 = opponent's. (Optional plane 2 = constant "mover is player 1" flag; off by default.)
- Value: +1 win / -1 loss / 0 draw for the player to move. Illegal columns are masked before the softmax.

## Network
CNN: 3 x conv3x3 (pad 1, width 32, BN, ReLU); policy head flatten->Linear(7); value head flatten->Linear(64)->ReLU->Linear(1)->tanh. **114,760 params**.
MLP variant (`network.type: mlp`): 36,616 params. Export folds BN into the convs (`outputs/best_weights.npz`, float32).

## PUCT and training
- Selection: `argmax_a Q(a) + c_puct * P(a) * sqrt(sum_b N(b)) / (1+N(a))`, `Q=W/N` (0 if unvisited), `c_puct=1.5`. No rollouts; leaf value = value head,
  terminal = exact (-1 for the player to move after the opponent won). Backup flips sign each ply.
- Self-play only: root Dirichlet noise (alpha 1.0, eps 0.25), temperature 1 for the first 10 plies then argmax. Evaluation/submission: no noise, argmax visits.
- Sim budget and wall-clock budget, subtree reuse, per-game evaluation cache, lockstep batched self-play (one NN call per round).
- Training: mirror augmentation (flag), replay buffer, Adam, loss = CE(policy) + MSE(value) + L2 weight decay, gating arena (both seats, argmax, promote if score >= 0.55), checkpoint every iteration (`--resume`), progress evaluation vs fixed opponents every 4 iterations.
- **Full preset used**: 40 iterations, 64 games/iter, 100 sims/move, buffer 100k, batch 256, 200 steps/iter, lr 0.002, wd 1e-4, mirror on, gating 0.55 with 40 arena games at 50 sims, seed 0. Took ~19 min.
- Arena/eval games start from random 4-ply openings (each opening played in both seats) so that deterministic argmax agents produce varied games. This is a deliberate deviation from playing only from the empty board.

## How to run
```
pip install -r requirements.txt          # only if something is missing; torch must already exist
python train_az.py --quick               # smoke test, ~20 s  (--tag NAME writes to outputs/runs/NAME)
python train_az.py --full                # ~20 min, prints estimated runtime first
python train_az.py --full --resume
python submission\build_submission.py    # export weights, parity check (1000 positions), build submission\submission.py
python evaluate.py --quick | --full      # benchmark (full: ~35 min)
python evaluate.py --ablations           # small ablation runs
python evaluate.py --hybrid              # stretch
python -m pytest tests -q                # 49 tests, ~90 s
```

## Evaluation methodology and metric definitions
**Game and score.** Each pairing is `n` games (200; 20 with `--quick`; 40 for the submission agent; 60 in ablations; 100 in the hybrid benchmark), played from random 4-ply openings.
Every opening is played twice, with the two agents swapping seats, so both seats are always covered. Agents are deterministic (argmax), so the openings provide the variety.
Per game the first-named agent A scores 1 (win), 0.5 (draw) or 0 (loss). An illegal move scores a loss for the offender (none occurred).
**Score** = mean of these per-game scores = (W + 0.5 D) / n. W/D/L counts are given from A's side.

**Confidence interval.** Wilson 95% interval (z = 1.96) on the score, treating draws as half a win: `centre = (p + z^2/2n) / (1 + z^2/n)`, `half = z*sqrt(p(1-p)/n + z^2/4n^2) / (1 + z^2/n)`.
This is an approximation (draws are not Bernoulli), but it is simple and conservative enough for comparing cells. With n=200 the interval is about +/-0.07 near 0.5; with n=60 about +/-0.12.
Treat a result as different from 0.5 only if its interval excludes 0.5. Cells are compared independently; no multiple-comparison correction was applied.

**Agents under test.**
- *policy-only*: argmax of the masked network policy, zero simulations.
- *PUCT N sims*: PUCT search with N simulations per move, `c_puct` 1.5, no Dirichlet noise, argmax of root visits, no subtree reuse in the evaluation matches unless stated. The real sims/move can be slightly below N (terminal and forced moves), see the sims/move table.
- *submission-np*: `submission/submission.py` through the NumPy network at its real time budget (min(0.5*limit, limit-0.4) s with the 2 s default limit, at most 5000 sims).

**Opponents.**
- *random*: uniform legal move. *greedy*: win if possible, else block, else centre-preferring move.
- *negamax-dD*: fail-hard alpha-beta with iterative deepening to depth D, centre-first ordering (previous best and transposition-table move first), a transposition table, and a window-count heuristic at the horizon (scores +/-(WIN - ply) for decided positions, so quicker wins are preferred).
  Capped at 5 s per move. Negamax uses full depth D at every move and is not time-matched to the AZ agent; at D=6/8 it takes about 5-20 ms per move here, so it is also cheaper than the AZ agents.
  *negamax-d42-tT* (hybrid benchmark only) means iterative deepening limited by time T per move, not depth.
- *mcts-rand / mcts-heur*: plain UCT, exploration constant 1.4, random rollouts (rand) or rollouts that win/block when possible and otherwise play randomly (heur), robust-child (most visited) move choice.
  *Equal simulations*: the MCTS gets the same N as the AZ agent (`mcts-...-Nsims`). *Equal time*: the MCTS gets a wall-clock budget per move equal to the AZ agent's measured mean seconds per move (`mcts-...-t0.xxxs`, taken from the matching AZ row), so the budget depends on this machine.

**Timing columns.** *ms/move* is the mean wall-clock time of the agent's `select` calls, measured inside worker processes while several games ran in parallel (about 12 workers on 16 logical CPUs), so absolute values include CPU contention and the numbers are only comparable within one run.
The AZ evaluation uses the NumPy network (batch size 1 per leaf), not PyTorch.

**Training metrics (logs and `training_losses.png`, `arena_score.png`, `strength_vs_iteration.png`).**
- *Lp*: mean cross-entropy between the MCTS visit distribution and the network policy over the iteration's gradient steps. It cannot reach 0 because targets are soft distributions. *Lv*: mean squared error of the value head against the game outcome (+1/0/-1 from the mover's view). The weight decay term is not included in either number.
- *pos*: number of positions generated in that iteration (before mirroring); *buf*: replay-buffer size (after mirroring).
- *arena*: score of the new network against the current best (40 games, both seats, argmax, 50 sims, random openings), same score definition as above. Promoted if >= 0.55. With 40 games the standard error is about 0.08, so single arena results are noisy.
- *progress*: every 4 iterations the current best network (50 sims, no noise) plays 20 games per opponent (random, greedy, negamax-d2, negamax-d4). With only 20 games the standard error is up to about 0.11, so the strength curve shows the trend, not precise values.

**Seeds.** Seed 0 for Python, NumPy and PyTorch; evaluation openings and per-game agent seeds are derived from it, so a re-run of the same command gives the same openings. Results are cached by pairing in `outputs/results/*.json`. Time-limited agents (equal-time MCTS, the submission, hybrid) are not exactly reproducible because they depend on machine speed.

**Independent check.** `play_kaggle_style` plays games only through the public `agent(observation, configuration)` interface with dict observations (board top row first, `mark`) using the slow reference engine, enforcing legality and a per-move timeout. It is used for the submission tests, not for the benchmark tables.

## Results (trained net: 40 iterations, `outputs/best_weights.npz`)
Score = wins + 0.5 draws, both seats, 200 games per cell (submission: 40), Wilson 95% CI. Full table: `outputs/results/eval_table.md`.

| AZ agent (ms/move) | random | greedy | negamax d2 | d4 | d6 | d8 |
|---|---|---|---|---|---|---|
| policy-only (1) | 0.94 [0.90-0.97] | 0.35 [0.29-0.42] | 0.12 [0.08-0.18] | 0.08 [0.05-0.13] | 0.09 [0.06-0.14] | 0.09 [0.06-0.14] |
| PUCT 50 sims (50) | 1.00 [0.98-1.00] | 0.81 [0.76-0.86] | 0.30 [0.25-0.37] | 0.29 [0.23-0.36] | 0.26 [0.20-0.32] | 0.23 [0.18-0.29] |
| PUCT 200 sims (120) | 0.99 [0.97-1.00] | 0.87 [0.82-0.91] | 0.41 [0.34-0.48] | 0.40 [0.33-0.46] | 0.30 [0.24-0.36] | 0.24 [0.19-0.31] |
| PUCT 800 sims (381) | 0.99 [0.97-1.00] | 0.92 [0.87-0.95] | 0.59 [0.52-0.66] | 0.55 [0.48-0.61] | 0.44 [0.38-0.51] | 0.34 [0.28-0.41] |
| submission NumPy, ~0.87 s/move, n=40 | 1.00 [0.91-1.00] | 0.94 [0.82-0.98] | 0.66 [0.51-0.79] | 0.60 [0.45-0.74] | 0.45 [0.31-0.60] | not run |

Versus plain UCT-MCTS (score of the AZ agent):

| AZ agent | MCTS random rollouts, equal sims | MCTS random, equal time | MCTS heuristic, equal sims | MCTS heuristic, equal time |
|---|---|---|---|---|
| 50 sims | 0.86 | 0.46 | 0.80 | 0.26 |
| 200 sims | 0.65 | 0.28 | 0.45 | 0.31 |
| 800 sims | 0.53 | 0.35 | 0.41 | 0.39 |

**Honest reading**
- Beats random and greedy clearly; at equal simulations it beats random-rollout MCTS (clear at 50-200, roughly equal at 800) but **not** heuristic-rollout MCTS at 200-800 sims.
- At **equal wall-clock time, plain MCTS beats the AZ agent** (our Python network evaluations are slower per simulation than a rollout). The small net trained for 20 min is not worth its inference cost against well-tuned MCTS.
- **Negamax at depth 6 and 8 beats the AZ agent at every simulation budget tested** (d6: 0.44 at 800 sims, 0.45 for the submission; d8: 0.34 at 800 sims). Only at <=800 sims versus depth 2-4 does the AZ agent get to or above 0.5. Note that negamax depth 6/8 plays about 5-20 ms per move here, so it is also far cheaper than the AZ agent.
- Policy-only is weak (loses to greedy, 0.35): the net alone is not tactically sound; the search provides the tactics.
- Stronger training (many more iterations/games/sims) would very likely help, but was not done. Training progress curve is noisy (20-game samples): vs negamax-d4 it moved from 0.15 to about 0.3.

### Training plots (`outputs/plots/`)
`training_losses.png`, `arena_score.png`, `strength_vs_iteration.png`, `winrate_vs_budget.png`, `seconds_per_move.png`.
Policy loss fell from 1.81 to 1.38; value loss rose from 0.27 to ~0.45 as self-play games became harder to predict. Arena scores hover around 0.5 (most iterations are statistically no better than noise at 40 games), so gating mostly filters noise.

### Ablations (`outputs/results/ablations_medium.md`)
Medium preset (12 iterations, 32 games, 50 sims, 100 steps), **single seed, 60 games per cell** (CIs about +/-0.1). All variants are very weak vs negamax (0.02-0.15) and differences are within noise.

| variant | policy-only vs greedy | 50 sims vs greedy | 50 sims vs negamax-d2 | 50 sims vs negamax-d4 |
|---|---|---|---|---|
| base (CNN, mirror, gating) | 0.10 | 0.52 | 0.15 | 0.14 |
| no mirror augmentation | 0.12 | 0.37 | 0.09 | 0.13 |
| no gating | 0.16 | 0.61 | 0.08 | 0.15 |
| MLP | 0.12 | 0.48 | 0.12 | 0.10 |

- c_puct sweep (200 sims, vs negamax-d2/d4): 1.0 -> 0.09/0.10, 1.5 -> 0.12/0.12, 2.5 -> 0.12/0.08. No significant difference.
- Network vs random rollouts at equal simulations: see the MCTS table above (full-size net).
- No conclusion on mirror/gating/CNN-vs-MLP can be drawn from these small runs; the MLP is ~6x faster per search (6 ms vs 28 ms per 50 sims) at similar strength.

### Stretch: hybrid (`hybrid.py`, kept separate)
Iterative-deepening negamax with the value head as leaf evaluation and policy-ordered root moves; equal time, 100 games per cell.

| time/move | hybrid vs negamax | hybrid vs PUCT | PUCT vs negamax |
|---|---|---|---|
| 0.05 s | 0.32 [0.23-0.41] | 0.59 [0.50-0.69] | 0.24 [0.17-0.33] |
| 0.2 s | 0.28 [0.20-0.37] | 0.49 [0.40-0.59] | 0.28 [0.21-0.38] |
| 0.5 s | 0.27 [0.19-0.36] | 0.54 [0.44-0.63] | 0.33 [0.24-0.42] |

The hybrid is **worse than plain negamax** at equal time (the net's leaf values are less reliable than exact search plus the window heuristic) and roughly on par with PUCT. It does not help. Caveat: at 0.05 s the hybrid overshot its time budget (171 ms mean per move), which flatters it in that row.

## Submission
`submission/submission.py`: one self-contained file, **690 KiB** (451 KiB weights embedded as base64 npz), imports only numpy and the standard library.
`agent(observation, configuration)` rebuilds the board from `board` (top row first) and `mark`, runs NumPy PUCT (no noise), reads the limit from `actTimeout`, else `timeout`, else 2 s, searches for `min(0.5*limit, limit-0.4)` s (max 5000 sims), always returns a legal int and falls back to a centre-preferring legal move on any exception.
Max measured move time about 1.0 s. Tested with a mock observation/configuration harness; NumPy/PyTorch parity on 1000 random positions: max |dlogit| 1.8e-6, |dvalue| 2.2e-6.
**The real Kaggle submission format, limits and package availability are UNVERIFIED. Check the competition page yourself.**

## Tests
49 pytest tests cover: win detection vs reference, plane perspective, mirror, masking, PUCT (budget, legal move, immediate win/block with random weights), backup sign, noise only in self-play, self-play data shapes/targets, training smoke, NumPy/PyTorch parity, submission legality on 1000 positions, determinism, read-only `KAGGLE_DIR`. All pass.

## Limitations
- ConnectX 6x7 is a solved game (first player wins); strength is only relative. Well-optimised alpha-beta is expected to be very strong; here it was, and our laptop-trained agent does not match it at depth 6+.
- Short training (one 19-min run, single seed); progress and ablation samples are small and noisy. The agent may overfit its own self-play; fixed baselines in both seats are the check.
- Arena and evaluation use random 4-ply openings (documented deviation). Negamax per-move time capped at 5 s.
- Equal-time MCTS uses the AZ agent's measured mean ms/move; timings depend on this machine and on 16 parallel eval workers.
- Real competition time limits, packages and submission format: UNVERIFIED.
