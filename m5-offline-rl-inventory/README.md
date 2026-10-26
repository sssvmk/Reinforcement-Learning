# Offline RL for inventory control on M5 — can FQE rank mixed candidate pools?

CPU-only, seeded pipeline. M5 supplies demand, prices and calendar only; costs are assumptions.

## Run
```
set M5_DIR=<folder with M5 files>      # optional; otherwise m5_dir in config.yaml
python step0_inspect.py                # writes DATA_NOTES.md (read-only on M5_DIR)
python -m pytest tests -q              # 35 tests
python run_experiment.py --quick       # ~30 s smoke test (numbers meaningless)
python run_experiment.py --full --series primary --reps 1   # ~26 min
# also: --series all, --reps N, --ablations
```
Outputs go to `./outputs/` (series cache, logs, candidates, results, plots, `results/summary.md`).
M5 is never modified. Missing files raise an error naming the searched path. Needs torch (not installed by us).

## Step 0 findings
See `DATA_NOTES.md`. Observed data matched the documentation: 30,490 series, 1,913 `d_*` columns,
1,969 calendar days (2011-01-29..2016-06-19), 6.84M price rows; missing prices are absent rows (not NaN).
Series: primary `FOODS_3_228_CA_1` (mean 9.0, 0.6% zeros), intermittent `FOODS_1_141_CA_1` (60% zeros),
seasonal `FOODS_3_697_CA_2` (weekday ratio 2.86).

## Design
- **MDP** (`env.py`): lost sales, lead time L=2 (order placed on day t arrives at start of day t+L), pure `transition`,
  vectorized rollout, common random numbers by `(seed, episode_id)`. Costs: unit cost 0.6·mean price, holding 0.5%/day,
  stockout 0.2·price, small fixed cost. γ=0.99, H=90. Horizon end is **truncation**: bootstrapped by all learners and FQE.
- **Splits**: 60/20/20 chronological. Logs, demand model, action grid and base-stock S use the train period only;
  train-log and FQE-log are split by episode id; ground truth uses the test period. Tests check leakage.
- **Logs**: eps-soft base-stock (S tuned on train), default eps 0.2, 600 episodes. Behavior true value 597.7.
- **Discretization**: 12 position bins x 4 on-hand bins x 7 weekdays = 336 states. This makes the tabular MDP only approximately Markov.
- **Candidates (49)**: CE (VI, cross-checked with PI; mask / pessimism rules), offline Dyna-Q (no Dyna-Q+, which needs online exploration),
  BC, DQN, CQL, MCTS (open-loop UCT with train-period demand model; cached; `max_decisions` budget), and references incl. bad anchors.
- **Estimators**: FQE, CE model-based OPE, WIS, doubly robust, plus "self" (candidate's own value estimate), behavior and random baselines.
- **Metrics**: tie-aware Spearman, Kendall, normalized regret@1, precision@3, pairwise accuracy, MAE, bias; per family and overall.

## Results (primary series, 1 run, medium coverage; see `outputs/results/summary.md`)
| estimator | Spearman | regret@1 (norm.) | pairwise acc. | bias |
|---|---|---|---|---|
| FQE | 0.198 | 0.127 | 0.633 | -213 |
| DR | 0.382 | 0.002 | 0.687 | -203 |
| CE-OPE | -0.099 | 0.011 | 0.468 | +189 |
| WIS | 0.123 | 0.002 | 0.438 | +10 |
| self-estimate | -0.254 | 0.094 | 0.372 | +194 |

Within-family FQE Spearman is high for BC (0.95), CQL (1.0), DynaQ (0.91), CE (0.89), but negative for DQN (-0.41) and MCTS (-0.10).
Across the mixed pool it drops to 0.20, and FQE's top pick is not in the true top 3.

## Honest conclusions and where FQE fails
- **Answer: not reliably.** On this run FQE ranks well within a family, poorly across families. Do not read more into one series and one repetition.
- **FQE is under-converged**: K=25 rounds at γ=0.99 covers about 25 steps of the ~100-step effective horizon, giving a large negative bias (about -480 on good policies). The offset is roughly constant for similar policies, which is why within-family ranking survives, but it distorts comparisons of very different policies. More rounds (>=100) are needed; not run because of laptop time. This is the most likely fix to try first.
- **Ground truth horizon** is 200 steps (test period) versus training/log episodes of 90; WIS estimates the 90-step truncated return, so it is inconsistent with the target.
- **WIS** has an effective sample size near 1 for most candidates (degenerate). **CE-OPE** overestimates (optimistic fallback for unvisited state-actions).
- **DQN** candidates range from -210 to 621; FQE fails to rank them (support problems for unconstrained DQN). CQL is better behaved.
- **MCTS**: FQE is evaluated on a subsample of logged states with cached actions; distilled-proxy estimates differ (see summary table). Truth uses 8 episodes only.
- Behavior-policy value is a constant, so its "ranking" is random. Estimator ties are handled tie-aware.
- **Not done**: replication series, 5+ repetitions, ablations (`--ablations`), and coverage sweeps were implemented but not run. All reported numbers are from one series and one repetition. No hyperparameter selection on the validation log (fixed settings); validation log is diagnostic only.
- Bug history: unit tests caught an off-by-one lead time (effective L+1); fixed and all results regenerated after the fix.
## About Fitted Q Evaluation (FQE)

Fitted Q Evaluation (FQE) estimates how good a fixed policy would be from logged data, without deploying it. It repeatedly fits a Q-function to the target r + γ·Q(s’, π(s’)), where the next action comes from the policy being evaluated.

### Main applications

**Pre-deployment policy evaluation**
- Checking that a new policy beats the current one before it touches real users, patients or inventory.
- Safety gating: ship only if the estimated value exceeds the logging policy by a margin.

**Offline RL model selection**
- With no environment to test in, FQE scores candidate policies (CQL, BCQ, IQL or DQN checkpoints, different hyperparameters) so you can pick one. It's the standard validation step in offline RL.

**Healthcare**
- Evaluating treatment policies (sepsis management, ventilation, dosing) from historical records, where online testing is unethical.

**Recommendation and advertising**
- Evaluating sequential recommenders and ad-allocation policies that optimize long-term engagement, where trajectory-level importance sampling has exploding variance.

**Supply chain and operations**
- Estimating the value of a new ordering policy from logged inventory decisions before changing real orders.

**Robotics and autonomous driving**
- Ranking policies learned from logged data before risky real-world tests.

**Finance and pricing**
- Evaluating dynamic pricing or trade-execution policies from historical logs.

**Research**
- A standard baseline in off-policy evaluation benchmarks, and a building block for doubly robust estimators.

### Why use it instead of importance sampling
- Long horizons: importance weights multiply across steps and blow up. FQE bootstraps instead.
- Usually lower variance, and it works with function approximation.
- No behavior propensities needed.

### Limitations
- It's biased if the function class is wrong or the data doesn't cover the actions the target policy takes.
- It can diverge or overestimate, like other bootstrapped methods.
- It's sensitive to hyperparameters, so run several seeds.
- It has no built-in confidence intervals, so bootstrap over trajectories.

### Fit with your projects
FQE and importance sampling are often combined as doubly robust estimators. For your Criteo data, FQE works only if you frame banner filling as a slot-by-slot decision, since the raw data is a single-step bandit. For your M5 inventory work, it would evaluate an ordering policy from logged decisions. In the simulator setup, you can run the policy directly instead.
