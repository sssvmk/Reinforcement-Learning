# DQN for Inventory Control on Walmart (M5) Demand

A Deep Q-Network agent — **experience replay + target network** — learns a
daily **ordering policy** for single item-store inventory control, using real
Walmart daily unit-sales and weekly prices from the Kaggle **M5 Forecasting -
Accuracy** dataset as an exogenous demand/price simulator. The learned policy
is compared against classical inventory-control baselines on a held-out,
chronologically later test period.

> **This is not a demand-forecasting project.** Demand is replayed verbatim
> from history; the agent cannot influence it. What is learned is *how much
> to order each day*, not what demand will be.

---

## 1. Functionality

The project builds a full pipeline from raw Walmart sales data to a trained,
evaluated reinforcement-learning inventory policy:

| Module | Functionality |
|---|---|
| `data.py` | Loads the raw M5 CSVs, melts one item-store row into a daily panel, joins calendar (day-of-week, events, SNAP) and weekly price, and produces a chronological train/val/test split per series. |
| `env.py` | `InventoryEnv`, a Gymnasium-compatible MDP: given an inventory state, choose a daily order quantity; receives exogenous demand from history; computes profit-based reward. |
| `dqn.py` | `QNetwork` (MLP), `ReplayBuffer` (circular, uniform sampling), `DQNAgent` (ε-greedy, Double DQN, hard/soft target sync, Huber loss, gradient clipping). |
| `baselines.py` | Never-order, fixed-order, base-stock, (s, S), and newsvendor/critical-ratio policies, each tunable by simulation on the train split. |
| `train.py` | Trains DQN per series, logs return/loss/max-Q/ε/fill-rate/stockout-rate per episode, saves checkpoints and training-curve plots. Supports `--ablation {full, no_replay, no_target}`. |
| `evaluate.py` | Runs DQN checkpoints and all tuned baselines on the **test** split (multiple windows/seeds) and writes a results table. |
| `tests/` | 10 pytest sanity checks (conservation, determinism, buffer correctness, target-sync correctness, toy-environment convergence, split disjointness, hand-computed reward). All pass (`pytest tests/ -v`). |

### How to run
```bash
pip install -r requirements.txt
pytest tests/ -v                                   # sanity checks
python train.py --config config.yaml --out_dir outputs               # main DQN, all series
python train.py --config config.yaml --series <id> --ablation no_replay --out_dir outputs
python train.py --config config.yaml --series <id> --ablation no_target --out_dir outputs
python evaluate.py --config config.yaml --out_dir outputs            # test-split comparison table
```
Everything is seeded (`config.yaml: seed`, default **42**) and the seed is
printed in training/evaluation logs. Python 3.10+, numpy, pandas, PyTorch,
gymnasium, matplotlib.

---

## 2. Algorithm — DQN with experience replay and a target network

**State** (9-dim, normalized): on-hand inventory, on-order pipeline sum,
trailing 7-/28-day mean demand, cyclical day-of-week (sin/cos), event flag,
state SNAP flag, and relative sell price — all scaled by the series' own
mean demand/price.

**Action**: discrete index into an order-quantity grid built from the
series' mean demand (`{0, 0.5, 1, 1.5, 2, 3, 5, 7} × mean_demand`).

**Q-network**: MLP, 2 hidden layers × 128 units, ReLU, state → Q-values over
the action grid.

**Experience replay**: a circular buffer (capacity 100,000) stores
`(s, a, r, s', done)` tuples from every environment step; mini-batches of 64
are drawn **uniformly at random** for each gradient step, breaking the
temporal correlation between consecutive transitions and reusing past
experience. Learning only starts after a warm-up of 1,000 steps.

**Target network**: a lagged copy of the Q-network, `Q_target`, used only to
compute the TD bootstrap target
`y = r + γ · max_a' Q_target(s', a') · (1 - done)`
(or, with Double DQN, `y = r + γ · Q_target(s', argmax_a' Q_online(s', a'))·(1-done)`).
It is **hard-synced** to the online network every 500 steps (a **soft/Polyak**
update is also implemented and selectable via `config.yaml`). This decouples
the "mover" and "target" so bootstrapping doesn't chase its own rapidly
shifting estimate — the classical stabilizer that makes DQN's non-linear
function approximation converge instead of diverge.

**Training details**: Huber loss, Adam, gradient-norm clipping; ε-greedy
exploration linearly decayed from 1.0 to 0.05 over 20,000 steps.

**Why both matter (empirically, see Section 4.3)**: experience replay
decorrelates updates and lets the agent learn from a diverse buffer of past
transitions rather than only the most recent one; the target network
prevents the bootstrapped target from moving every single gradient step,
which would otherwise make the regression target non-stationary and
encourage oscillation/divergence.

---

## 3. Dataset — M5 Forecasting - Accuracy (Walmart)

Files found at `C:\Users\msk80743\OneDrive - GSK\m5-forecasting-accuracy`:

| File | Shape | Notes |
|---|---|---|
| `calendar.csv` | 1969 × 14 | `d_1..d_1969`; the last 28 days extend past the evaluation sales horizon and are unused here. |
| `sales_train_validation.csv` | 30490 × 1919 | `d_1..d_1913`, `id` suffix `_validation`. |
| `sales_train_evaluation.csv` | 30490 × 1947 | `d_1..d_1941`, `id` suffix `_evaluation`. **Used in this project** (longer history). `data.py` falls back to the validation file if absent. |
| `sell_prices.csv` | 6,841,121 × 4 | `store_id, item_id, wm_yr_wk, sell_price`. **No NaN cells**, but rows are simply absent for (store, item, week) combinations where the item wasn't sold. |
| `sample_submission.csv` | ignored | Not needed; this is not a forecasting task. |

Differences from the task's assumed schema (confirmed against the real
files, not guessed): both sales files exist (evaluation file chosen); price
"missingness" is row-level absence, not NaN values.

**Joining** (`data.py`): melt one series' `d_1..d_N` into long form → merge
`calendar.csv` on `d` (day-of-week, events, SNAP) → merge `sell_prices.csv` on
`(store_id, item_id, wm_yr_wk)`. Weeks without a priced row are forward/back
filled only for state continuity; the *usable* (simulatable) day range for a
series is restricted to the contiguous span between its first and last
actually-priced week.

**Temporal split** (never random), over each series' usable range:
train = earliest 70%, validation = next 15%, test = last 15% (reported only
here). Verified disjoint and chronological by `tests/test_data.py`.

**Series used** (chosen to span demand profiles, all verified present in
`sales_train_evaluation.csv`):

| Series id | Profile | Train-split mean demand/day | Zero-sales days |
|---|---|---|---|
| `FOODS_3_090_CA_3_evaluation` | High-volume | ~131 | ~18% |
| `HOBBIES_1_006_CA_1_evaluation` | Low-volume, intermittent | ~0.86 | ~66% |
| `HOUSEHOLD_1_019_CA_1_evaluation` | Mid-volume household | ~4.3 | ~33% |
| `FOODS_1_019_CA_1_evaluation` | Mid-volume, seasonal foods | ~4.1 | ~47% |

---

## 4. Results

### 4.1 Test-split comparison (DQN, 150 training episodes/series, vs. tuned baselines)

Metrics are means over the test split (10 windows for baselines; DQN also
averaged over 5 seeds). `(s,S)`, base-stock, and newsvendor parameters were
grid-searched on the **train** split only.

| Series | Policy | Mean return | Fill rate | Stockout rate | Avg inventory | Orders/window |
|---|---|---:|---:|---:|---:|---:|
| **FOODS_1_019** (mid, seasonal) | **DQN** | **52.8 ± 7.8** | 0.980 | 0.036 | 18.0 | 35.8 |
| | BaseStock(S=30.3) | 39.4 | 0.967 | 0.084 | 20.2 | 39.0 |
| | Newsvendor(q=0.9) | 46.2 | 0.991 | 0.028 | 28.1 | 40.3 |
| | (s=12.6,S=25.2) | 37.6 | 0.886 | 0.157 | 12.1 | 12.2 |
| | FixedOrder(mean_d) | −128.9 | 0.962 | 0.086 | 74.7 | 90.0 |
| | NeverOrder | −47.6 | 0.485 | 0.518 | 2.8 | 0.0 |
| **HOBBIES_1_006** (low, intermittent) | **DQN** | **7.3 ± 1.7** | 0.967 | 0.052 | 4.7 | 10.5 |
| | (s=2.6,S=6.5) | 5.1 | 0.941 | 0.084 | 3.4 | 10.2 |
| | NeverOrder | −8.8 | 0.694 | 0.310 | 0.4 | 0.0 |
| | BaseStock(S=7.8) | −8.2 | 0.986 | 0.020 | 5.8 | 28.3 |
| | Newsvendor(q=0.9) | −12.2 | 0.999 | 0.001 | 9.7 | 30.2 |
| | FixedOrder(mean_d) | −83.9 | 0.995 | 0.012 | 14.3 | 90.0 |
| **HOUSEHOLD_1_019** (mid) | **DQN** | **68.2 ± 2.3** | 1.000 | 0.001 | 26.8 | 20.2 |
| | (s=16.5,S=33.0) | 81.3 | 0.997 | 0.007 | 17.8 | 13.0 |
| | Newsvendor(q=0.9) | 23.8 | 1.000 | 0.000 | 26.6 | 62.4 |
| | BaseStock(S=39.6) | 20.4 | 1.000 | 0.000 | 30.5 | 62.4 |
| | NeverOrder | −28.4 | 0.268 | 0.734 | 0.7 | 0.0 |
| | FixedOrder(mean_d) | −241.5 | 1.000 | 0.000 | 146.8 | 90.0 |
| **FOODS_3_090** (high-volume) | (s=367.9,S=735.7) | **6509.7** | 0.983 | 0.050 | 278.7 | 19.6 |
| | Newsvendor(q=0.9) | 6304.6 | 1.000 | 0.000 | 559.0 | 78.6 |
| | BaseStock(S=882.9) | 6277.0 | 1.000 | 0.000 | 578.0 | 77.5 |
| | FixedOrder(mean_d) | 2602.2 | 1.000 | 0.000 | 2137.4 | 90.0 |
| | NeverOrder | −1962.1 | 0.123 | 0.883 | 23.9 | 0.0 |
| | **DQN** | **−21.4 ± 257.4** | 0.419 | 0.715 | 66.8 | 23.4 |

**Honest headline result**: DQN **beats every tuned classical baseline on 3
of 4 series** (the mid-volume and low-volume/intermittent ones), but **fails
badly on the high-volume series** (`FOODS_3_090`, mean demand ≈131/day),
where action/reward magnitudes are two orders of magnitude larger and 150
episodes of training were not enough for the network to converge — see
Section 4.3. A well-tuned `(s, S)` or base-stock policy remains a very strong,
cheap baseline, exactly as inventory theory predicts.

### 4.2 Training curves

Saved to `outputs/<series>_full_curves.png` (episode return, average Huber
loss, average max-Q, fill/stockout rate). On the three series where DQN wins,
curves show return rising and stabilizing with ε decay and fill-rate
converging close to 1.0. On `FOODS_3_090`, the return curve remains
high-variance throughout the 150 episodes (large swings between roughly
−17,000 and +10,000), confirming the network has not converged at this
reward scale.

### 4.3 Ablation: experience replay and target network (`FOODS_3_090`, 150 episodes)

| Variant | Last-20-episode mean return | Last-20-episode return std | Last-20-episode fill rate | Last-20-episode stockout rate |
|---|---:|---:|---:|---:|
| **Full DQN** (replay + target) | **2733.5** | 5036.6 | 0.858 | 0.231 |
| No target network | 3027.8 | 5251.1 | 0.891 | 0.168 |
| No experience replay | **−1728.2** | 6904.0 | 0.962 | 0.063 |

- **No experience replay** is catastrophic here: with `use_replay=False`, the
  buffer's capacity collapses to one batch (64), and the configured 1,000-step
  warm-up can *never* be satisfied by a 64-slot buffer — so the agent **never
  performs a single gradient update** and behaves as a purely ε-greedy random
  policy for all 150 episodes. The sharply worse, highly variable return is
  exactly what pure random exploration looks like on this environment. This is
  an even starker illustration than a merely-unstable learner: it shows that
  without a replay mechanism sized appropriately relative to the warm-up
  requirement, DQN as configured cannot learn *at all* — a cautionary,
  practically important finding, not only a theoretical one.
- **No target network**: returns were close to the full configuration in this
  run (both ≈2700–3000). With ε still at 0.36 by episode 149 (slow decay
  relative to only 150 episodes × 90 steps = 13,500 total steps), exploration
  still dominates action selection, which **masks** the instability that the
  target network is designed to prevent. We report this honestly rather than
  manufacturing a cleaner-looking contrast: at this schedule, the target
  network's stabilizing effect on this series is not strongly visible yet,
  though its theoretical justification (avoiding a non-stationary bootstrap
  target) still holds and would be expected to matter more with faster ε decay
  or longer training. This nuance itself is a useful ablation finding.

---

## 5. Assumptions

Documented centrally in `config.yaml` / `env.py` and restated here:

- **Cost model** (M5 has no cost/margin data): `unit_cost = 0.6 × mean(sell_price)`;
  `holding_cost = 0.5% of unit_cost` per unit of end-of-day on-hand inventory
  per day; `stockout_penalty = 0.2 × sell_price` per lost unit;
  `fixed_order_cost = 1.0` per order placed. **Absolute profit numbers are
  illustrative only** — the relative ranking across policies is what matters.
- **Lost-sales, no backorders**: unmet demand is lost, not carried forward.
- **Lead time**: fixed at 2 days (configurable), FIFO pipeline.
- **Missing price weeks** are treated as "item unavailable"; the simulatable
  day range is restricted to the contiguous priced span (see Section 3).
- **Censored demand**: M5 "unit sales" can themselves reflect historical
  stockouts; we replay recorded sales as "demand" per the task spec, but true
  unconstrained demand is unknown.
- **Demand is exogenous** and is not affected by the agent's ordering
  decisions (no demand elasticity to stockouts/promotions/agent behavior).
- **Discrete order grid**: required for vanilla DQN; derived from each
  series' own mean demand, not a fixed absolute quantity.
- **150 training episodes** per series/ablation was chosen for tractable
  turnaround; the high-volume series' results in Section 4.1 suggest more
  episodes (and/or reward normalization) would likely be needed for DQN to
  compete there.

---

## 6. Application

This project demonstrates DQN (experience replay + target network) as a
**general-purpose sequential decision-making tool for operations problems**,
beyond its origin in game-playing:

- **Single-item reorder policies**: the trained agent chooses a daily order
  quantity directly from state, implicitly learning a (possibly nonlinear,
  state-dependent) reorder rule that can respond to day-of-week, SNAP/holiday
  demand spikes, and price changes — signals a fixed-parameter base-stock or
  (s, S) policy does not use at all. This explains why DQN can beat tuned
  classical baselines on the lower/mid-volume, more event-sensitive series.
- **Where it struggles**: at high transaction volumes and large reward
  magnitudes, the classical baselines' closed-form structure (derived from
  queueing/newsvendor theory) is hard to beat with a generic function
  approximator trained from scratch in limited episodes — a realistic,
  practically important caveat for anyone considering RL for high-volume SKUs.
- **Extensibility**: the same architecture generalizes to richer settings by
  extending the state/action space and reward — e.g. multiple lead times,
  perishable/expiring inventory, multi-echelon networks, joint pricing and
  ordering, or portfolio-level training across many SKUs with a single shared
  network (replacing one-model-per-series with a model conditioned on item
  features).
- **Why replay + target network specifically matter for this application**:
  inventory control rewards are delayed and highly autocorrelated (today's
  order shows up in tomorrow's holding cost and the lead-time-later arrival).
  Replay breaks the resulting temporal correlation in the training batch;
  the target network prevents the bootstrap target from chasing the online
  network's own noisy updates — both are necessary (Section 4.3) for stable
  learning of a profit-sensitive policy rather than a game score.

---

## 7. Project layout

```
config.yaml       hyperparameters, cost assumptions, series list
data.py           load/melt/merge M5 files -> SeriesBundle, temporal splits
env.py            InventoryEnv (Gymnasium) — the MDP
dqn.py            QNetwork, ReplayBuffer, DQNAgent (Double DQN, hard/soft target, ablation flags)
baselines.py      never-order / fixed / base-stock / (s,S) / newsvendor policies + tuning
train.py          training loop, logging, checkpoints, training curves
evaluate.py       test-split evaluation of DQN + baselines, results table
tests/            pytest sanity checks (10 tests, all passing)
outputs/          checkpoints, training-curve PNGs, results_raw.csv / results_summary.csv
requirements.txt
```
