# RL Value/Control Methods on a Finance Trading MDP

This folder implements five worked examples on the **same trading MDP**
(OHLCV + RSI + PSAR state, 5,000 shares / $100,000 cash / $300 avg-entry
starting account), as discussed in `RL.md`:

1. **Certainty-equivalence (fitted-MDP) evaluation** (Lecture 4 companion,
   CS234 Lecture 3) — `fitted_mdp.py` / `demo.py`
2. **GLIE Monte Carlo Control** (Lecture 5) — `glie_mc_control.py` / `demo_glie.py`
3. **SARSA(0)** (Lecture 5, on-policy TD control) — `sarsa.py` / `demo_sarsa.py`
4. **SARSA(λ)** (Lecture 5, eligibility-trace generalization) — `sarsa_lambda.py` / `demo_sarsa.py`
5. **Q-learning** (Lecture 5, off-policy TD control) — `q_learning.py` / `demo_q_learning.py`

All five share the same environment (`trading_env.py`) and state/action
discretization (`discretize.py`), making it easy to compare **model-based**
(plan on a fitted model), **full-episode model-free** (MC control),
**step-by-step on-policy** (SARSA / SARSA(λ)), and **step-by-step
off-policy** (Q-learning) approaches to the identical problem.

---

## 1. What is Certainty-Equivalence Evaluation?

Instead of estimating a value function directly from sampled returns (Monte
Carlo) or by bootstrapping one step at a time (TD), certainty-equivalence
evaluation works in two stages:

1. **Fit a model** of the environment from logged experience: a maximum-
   likelihood transition model `P(s'|s,a)` (empirical transition counts) and
   a reward model `R(s,a)` (average observed reward), built from a batch of
   `(s, a, r, s')` tuples.
2. **Plan on the fitted model** using ordinary dynamic programming — policy
   evaluation (for a fixed policy) or value iteration (for control) — exactly
   as if the fitted model were the true environment.

It is called "certainty-equivalence" because, once the model is fit, planning
proceeds *as if* that estimated model were certainly correct, ignoring the
model's own estimation uncertainty.

**Key properties vs. Monte Carlo / TD:**
- **Far more data-efficient**: a single observed transition updates the model
  for *every* state-action pair consistent with it, not just the states
  visited along one trajectory.
- **Exploits the Markov property** — good when the environment really is
  (approximately) Markov; can be misleading if it isn't.
- **Computationally heavier** — requires solving a full (even if small,
  discretized) MDP rather than updating single state values.
- Converges to the value function of the *maximum-likelihood MDP implied by
  the data*, not necessarily the same fixed point Monte Carlo would reach on
  finite data (the classic TD-vs-MC divergence example in `RL.md`).

---

## 2. Applications

Certainty-equivalence evaluation is most useful in **batch/offline settings**
where data is limited/expensive but compute is cheap, and where reusing every
transition across many consistent state-action pairs matters:

- **Finance** (this example): portfolio management, trade execution, market
  making — evaluating a candidate trading policy from a fixed window of
  historical market data without further live exploration.
- **Healthcare**: evaluating treatment policies (e.g., sepsis/ICU management)
  from limited, costly patient trajectories.
- **Education**: evaluating tutoring/content-selection policies from small
  student-interaction logs.
- **Robotics**: fitting a dynamics/reward model from a limited number of
  expensive physical trials, then planning on the model instead of costly
  re-collection.
- **Any offline RL problem** with a small/discretizable state-action space,
  where a tabular (or simple regression) model is cheap to fit.

**Main risk in all of these**: the fitted model is only as good as the data
coverage. In finance particularly, market regimes shift (bull/bear/volatile),
so a model fit on one period can be badly wrong out-of-sample — mitigated in
practice by rolling re-fits or adding pessimism (à la CQL/MOPO, see `RL.md`).

---

## 3. The Trading MDP Spec Implemented Here

Built up interactively to a complete, implementable specification:

### State `s`
| Component | Source |
|---|---|
| OHLCV | historical/synthetic market data |
| RSI | `indicators.compute_rsi` |
| PSAR (Parabolic SAR) | `indicators.compute_psar` |
| `shares_held` | account (starts at **5,000**) |
| `cash_balance` | account (starts at **$100,000**) |
| `avg_entry_price` | account (starts at **$300**) |
| *(derived)* `unrealized_PnL`, `portfolio_value` | computed from the above |

### Action `a`
Signed, continuous order size: positive = buy, negative = sell/short.
Action granularity is **flexible** (not fixed-lot), capped each step by the
cash and margin constraints below.

### Constraints
- **Cash floor**: cash balance can never go negative — buy size is capped by
  available cash (including transaction cost).
- **Short selling allowed**, subject to:
  - **50% margin requirement**: collateral (cash) must be ≥ 50% of the open
    short position's market value.
  - **Forced partial cover**: if a price move or fee accrual drops collateral
    below 50%, the environment automatically buys back just enough shares to
    restore the 50% ratio (see `trading_env._maybe_force_cover`).

### Costs and accruals (applied every step)
| Item | Rate | Mechanism |
|---|---|---|
| Transaction cost | **0.05%** of order value | charged on every buy/sell/cover |
| Cash interest | **10% annualized**, continuous compounding | `cash *= exp(0.10 × Δt)` |
| Short borrow/lending fee | **0.5% annualized** of short value | accrued continuously while short |

`Δt = 1/252` per trading day.

### Average entry price convention
- Same-direction trade (adding to a long or a short): standard **weighted
  average** of old and new entry prices.
- **Flip through zero** (long→short or short→long): `avg_entry_price` is
  **reset to the PSAR value** at the moment of the flip, rather than the
  trade's fill price — ties the new cost basis to the trend-reversal signal
  already present in the state.
- Reducing a position (without crossing zero) leaves `avg_entry_price`
  unchanged (standard convention).

### Reward
`r_t = portfolio_value_{t} − portfolio_value_{t-1}` (net of all costs/fees/interest).

### Episode
Rolling **5-trading-day** window (`EPISODE_LENGTH = 5` in `trading_env.py`).

---

## 4. Assumptions and Simplifications

Explicitly called out so they can be revisited before any real use:

1. **Single instrument** — no cross-asset portfolio effects, correlations, or
   diversification modeled.
2. **No slippage/market-impact model beyond the flat 0.05% cost** — real
   execution costs often scale with order size relative to volume; this is
   ignored here.
3. **Behavior policy is synthetic**, not a real trade log: `behavior_policy.py`
   implements a simple RSI/PSAR rule (with 15% random exploration noise) to
   generate a plausible `(s,a,r,s')` dataset, since no actual historical
   trading decisions were supplied. Replace this with real logged trades for
   a genuine offline-RL application.
4. **State discretization is coarse** (`discretize.py`): 3 RSI buckets ×
   2 trend buckets × 3 position buckets = 18 states; 5 action-size buckets.
   This keeps the tabular fitted-MDP tractable but discards information
   (e.g., exact RSI value, exact position size) — a real implementation would
   likely need finer bins or a regression-based (non-tabular) transition/
   reward model.
5. **Infinite-horizon value iteration (γ=0.99) vs. 5-day episodic reality**:
   `fitted_mdp.value_iteration` currently treats the MDP as continuing
   indefinitely, which inflates value-function magnitudes relative to the
   true 5-day episodic process. A finite-horizon backward DP (5 value
   functions, one per remaining day) would be the faithful fix — not yet
   implemented.
6. **Synthetic OHLCV data** (`demo.py: make_synthetic_ohlcv`) — a geometric
   random walk, used only to make the pipeline runnable end-to-end. Swap in
   real historical OHLCV for any meaningful evaluation.
7. **No bankruptcy/liquidation-failure handling**: if a price shock is severe
   enough that even fully covering a short can't restore solvency, this is
   not specially handled (flagged as a known edge case, not implemented).
8. **Data coverage was sparse** in the demo run (23 of 90 state-action cells
   visited across 300 episodes) — unvisited cells default to a self-loop with
   zero reward, which is a conservative but crude fallback; more
   episodes/exploration would be needed for a trustworthy fit.

---

## 5. GLIE Monte Carlo Control (`glie_mc_control.py`, `demo_glie.py`)

### 5.1 What it is
Unlike certainty-equivalence, GLIE Monte Carlo (MC) control never fits a
transition/reward model at all. It learns the action-value function `Q(s,a)`
**directly from complete sampled episodes**, using the actual observed
returns, and improves its own behavior policy as it goes (an on-policy,
fully model-free method).

**GLIE = Greedy in the Limit with Infinite Exploration**, the two conditions
needed for this to provably converge to `Q*`:
1. **Infinite exploration**: every `(s,a)` pair is visited infinitely often.
2. **Greedy in the limit**: the policy becomes greedy (`ε → 0`) as the
   episode/visit count → ∞.

A simple schedule satisfying both is a decaying ε per state:
`ε(s) = N0 / (N0 + N(s))`, implemented in `epsilon_for_state`. As a state is
visited more (`N(s) → ∞`), `ε(s) → 0`, while every state still keeps getting
visited (and explored) early on.

### 5.2 Algorithm (first-visit MC control)
For each episode (here: one 5-day `TradingEnv` rollout):
1. **Generate the episode** by following the current ε-greedy policy over
   discretized action buckets (`select_action`), converting each bucket to
   an actual order size via `action_bucket_to_shares` (capped by the
   env's cash/margin limits).
2. **Compute returns**: walking backward through the episode,
   `G_t = r_t + γ·G_{t+1}`.
3. **First-visit update**: for the *first* occurrence of each `(s,a)` pair in
   the episode, do an incremental-mean update:
   ```
   N(s,a) ← N(s,a) + 1
   Q(s,a) ← Q(s,a) + (G_t − Q(s,a)) / N(s,a)
   ```
4. Repeat for many episodes; the greedy policy `argmax_a Q(s,a)` converges to
   `π*` as exploration vanishes.

### 5.3 Assumptions and simplifications (GLIE-specific)
1. **Complete episodes required** — MC control only updates after a full
   5-day episode finishes; it cannot learn mid-episode or from a continuing
   (non-terminating) stream, unlike TD/SARSA/Q-learning.
2. **State-dependent ε schedule** (`N0=100`) is a design choice — a simpler
   global `ε_k = 1/k` (k = total episode count) is the textbook GLIE
   schedule and would also satisfy the convergence conditions, but explores
   less efficiently across a large discrete state space.
3. **Same coarse discretization** as the certainty-equivalence example (18
   states × 5 action buckets) — MC control needs *more* episodes than fitting
   a model to reach the same state-action coverage, since every update
   requires a full episode visiting that pair, rather than reusing one
   transition across all consistent (s,a) pairs.
4. **High variance, slow convergence**: full-episode Monte Carlo returns are
   unbiased but noisy (compounding 5 days of price randomness, trading
   costs, and interest/fee accrual per return), so many more episodes are
   needed than a bootstrapped method (SARSA/Q-learning) would require for
   the same accuracy.
5. **On-policy only**: the policy being learned is the same one generating
   the data — it cannot reuse logged trades from a *different* historical
   behavior policy the way certainty-equivalence's offline dataset can.
6. **Synthetic OHLCV data**, same caveat as before — swap in real historical
   data for any meaningful result.

### 5.4 Observed result (20,000 episodes, synthetic data)
| Metric | Certainty-Equivalence (300 episodes) | GLIE MC Control (20,000 episodes) |
|---|---|---|
| (s,a) pairs covered | 23 / 90 | 54 / 90 |
| Needs a fitted model? | Yes (`P`, `R`) | No — pure sampled returns |
| Data efficiency | High (one transition updates many consistent cells) | Low (needs full episodes visiting a cell) |
| Update timing | After fitting, via DP (batch) | After every full episode (online) |

This illustrates the core model-based vs. model-free trade-off from `RL.md`:
certainty-equivalence reaches usable coverage with far fewer episodes by
exploiting the Markov structure to generalize counts across the state-action
space, while GLIE MC control needs many more episodes but makes no
assumptions about the model at all beyond the Markov property implicit in
using state-action returns.

---

## 6. SARSA(0) and SARSA(λ) (`sarsa.py`, `sarsa_lambda.py`, `demo_sarsa.py`)

### 6.1 What they are
Both are **on-policy TD control** methods: like GLIE MC control, the policy
being learned is the same one generating the data (via ε-greedy over the
current `Q`); unlike GLIE MC control, neither waits for a full episode to
update — they bootstrap from the current `Q` estimate after every single
step.

- **SARSA(0)**: updates `Q(s,a)` toward the TD target
  `r + γ·Q(s′,a′)`, where `a′` is the action the current ε-greedy policy
  actually selects next (hence "SARSA" = State-Action-Reward-State-Action).
- **SARSA(λ)**: generalizes SARSA(0) using **eligibility traces** (the
  efficient backward view of the λ-return) — the one-step TD error is
  propagated backward to every `(s,a)` pair visited earlier in the episode,
  weighted by a trace that decays by `γλ` each step. `λ=0` reduces to plain
  SARSA(0); `λ=1` behaves like episodic Monte Carlo control.

### 6.2 Algorithm

**SARSA(0)** (`sarsa.py`), per step:
```
N(s,a) ← N(s,a) + 1
alpha  ← 1 / N(s,a)                      # Robbins-Monro step size
target ← r                                # if episode ends this step
       or r + γ·Q(s', a')                 # otherwise, a' chosen by the same ε-greedy policy
Q(s,a) ← Q(s,a) + alpha · (target − Q(s,a))
```

**SARSA(λ)** (`sarsa_lambda.py`), per step, with eligibility trace `E`
(reset to 0 at the start of each episode):
```
target  ← r  or  r + γ·Q(s', a')          # same as above
delta   ← target − Q(s,a)
E(s,a)  ← E(s,a) + 1                      # accumulating trace
Q       ← Q + alpha · delta · E           # updates EVERY (s,a) with nonzero trace
E       ← γ·λ·E                           # decay all traces
```

Both use the same ε-greedy policy and GLIE-style decaying ε
(`ε(s) = N0/(N0+N(s))`) as the certainty-equivalence/GLIE-MC examples.
Episodes end without bootstrapping past the terminal step (the TD target is
just `r`, not `r + γ·Q(s′,a′)`), since each 5-day episode resets the account
independently and chaining `Q` across unrelated episode starts would not be
meaningful.

### 6.3 Assumptions and simplifications (SARSA-specific)
1. **SARSA(0) step size** `alpha = 1/N(s,a)` satisfies the Robbins-Monro
   conditions (`Σα=∞`, `Σα²<∞`) that, together with GLIE exploration,
   theoretically guarantee convergence to `Q*` in the tabular setting.
2. **SARSA(λ) uses a constant step size** (`alpha=0.05` by default) instead,
   for a simple vectorized trace update across the whole Q-table each step.
   This is a common practical simplification, but it does **not** strictly
   satisfy Robbins-Monro, so SARSA(λ)'s convergence here is empirical, not
   guaranteed.
3. **Accumulating eligibility traces** (`E(s,a) += 1`) are used, not
   replacing traces — the simpler of the two standard variants, but can
   over-weight states visited repeatedly in a short span.
4. **`λ=0.9`** is a fixed choice in the demo; the right value is
   problem-dependent (Lecture 5/6 note that intermediate λ usually
   outperforms both λ=0 and λ=1, but this isn't tuned here).
5. Same shared caveats as the other methods: coarse 18-state/5-action
   discretization, synthetic OHLCV data, single instrument, flat 0.05%
   transaction cost only (no market-impact model).
6. **Bootstrapped, so less sensitive to the infinite-horizon-vs-episode
   mismatch** than certainty-equivalence's value iteration: because TD
   targets never extrapolate indefinitely (the terminal step uses `r` alone,
   not a bootstrapped tail), SARSA/SARSA(λ) `V(s)` magnitudes stay realistic
   even though `gamma=0.99` is nominally an infinite-horizon discount.

---

## 7. Q-learning (`q_learning.py`, `demo_q_learning.py`)

### 7.1 What it is
Q-learning is **off-policy TD control**: like SARSA(0), it updates `Q(s,a)`
after every single step using a one-step TD target — but the target
bootstraps from the **greedy** action at `s'` (`max_a' Q(s', a')`), not from
whatever action the current ε-greedy *behavior* policy actually picks next.
This means the policy generating the data (ε-greedy, for exploration) and
the policy being evaluated/improved (greedy, implicit in the `max`) can
differ — the defining feature of off-policy learning. Q-learning directly
samples the **Bellman optimality equation**, making it the TD analogue of
value iteration, exactly as SARSA is the TD analogue of (on-policy) policy
iteration's evaluation step.

### 7.2 Algorithm
Per step (`q_learning.py`):
```
N(s,a) ← N(s,a) + 1
alpha  ← 1 / N(s,a)                        # Robbins-Monro step size
target ← r                                  # if episode ends this step
       or r + γ·max_a' Q(s', a')            # otherwise — GREEDY at s', regardless
                                             #   of what the behavior policy picks next
Q(s,a) ← Q(s,a) + alpha · (target − Q(s,a))
```
Action selection (behavior policy) uses the same GLIE-style decaying
ε-greedy (`ε(s) = N0/(N0+N(s))`) as every other method here. As with SARSA,
episodes don't bootstrap past the terminal step (`target = r` alone there),
since each 5-day episode resets the account independently.

### 7.3 Assumptions and simplifications (Q-learning-specific)
1. **Step size** `alpha = 1/N(s,a)` satisfies Robbins-Monro, same as
   SARSA(0); together with GLIE exploration this is the standard tabular
   convergence argument for `Q → Q*`.
2. **Off-policy, so in principle could learn from *any* logged data**
   (e.g., `behavior_policy.py`'s RSI/PSAR rule, random exploration, or old
   policy snapshots) rather than requiring on-policy ε-greedy rollouts like
   SARSA/SARSA(λ)/GLIE MC control — this demo still uses on-policy ε-greedy
   rollouts for a fair side-by-side comparison, but swapping in offline
   data for training would be a natural extension.
3. **Same shared caveats** as the other methods: coarse 18-state/5-action
   discretization, synthetic OHLCV data, single instrument, flat 0.05%
   transaction cost only.
4. **Bootstrapped like SARSA**, so `V(s)` magnitudes stay realistic (no
   infinite-horizon blow-up), unlike certainty-equivalence's DP sweep.
5. **Maximization bias** is a known issue with vanilla Q-learning (the
   `max` operator is a biased estimator of the true max expected value);
   Double Q-learning addresses this but isn't implemented here.

### 7.4 Observed result (20,000 episodes, synthetic data)
Compared directly against SARSA(0) trained on the identical environment and
seed (`demo_q_learning.py`): Q-learning reached 59/90 (s,a) coverage vs.
SARSA(0)'s 55/90, and their **greedy policies differ at 6 of 18 states** —
a concrete, reproducible illustration of on-policy vs. off-policy
divergence (the same phenomenon as the classic Cliff Walking example, where
Q-learning learns the riskier "optimal" path and SARSA learns a safer one
shaped by its own exploration).

---

## 8. Datasets

Both demos run on **synthetic OHLCV data**, generated by
`demo.py: make_synthetic_ohlcv` — a geometric random walk (log-returns
`~N(0.0002, 0.01²)`), with `high`/`low` jittered around `close` and random
integer `volume`. This keeps the pipeline runnable with no external
dependency; it is **not** meant to reflect real market statistics.

| Column | Produced by | Required by |
|---|---|---|
| `open`, `high`, `low`, `close`, `volume` | `make_synthetic_ohlcv` (or your own historical loader) | `TradingEnv`, `indicators.py` |
| `rsi` | `indicators.compute_rsi(close)` | `discretize.rsi_bucket` |
| `psar` | `indicators.compute_psar(high, low, close)` | `discretize.trend_bucket`, PSAR-reset entry-price rule |

**To use real data**: build a `pandas.DataFrame` with at least
`open/high/low/close/volume`, then run:
```python
df["rsi"] = compute_rsi(df["close"])
df["psar"] = compute_psar(df["high"], df["low"], df["close"])
env = TradingEnv(df, initial_shares=5000.0, initial_cash=100_000.0, initial_avg_entry=300.0)
```
and pass `env` to either `collect_offline_dataset` (certainty-equivalence) or
`GLIEMonteCarloControl.train` (GLIE MC control) in place of the synthetic
frame. No separate dataset file is checked in — generate or supply one at
run time.

---

## 9. Requirements / Specs per Algorithm

| Requirement | Certainty-Equivalence (`demo.py`) | GLIE MC Control (`demo_glie.py`) | SARSA(0) (`sarsa.py`) | SARSA(λ) (`sarsa_lambda.py`) | Q-learning (`q_learning.py`) |
|---|---|---|---|---|---|
| Environment | `TradingEnv` (shared) | `TradingEnv` (shared) | `TradingEnv` (shared) | `TradingEnv` (shared) | `TradingEnv` (shared) |
| State/action discretization | `discretize.py` (18 states × 5 action buckets) | same | same | same | same |
| Data source | Logged/simulated `(s,a,r,s')` via `behavior_policy.rsi_psar_policy` | Live episode rollouts under evolving ε-greedy policy | Live step-by-step rollout under evolving ε-greedy policy | Live step-by-step rollout under evolving ε-greedy policy | Live step-by-step rollout under evolving ε-greedy policy (any behavior policy would work, off-policy) |
| On/off-policy | — (model-based planning) | On-policy | On-policy | On-policy | **Off-policy** |
| Key hyperparameters | `gamma` (0.99), `n_episodes` for data collection (300) | `gamma` (0.99), `n0` (100.0), `n_episodes` (20,000), `seed` | `gamma` (0.99), `n0` (100.0), `n_episodes` (20,000), `seed` | `gamma` (0.99), `lam` (0.9), `alpha` (0.05, constant), `n0` (100.0), `n_episodes` (20,000), `seed` | `gamma` (0.99), `n0` (100.0), `n_episodes` (20,000), `seed` |
| Core outputs | `P(s'|s,a)`, `R(s,a)`, `V*`, `Q*`, greedy policy, fixed-policy evaluation | `Q(s,a)`, `N(s,a)`, `V(s)`, greedy policy, final ε | `Q(s,a)`, `N(s,a)`, `V(s)`, greedy policy | `Q(s,a)`, `V(s)`, greedy policy, (diagnostic visit counts) | `Q(s,a)`, `N(s,a)`, `V(s)`, greedy policy |
| Update timing | Batch (fit once, DP sweep) | After each full episode | After every single step | After every single step, propagated to all eligible `(s,a)` pairs | After every single step |
| Convergence requirement | Enough (s,a) coverage for a reliable MLE fit | GLIE: infinite exploration + ε→0 | GLIE + Robbins-Monro step size (`alpha=1/N(s,a)`) | GLIE; step size is constant here, so convergence is empirical only | GLIE + Robbins-Monro step size (`alpha=1/N(s,a)`) |
| Compute pattern | One-time fit + DP sweep (batch) | Incremental, full-episode granularity | Incremental, per-step granularity | Incremental, per-step, updates multiple `(s,a)` per step via traces | Incremental, per-step granularity |

---

## 10. Results (synthetic data, as run)

### Certainty-Equivalence (`python demo.py`, 300 episodes to build the dataset)
```
Visited (s,a) pairs: 21 / 90

Fitted-MDP optimal value function V*(s) by discrete state id:
[     0.        0.   595757.17      0.        0.   662143.27      0.
      0.   599610.73 851374.18      0.   639863.37      0.        0.
 307349.67 441088.65      0.   620722.55]

Greedy action bucket per state (index into [-1,-0.5,0,0.5,1] x max size):
[0 0 2 0 0 4 0 0 2 2 0 1 0 0 0 1 0 0]
```
The evaluated (approximate) behavior policy's `V_π(s)` comes out strongly
**negative** for several states in this run — an artifact of the known
infinite-horizon-vs-5-day-episode mismatch (see Assumptions §4.5): with
γ=0.99 the DP sweep extrapolates the fitted one-step model far beyond the
true 5-day horizon, so these magnitudes (both the very large positive `V*`
and the large negative `V_π`) should be read as illustrative of the pipeline
mechanics, not literal dollar values. Re-running fixes `V*`/`V_π` on a
*different* random dataset, so the state ids with non-zero values will also
shift from run to run (sparse coverage: 21/90 here).

### GLIE MC Control (`python demo_glie.py`, 20,000 training episodes)
```
Visited (s,a) pairs: 53 / 90

Learned V(s) = max_a Q(s,a):
[12698.68     0.    -316.11     0.       0.   33610.43  3343.58     0.
   691.83   629.82     0.     391.82 32252.37     0.   -2612.25   949.56
     0.    2257.8 ]

Greedy policy (action bucket index per state, [-1,-0.5,0,0.5,1] x max size):
[1 0 3 0 0 0 0 0 2 2 0 1 0 0 0 1 0 1]
```
GLIE's learned `Q`/`V` magnitudes are far smaller and more plausible than the
fitted-MDP's, because MC returns are bounded by actual 5-day episode sums
(no infinite-horizon extrapolation) — direct evidence of the discount/horizon
caveat above. Coverage (53/90) is better than certainty-equivalence's
21/90 despite far more episodes needed to get there, confirming the
data-efficiency trade-off in §5.4.

### SARSA(0) and SARSA(λ) (`python demo_sarsa.py`, 20,000 training episodes each)
```
=== SARSA(0) ===
Visited (s,a) pairs: 53 / 90
V(s) = max_a Q(s,a):
[ -720.17     0.    4447.93     0.       0.   32675.42  4391.44     0.
  4214.67  4819.85     0.    3962.19 32252.37     0.    3910.85  2409.38
     0.    2956.01]
Greedy policy:
[4 0 2 0 0 4 0 0 1 0 0 0 0 0 0 2 0 1]

=== SARSA(lambda=0.9) ===
Visited (s,a) pairs: 56 / 90
V(s) = max_a Q(s,a):
[ 2.096000e+01  0.000000e+00  1.466163e+04  3.067550e+03  0.000000e+00
  2.117107e+04  9.459700e+02  0.000000e+00 -4.827880e+03  5.243000e+02
  0.000000e+00  5.166370e+03  1.477300e+02  0.000000e+00 -2.344230e+03
  1.114680e+03  0.000000e+00  5.604790e+03]
Greedy policy:
[2 0 4 4 0 1 2 0 1 2 0 1 2 0 4 3 0 2]
```
Both SARSA variants reach similar (s,a) coverage to GLIE MC control (53–56
of 90) with the same 20,000 episodes, but update `Q` every step rather than
only at episode end — in principle more sample-efficient per episode since
useful information propagates immediately rather than waiting for the
episode's return. SARSA(λ) visits a few more (s,a) pairs than SARSA(0) here
because its eligibility-trace updates touch every pair visited earlier in
the same episode, not just the current one, giving each episode more
"effective" learning signal per step. `V(s)` magnitudes for both stay in a
realistic range (no infinite-horizon blow-up), consistent with §6.3's point
that bootstrapped TD targets never extrapolate past the terminal step.

### Q-learning vs. SARSA(0) (`python demo_q_learning.py`, 20,000 training episodes each, same seed)
```
=== Q-learning (off-policy) ===
Visited (s,a) pairs: 59 / 90
V(s) = max_a Q(s,a):
[ 4641.34     0.    8132.65 26976.75     0.   32382.14  1559.85     0.
  5115.55  1012.07     0.    4225.77 32252.37     0.   -4809.92  3139.04
     0.    6859.03]
Greedy policy:
[3 0 0 4 0 4 3 0 1 3 0 2 0 0 0 3 0 2]

=== SARSA(0) (on-policy, for comparison) ===
Visited (s,a) pairs: 55 / 90
V(s) = max_a Q(s,a):
[  579.74     0.    1627.19 10335.34     0.   31607.91  1358.57     0.
  2631.72  1859.95     0.    2661.84 32252.37     0.   -8596.18  2166.49
     0.    3362.56]
Greedy policy:
[3 0 1 3 0 4 3 0 2 3 0 3 0 0 0 4 0 3]

States where Q-learning and SARSA(0) greedy policies differ: [2, 3, 8, 11, 15, 17]
```
Q-learning reaches slightly broader coverage (59/90 vs. 55/90) under
identical conditions, and its **greedy policy disagrees with SARSA(0)'s at
6 of 18 states** — direct, reproducible evidence of the on-policy/off-policy
distinction from §7.1: Q-learning's target always bootstraps from the
*greedy* action at `s'`, independent of the ε-greedy exploration actually
used to generate the trajectory, while SARSA(0)'s target reflects whatever
(possibly exploratory) action its own policy picks next. `V(s)` magnitudes
for Q-learning stay in the same realistic range as the other TD methods (no
infinite-horizon blow-up).

**Overall takeaway across all five methods**: none of these runs should be
treated as a real trading signal — all use synthetic random-walk data and a
coarse 18-state discretization. They demonstrate each algorithm's mechanics
and trade-offs (model-based vs. model-free, full-episode vs. step-by-step
bootstrapping, on-policy vs. off-policy), not a validated strategy.

---

## 11. Files

| File | Purpose |
|---|---|
| `indicators.py` | RSI and Parabolic SAR calculation from OHLCV |
| `trading_env.py` | `TradingEnv` — implements the full MDP transition/reward/constraint logic above |
| `discretize.py` | Maps continuous state/action to discrete bucket IDs for both tabular methods |
| `behavior_policy.py` | RSI/PSAR rule-based policy used to generate the certainty-equivalence offline dataset |
| `fitted_mdp.py` | `FittedMDP` — fits `P(s'|s,a)`/`R(s,a)` from counts; runs value iteration and fixed-policy evaluation |
| `demo.py` | Certainty-equivalence end-to-end script: synthetic data → offline dataset → fit → evaluate |
| `glie_mc_control.py` | `GLIEMonteCarloControl` — first-visit MC control with decaying ε-greedy exploration |
| `demo_glie.py` | GLIE MC control end-to-end script: synthetic data → train from episodes → report Q/V/policy |
| `sarsa.py` | `SARSA` — on-policy TD(0) control with `1/N(s,a)` step size |
| `sarsa_lambda.py` | `SARSALambda` — on-policy TD(λ) control via accumulating eligibility traces |
| `demo_sarsa.py` | SARSA/SARSA(λ) end-to-end script: synthetic data → train from episodes → report Q/V/policy for both |
| `q_learning.py` | `QLearning` — off-policy TD(0) control bootstrapping from `max_a' Q(s',a')` |
| `demo_q_learning.py` | Q-learning end-to-end script: trains Q-learning and SARSA(0) side-by-side and reports where their greedy policies diverge |

## 12. Running the Demos

```bash
cd trading-mdp-value-methods
python demo.py            # certainty-equivalence: fit model, then plan via DP
python demo_glie.py       # GLIE MC control: learn Q directly from sampled episodes
python demo_sarsa.py      # SARSA(0) and SARSA(lambda): step-by-step TD control
python demo_q_learning.py # Q-learning vs. SARSA(0): off-policy vs. on-policy TD control
```

`demo.py` prints: state-action coverage, the fitted MDP's optimal value
function and greedy policy, and the certainty-equivalence evaluation of the
(approximate) behavior policy.

`demo_glie.py` prints: state-action visit counts, the learned `V(s) =
max_a Q(s,a)`, the greedy policy, and the final per-state ε values (showing
the GLIE decay in action).

`demo_sarsa.py` prints, for both SARSA(0) and SARSA(λ): state-action
coverage, `V(s) = max_a Q(s,a)`, and the greedy policy.

`demo_q_learning.py` prints, for both Q-learning and SARSA(0) (trained on
the identical environment/seed): state-action coverage, `V(s) = max_a
Q(s,a)`, the greedy policy, and the list of states where the two greedy
policies disagree.

## 13. Suggested Next Steps

- Replace synthetic OHLCV with real historical data (shared by all demos).
- Replace the synthetic behavior policy with actual logged trade decisions,
  if available (for the certainty-equivalence path).
- Switch `value_iteration`/`evaluate_policy` to finite-horizon (5-step)
  backward DP to match the episodic spec exactly.
- Consider a regression-based (non-tabular) transition/reward model, or a
  function-approximation Q (e.g., linear/tile-coded, per Lecture 6), if finer
  state resolution is needed — all tabular methods here discard continuous
  detail (exact RSI value, exact position size).
- Train Q-learning on **offline/logged data** (e.g., `behavior_policy.py`'s
  RSI/PSAR rule, or real historical trade logs) instead of on-policy
  ε-greedy rollouts — since Q-learning is off-policy this should work
  without modification, unlike SARSA/SARSA(λ)/GLIE MC control.
- Implement Double Q-learning to address vanilla Q-learning's maximization
  bias (§7.3).
- Tune/compare `λ` for SARSA(λ) (e.g., a sweep over 0, 0.3, 0.5, 0.7, 0.9, 1)
  rather than the single fixed value used in the demo.
