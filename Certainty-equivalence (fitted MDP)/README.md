# Certainty-Equivalence (Fitted-MDP) Evaluation — Finance Trading Example

This folder implements a worked example of **certainty-equivalence (fitted MDP)
evaluation**, applied to a single-instrument trading problem, as discussed in
`RL.md` (Lecture 4 companion notes, CS234 Lecture 3).

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

## 5. Files

| File | Purpose |
|---|---|
| `indicators.py` | RSI and Parabolic SAR calculation from OHLCV |
| `trading_env.py` | `TradingEnv` — implements the full MDP transition/reward/constraint logic above |
| `discretize.py` | Maps continuous state/action to discrete bucket IDs for the tabular fitted MDP |
| `behavior_policy.py` | RSI/PSAR rule-based policy used to generate the offline dataset |
| `fitted_mdp.py` | `FittedMDP` — fits `P(s'|s,a)`/`R(s,a)` from counts; runs value iteration and fixed-policy evaluation |
| `demo.py` | End-to-end script: synthetic data → offline dataset → fit → evaluate |

## 6. Running the Demo

```bash
cd finance_mdp
python demo.py
```

Prints: state-action coverage, the fitted MDP's optimal value function and
greedy policy, and the certainty-equivalence evaluation of the (approximate)
behavior policy.

## 7. Suggested Next Steps

- Replace synthetic OHLCV with real historical data.
- Replace the synthetic behavior policy with actual logged trade decisions,
  if available.
- Switch `value_iteration`/`evaluate_policy` to finite-horizon (5-step)
  backward DP to match the episodic spec exactly.
- Consider a regression-based (non-tabular) transition/reward model if finer
  state resolution is needed, since the current tabular approach discards
  continuous detail.
