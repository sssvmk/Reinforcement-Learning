# On-policy Monte Carlo Control (ε-greedy / GLIE) for Daily Trading with RSI + PSAR

`mc_control_trading.py` learns a **long / flat** trading policy from daily OHLCV data using
**first-visit on-policy Monte Carlo control** with an **ε-greedy, GLIE** exploration schedule.

## 1. Functionality

1. Loads daily OHLCV (CSV or `.csv.gz`), sorted by date if a date column exists.
2. Computes RSI (Wilder, 14) and Parabolic SAR (0.02 / 0.2) and encodes each day as a discrete state.
3. Generates many short **episodes** (random start in the training period, fixed length in days).
4. Updates `Q(s,a)` from full-episode returns, improving the ε-greedy policy as it goes.
5. Back-tests the learned greedy policy on the training and the held-out test period against buy-and-hold.
6. Saves the Q table and prints the recommended action for the latest bar.

## 2. Input data set

`sample_ohlcv.csv` (included): 1,500 business days of **synthetic** OHLCV (random walk with mild momentum), for demonstration only.

| Column | Required | Description |
|---|---|---|
| `date` / `timestamp` / `datetime` / `time` | optional | used only to sort rows |
| `open`, `high`, `low`, `close` | yes | daily prices (names are case-insensitive) |
| `volume` | optional | not used |

Use your own data: `python mc_control_trading.py --file prices.csv.gz`.

## 3. Algorithm

**Episode:** `--horizon` trading days (default 20), starting flat on a random training day. Episodes must end for MC.

**State** (72 = 6 × 2 × 3 × 2): RSI bucket (<30, 30-40, 40-50, 50-60, 60-70, >70) × PSAR trend (bull/bear) ×
PSAR distance `|close − SAR| / close` tercile (cut points from the training period only) × current position.

**Actions:** 0 = flat, 1 = long for the next day.

**Reward:** `action × next-day log return − cost × |action − previous position|` (cost 0.0005).

**Return:** `G_t = r_t + γ·G_{t+1}` to the end of the episode (γ = 1 by default).

**Policy and GLIE:** ε-greedy. With probability ε take a random action, otherwise `argmax_a Q(s,a)`.
ε decays as `ε_k = 1/k` (k = episode number), so exploration vanishes while every state-action pair keeps
being visited, which makes the greedy policy converge to the optimal one. The policy that acts is the same one
being improved (on-policy).

**Update (first-visit):** for the first occurrence of each `(s,a)` in an episode:

```
N(s,a) ← N(s,a) + 1
Q(s,a) ← Q(s,a) + (1 / N(s,a)) · (G − Q(s,a))
```

## 4. Outcome

- Console: strategy vs buy-and-hold on the train and test periods (return, Sharpe, max drawdown) and the action for the latest bar.
- `mc_q_values.csv`: `state`, `Q_flat`, `Q_long`, `N_flat`, `N_long`, `best_action` for visited states.

Result on `sample_ohlcv.csv` (30,000 episodes): test +14.6% (Sharpe 0.90) vs buy-and-hold +11.5% (Sharpe 0.37).
The data is synthetic and nearly random, so this gap is probably chance, not a real edge. Always check
out-of-sample results on real data, across several assets and periods.

## 5. How to run

```
pip install numpy pandas
python mc_control_trading.py --demo
python mc_control_trading.py --file sample_ohlcv.csv
python mc_control_trading.py --file prices.csv.gz --episodes 50000 --horizon 20 --train-frac 0.7 --cost 0.0005
```

Options: `--gamma`, `--no-glie --epsilon 0.1` (fixed ε), `--min-eps`, `--rsi-n`, `--seed`, `--out`.

## 6. Limitations

- Daily data gives few independent episodes; Q-values are noisy. Use many years and several assets.
- Historical replay: actions don't affect prices, so there is no market impact or slippage.
- Only two actions and coarse states; no position sizing or risk limits.
- MC has high variance and waits for each episode to finish (TD methods such as SARSA update every step).
- Research tool, not trading advice.

## 7. Files

| File | Purpose |
|---|---|
| `mc_control_trading.py` | Indicators, states, MC control, back-test |
| `sample_ohlcv.csv` | Synthetic input data set |
| `mc_q_values.csv` | Output: learned Q table |
| `readme.md` | This file |
