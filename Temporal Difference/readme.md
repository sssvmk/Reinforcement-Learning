# TD(0) Value Estimation of Trading States (PSAR + RSI on OHLCV)

`td0_trading.py` uses **Temporal-Difference TD(0)** to learn the value of each trading state from OHLCV price
data, using two indicators: **Parabolic SAR (PSAR)** and **RSI**.

> This folder is separate from the parent `readme.md`, which documents the Value Iteration inventory project.

## 1. Functionality

1. Loads OHLCV data (CSV or `.csv.gz`) and sorts by the date column if present.
2. Computes RSI (Wilder, 14) and Parabolic SAR (step 0.02, max 0.2).
3. Encodes each bar as a discrete state and applies a fixed trading rule (the policy).
4. Runs TD(0) over the first 70% of bars (several epochs) to learn `V(state)`.
5. Validates on the remaining 30%: compares `V(state)` with the realised discounted return.
6. Prints the best and worst states and the value of the current (latest) state; writes all values to `td0_state_values.csv`.

## 2. Input data set

`sample_ohlcv.csv` (included): 1,500 business days of **synthetic** OHLCV data (random walk with mild momentum),
for demonstration only.

| Column | Required | Description |
|---|---|---|
| `date` / `timestamp` / `datetime` / `time` | optional | used only to sort rows |
| `open`, `high`, `low`, `close` | yes | bar prices (case-insensitive names) |
| `volume` | optional | not used by the model |

Replace it with real data: `python td0_trading.py --file your_prices.csv.gz`.

## 3. Algorithm

**State** `s = (RSI bucket, PSAR trend, PSAR distance bucket, position)` = 6 × 2 × 3 × 2 = 72 states
- RSI buckets: <30, 30-40, 40-50, 50-60, 60-70, >70
- PSAR trend: bullish (price above SAR) or bearish
- Distance `|close − SAR| / close`, split in terciles: near / mid / far
- Position: the policy's holding (long or flat)

**Policy (evaluated, not learned):** long when PSAR is bullish and RSI < `--rsi-exit` (70); otherwise flat.

**Reward:** `r_t = position_t × ln(close_{t+1}/close_t) − cost × |position_t − position_{t−1}|` (cost 0.0005 by default).

**TD(0) update:**

```
δ = r_t + γ · V(s_{t+1}) − V(s_t)
V(s_t) ← V(s_t) + α · δ
```

`α` = 0.05, `γ` = 0.97, 20 epochs by default. TD(0) learns from every bar by bootstrapping on the next state's value,
so it needs no waiting for the trade to end (unlike Monte Carlo).

## 4. Outcome

- Console: mean |TD error| per epoch, out-of-sample correlation, top 5 and bottom 5 states, current state value.
- `td0_state_values.csv`: columns `state`, `V`, `visits`, sorted by value.

Interpretation: `V(s)` is the expected discounted sum of future (net) log returns when the rule is followed from state `s`.
Positive means the rule tends to earn from there; negative means it tends to lose.

Result on `sample_ohlcv.csv` (10 epochs): out-of-sample correlation **0.022**, i.e. essentially no predictive
signal. That is expected on near-random synthetic data. Evaluate real data before drawing conclusions.

## 5. How to run

```
pip install numpy pandas
python td0_trading.py --demo                                   # synthetic data
python td0_trading.py --file sample_ohlcv.csv                  # included data set
python td0_trading.py --file prices.csv.gz --alpha 0.05 --gamma 0.97 --epochs 20 --cost 0.0005
```

Options: `--rsi-n`, `--rsi-exit`, `--psar-step`, `--psar-max`, `--train-frac`, `--out`, `--lam`, `--view`.

### TD(λ) option

`--lam` (default 0 = TD(0)) turns on TD(λ):

```
python td0_trading.py --file sample_ohlcv.csv --lam 0.9                 # backward view (eligibility traces, online)
python td0_trading.py --file sample_ohlcv.csv --lam 0.9 --view forward  # forward view (λ-return, offline)
```

- **Backward view:** `e(s) ← γλ·e(s) + 1(S_t = s)`, then `V(s) ← V(s) + α·δ_t·e(s)` for all states, so each TD error also credits recently visited states.
- **Forward view:** `G_t = r_t + γ[(1−λ)·V(s_{t+1}) + λ·G_{t+1}]`, then `V(s_t) ← V(s_t) + α(G_t − V(s_t))`.
- `--lam 0` reproduces TD(0) exactly. On `sample_ohlcv.csv` (10 epochs) the out-of-sample correlation was 0.022 for λ=0, 0.049 for λ=0.9 backward and 0.047 for λ=0.9 forward. All are near zero because the sample data is random.

## 6. Limitations

- TD(0) **evaluates** a fixed policy; it does not find a better one (use SARSA or Q-learning for control).
- Coarse tabular states: many are rarely visited, so their values are noisy (see `visits`).
- Buckets use full-sample quantiles for PSAR distance (a small look-ahead); fix this for live trading.
- No slippage, position sizing, or risk limits. Research tool only, not trading advice.

## 7. Files

| File | Purpose |
|---|---|
| `td0_trading.py` | Indicators, state encoding, TD(0), validation |
| `sample_ohlcv.csv` | Synthetic input data set |
| `td0_state_values.csv` | Output: learned state values |
| `readme.md` | This file |
