# Monte Carlo Risk Assessment

Monte Carlo risk assessment of an asset from OHLCV price data, with background on Monte Carlo (MC) learning in RL.

## 1. Monte Carlo learning (RL)

MC learning estimates the value of a state or policy by averaging the **complete returns** of sampled episodes.

- Needs no model of the environment, only samples or a simulator.
- Unbiased, but high variance.
- Needs episodes to end before updating.
- Does not bootstrap, so it does not rely on the Markov property.

Compare with TD(0), which updates every step using a bootstrapped estimate (lower variance, biased), and TD(λ), which sits in between.

Typical applications: games (Blackjack, MCTS rollouts), policy evaluation in simulators, off-policy evaluation from logged episodes, finance, healthcare and operations.

### Simulation vs learning

| | Monte Carlo **simulation** | Monte Carlo **learning** |
|---|---|---|
| Goal | Estimate a distribution of outcomes (e.g. risk) | Learn value or policy from episode returns |
| Used here | Yes (`mc_risk.py`) | Optional extension (learn a hedging or sizing policy on simulated paths) |

## 2. Risk assessment application

`mc_risk.py` simulates many possible future price paths and measures the distribution of losses.

### Input
A CSV (or gzip-compressed `.csv.gz`) for a single asset with columns:
`Date, Open, High, Low, Close, Volume` (case-insensitive).

### Usage
```
python mc_risk.py asset_ohlcv.csv.gz --horizon 10 --sims 20000
python mc_risk.py --demo          # synthetic data
```
Requires `numpy` and `pandas`.

| Argument | Meaning | Default |
|---|---|---|
| `csv` | Path to the OHLCV file (`.csv` or `.csv.gz`) | demo data |
| `--horizon` | Trading days to simulate ahead (1 = 1 day, 21 ≈ 1 month, 252 ≈ 1 year) | 10 |
| `--sims` | Number of simulated paths | 20000 |
| `--demo` | Use synthetic data | off |

### Method
1. Compute daily log returns: `r_t = ln(C_t / C_{t-1})`.
2. Simulate paths with two methods:
   - **GBM** (Geometric Brownian Motion): `S_{t+1} = S_t · exp((μ − σ²/2) + σ·Z)`, `Z ~ N(0,1)`, with σ from the Garman–Klass OHLC estimator. Assumes normal returns, so it tends to understate tail risk.
   - **Bootstrap**: resamples historical returns, which preserves fat tails.
3. Compute metrics over the simulated paths.

### Output metrics
| Metric | Meaning |
|---|---|
| VaR 95% | Loss not exceeded in 95% of scenarios over the horizon |
| CVaR 95% | Average loss in the worst 5% of scenarios (Expected Shortfall) |
| P(loss), P(loss > 10%) | Probability of ending below the start, or more than 10% below |
| Expected return | Mean return over the horizon |
| Mean / worst max drawdown | Largest peak-to-trough fall along each path |

### Limitations
- Assumes the past resembles the future; history lacking crises understates risk.
- GBM assumes constant volatility and independent returns.
- Single asset only. Volume is not used.
- Not financial advice.

### Possible extensions
Multi-asset portfolios with correlations (Cholesky), GARCH volatility, Student-t or jump-diffusion shocks, and an RL policy (position sizing or hedging) trained on the simulated paths.
