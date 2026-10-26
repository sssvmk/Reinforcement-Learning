# Finite-difference policy gradient (FDPG): M5 inventory + trading replay

**Status: trading and inventory DONE (quick + full).**
Nothing was invented for inventory; setting `inventory.enabled: auto` (default) activates it once the file appears.

## Data
- Trading: `C:/Users/msk80743/OneDrive - GSK/Muni/project/India/TECHM/TECHM_extend.csv.gzip` (4677 rows, 2007-09-17..2026-10-23, no duplicate dates).
- M5: `C:\Users\msk80743\OneDrive - GSK\m5-forecasting-accuracy` has `calendar.csv`, `sales_train_validation.csv`, `sales_train_evaluation.csv`; **`sell_prices.csv` missing**.
- Paths live in `config.yaml` (override with env `TRADING_FILES`, `M5_DIR`).

### Audit findings and deviation (important)
- RSI = Wilder period 14 on adjusted `close` (MAE 7e-10).
- SAR best match step 0.02 / max 0.3 (84% of rows within 0.5%): **only partly verified** (path-dependent start).
- **Price-scale audit FAILED**: `low<=close<=high` on ~0% of rows 2007-2024. `close` is split/dividend-adjusted while open/high/low are not (ratio drifts smoothly 0.65 -> 0.97).
- Under the strict rule the file would be excluded. **Documented deviation**: returns use adjusted `close` (`trading.allow_adjusted_close: true`); SAR stays on the unadjusted OHLC scale. Results are therefore a replay of close-to-close returns, not an executable-price backtest.
- The file contains dates after today's real calendar (to 2026-10); treated as provided.

## Design
- Trading MDP: position in [-1,1] chosen from features (rsi, trend, osar, ret5, volz); reward = pos*next-day return - 5 bps*|delta pos|; 250-day windows starting flat; 60/20/20 chronological train/val/test; test never used for fitting/tuning. Cost is an assumption.
- Inventory (implemented and unit-tested on synthetic data only): lost-sales, lead time 2, policies I1-I3, base-stock/(s,S)/cover benchmarks.
- Policies: T1 (7 params, standardized features), T2 (3 params). Objective normalized by buy-and-hold window std.
- Estimators: forward, central, SPSA, regression, Gaussian smoothing; CRN on/off; norm clipping, optional Adam; (eps, a) tuned on validation with central estimator and reused for all. Budget counts training rollouts only.
- Baselines: grid (n<=2, skipped here), random search, Nelder-Mead, buy-and-hold, flat, tuned RSI rule, SAR rule.

## Run
```
pip install -r requirements.txt
python -m pytest tests -q            # 34 tests
python audit_data.py
python run_fdpg.py --domain all --quick     # or --full; --domain trading|inventory also works
                                            # (inventory needs sell_prices.csv in the M5 folder)
python oracle_benchmarks.py                 # optional: test-tuned upper-bound benchmarks (after the full run)
python evaluate_fdpg.py --quick|--full      # algorithm performance
python evaluate_business.py --quick|--full  # business metrics
```
Outputs: `outputs/runs`, `outputs/algo_eval/{quick,full}/SUMMARY.md`, `outputs/business_eval/{quick,full}/SUMMARY.md` (+CSVs, plots). `analysis_plan.yaml` is frozen before full runs (hash recorded).

## Results (TECHM, full mode, 10 seeds, budget 40k rollouts)
Test period 2023-01-31..2026-10-22, cost 5 bps:

| strategy | cum return | Sharpe | max DD |
|---|---|---|---|
| FD-T1 (forward, best on val) | 0.78 ± 0.08 | 0.78 | -0.26 |
| FD-T2 (regression) | 0.99 | 0.86 | -0.29 |
| buy-and-hold | 1.00 | 0.85 | -0.29 |
| tuned RSI rule | 1.00 (degenerated to buy-and-hold) | 0.85 | -0.29 |
| SAR rule | 0.49 | 0.69 | -0.26 |

Walk-forward pooled (3 folds): FD-T1 0.85, FD-T2 1.06, B&H 1.08, RSI 1.09.

Algorithm findings (see `outputs/algo_eval/full/SUMMARY.md`):
- **CRN matters for gradient quality**: at the start point, forward-difference variance 0.012 with CRN vs 0.897 without (cosine 0.94 vs 0.30); central 0.011 vs 0.225.
- CRN estimators converge reliably (100% of seeds) but slowly; central/forward/regression/Gaussian/SPSA are statistically similar in final validation return (~0.74-0.77).
- Nelder-Mead is clearly worse (0.27). Random search is competitive (0.77), and the no-CRN variants reached equal or higher final validation return in this setting with high seed variance. This is not evidence that CRN is unnecessary; it mainly shows the landscape is easy and noisy-gradient methods still progress.
- "rollouts_to_X%" values above the budget are censored at 2x budget.

## Inventory results (M5, full mode, 10 seeds; `outputs/business_eval/full/SUMMARY.md`)
`sell_prices.csv` was later supplied, so inventory now runs on 3 series: FOODS_3_124_CA_3 (primary), FOODS_2_128_WI_2 (intermittent), HOBBIES_1_371_WI_1 (seasonal), each with policies I1-I3.
Primary series, test profit per window (FD estimator best on validation = SPSA):

| policy | FD profit | tuned base-stock | tuned (s,S) | cover (L+1)mu |
|---|---|---|---|---|
| I1 | 758.6 | 678.7 | 748.4 | 744.1 |
| I2 | 808.7 | 678.7 | 748.4 | 744.1 |
| I3 | 761.1 | 678.7 | 748.4 | 744.1 |

- I2 beats every benchmark with 95% bootstrap CIs excluding zero (+7.6% vs (s,S), +18.6% vs base-stock). I1 does not significantly beat (s,S) (CI [-8, 30]); I3 beats (s,S) marginally (+2.7%).
- FD policies trade a lower fill rate (~98%) for much lower holding cost (avg on-hand ~20 vs ~41-46).
- Integer vs fractional orders give near-identical profit.
- Why tuned base-stock is weak (diagnosed, not a bug): benchmarks are tuned on the train split only. On the primary series the train-optimal level is 6.3 (profit 1326 on train) but val/test optima are ~4 (test: 756.8 at 4.0 vs 678.7 at 6.3). Demand/regime shifts between train and later periods, so the train-tuned level over-stocks. The FD policies are also trained on train only but select the best iterate on validation, which partly protects them. Treat the "vs base-stock" percentages as an advantage of validation-based selection plus a richer policy, not of FD alone.
- Test-tuned "oracle" benchmarks (upper bound, uses test data, not a fair competitor; `oracle_benchmarks.py`, `inventory_oracle.csv`):

| series | oracle base-stock | oracle (s,S) | best FD (policy) |
|---|---|---|---|
| FOODS_3_124_CA_3 | 759.0 | 831.6 | 808.7 (I2) |
| FOODS_2_128_WI_2 | 171.8 | 175.9 | 171.6 (I2) |
| HOBBIES_1_371_WI_1 | 104.3 | 113.5 | 102.9 (I2) |

  FD I1 (a base-stock policy) matches the oracle base-stock (758.6 vs 759.0 on the primary series), which is the expected ceiling for that family. No FD policy reaches the oracle (s,S) on any series.
- Intermittent series (FOODS_2_128_WI_2): FD beats all train-tuned benchmarks by 16-31%, CIs exclude zero, but this is again mostly the train-tuned benchmarks being mistuned; against the oracle it only ties.
- Seasonal series (HOBBIES_1_371_WI_1): FD does NOT beat the train-tuned (s,S) (I1 -12%, I2 -2%, I3 -9%; I2's CI against base-stock is slightly positive, +4%). It beats only the cover rule (+36-52%). Simple tuned baselines win or tie here.

## Honest conclusions / caveats
- **No FD policy beat buy-and-hold on the test period.** TECHM trended up; the best a learned policy can do is stay long. T2 effectively learned that; T1 over-trades (turnover ~33 vs 1) and loses ~0.2 cumulative return. The validation winner (T1) was not the test winner (ranking disagrees).
- Bootstrap CIs on cumulative return are very wide ([-0.4, 2.9]); differences to buy-and-hold are not statistically distinguishable from zero for T2.
- Single instrument, single period, adjusted-close deviation, assumed costs, no slippage/shorting costs: do not generalize.
- FD scales as O(n) evaluations per gradient; fine for 3-7 parameters, not for large policies.
- Simple baselines win or tie here.
- `pyyaml` is used for config (not in the original allowed list).

