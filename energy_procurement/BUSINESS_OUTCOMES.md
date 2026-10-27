# Business outcomes – Spanish day-ahead electricity procurement

Customer: 10 MW mean load. Baseline: buy the TSO day-ahead load forecast at the day-ahead price and settle any deviation at the (mostly higher) adjustment price. Test year: 2018 (365 days). Training/validation: 2015–2017. All figures in `outputs/eval/business/`.

## Headline (test year 2018, savings vs TSO-forecast purchasing)

| Policy | EUR/day | % of cost | Annualised EUR | 95% CI (EUR/day) |
|---|---|---|---|---|
| Newsvendor B, GBM scenarios | 183.5 | 1.29 % | ~67k | 62 – 352 |
| Mean-variance (GBM, kappa=8) | 139.9 | 0.99 % | ~51k | 58 – 256 |
| CVaR SP (GBM, alpha=.95, lambda=1) | 135.8 | 0.96 % | ~50k | 53 – 256 |
| Newsvendor B, bias scenarios (no GBM) | 41.8 | 0.29 % | ~15k | 34 – 51 |
| Oracle q = actual load | 73.0 | 0.51 % | ~27k | 65 – 82 |
| Bias-corrected forecast | 4.7 | 0.03 % | ~1.7k | 4 – 5 |
| Buy nothing day-ahead | -1,413 | -9.9 % | | |
| Buy at cap | -6,994 | -49 % | | |

Mean TSO cost is about EUR 14,202/day. CIs are bootstrap over days; the GBM CIs are wide (the gain is concentrated in a minority of days), so the GBM savings are positive but imprecise.

**The oracle is not a lower bound.** It knows the load but not the price gap `pA - pDA`, so a policy that forecasts the gap and deliberately under-buys when the gap is negative or zero beats it. Most of the saving comes from (i) the gap forecast and (ii) buying zero in the ~11 % of hours where the expected unit cost of buying is non-positive, not from load forecasting. Forecast value: bias correction alone is worth almost nothing (~5 EUR/day).

## Risk
CVaR(95 %) of daily cost falls from 19,066 (TSO) to ~18,790 for newsvendor/CVaR/MV policies; the worst day barely changes. Increasing lambda to 4 trades savings for risk and loses significance (32.7 EUR/day, CI includes 0). The effect on risk is modest (~1.5 %).

## Behaviour
The savings policies hedge ~87–95 % of the TSO volume, accepting 5–13 % shortfall settled at the adjustment price (`behaviour_test.png`). The policy never over-buys.

## Sensitivity (test year, EUR/day)
- Bias scenarios: base 41.8; autocorrelated errors rho=0.5: 25.2, rho=0.8: 69.0; shortfall penalty 10 EUR/MWh: 23.1.
- GBM scenarios: cutoff 1/2/3 days -> 231.7 / 181.7 / 174.0.
- Customer size scales linearly (x0.5 -> 20.9, x2 -> 83.7 for bias scenarios).

## Rolling-origin (quarterly retraining, bias scenarios, 2017Q1–2018Q4)
Newsvendor B saves 35–54 EUR/day in every quarter (8/8 positive); CVaR lambda=1 saves 19–34; MV kappa=8 saves 36–51. This is consistent with the 41.8 EUR/day static estimate. Rolling retraining of the GBM (and of the new neural forecaster) is reported in its own section below.

## Neural (PyTorch) forecaster vs gradient-boosted forecaster
`agents/forecast_nn.py`: one MLP (2×64, ReLU, dropout 0.1) outputs 19 monotone conditional quantiles plus a mean (pinball + MSE loss, early stopping, 3-seed ensemble). It is a drop-in replacement for the GBM scenario centre/quantiles (`run_extensions.py`, outputs in `outputs/eval/extensions/`). Training takes ~40 s versus ~22 min for the GBM bundle.

Forecast quality is slightly worse for the MLP (test pinball: load error 0.0038 vs 0.0035, gap 2.05 vs 1.85 EUR/MWh; mean-of-quantile-loss, lower is better) but its intervals are better calibrated (80 % band coverage 0.85 vs 0.82 for load error, 0.78 vs 0.76 for gap).

Savings vs TSO forecast, 2018 test year, EUR/day (95 % block-bootstrap CI):

| Policy | GBM | Neural MLP |
|---|---|---|
| Newsvendor B | 182 (63–355) | 174 (35–375) |
| CVaR SP (lam=1) | 136 (53–256) | 172 (56–347) |
| Mean-variance (kappa=8) | 140 (59–259) | 157 (60–301) |

The two forecasters are statistically indistinguishable; the neural model is a viable, ~30x cheaper substitute, not a clear improvement.

## Rolling retraining of the forecaster (GBM and MLP refit every quarter, 2017Q1–2018Q4)
Each quarter the forecaster is refit on all earlier days and scenario residuals come from earlier out-of-sample forecasts. Mean quarterly savings, EUR/day (quarters positive):

| Policy | GBM | Neural MLP |
|---|---|---|
| Newsvendor B | 174 (7/8) | 179 (7/8) |
| CVaR SP (lam=1) | 148 (8/8) | 149 (8/8) |
| Mean-variance (kappa=8) | 144 (8/8) | 146 (8/8) |

This confirms the GBM-based ~EUR 140–180/day estimate out of sample over eight quarters instead of one year, with the same sign in 7–8 of 8 quarters. The newsvendor lost money in 2018Q3 (-144 GBM, -92 MLP): the plain newsvendor is not robust in every quarter, while CVaR/mean-variance stayed positive (29/10 EUR/day in that quarter). Quarterly savings range from about 10 to 380 EUR/day, so the point estimate hides large quarter-to-quarter variation.

## Policy-gradient agents vs classical
Skill = (J_tso - J)/(J_tso - J_ref); 0 = TSO forecast, 1 = classical reference (10 seeds cold / 5 seeds warm, median, best-validation checkpoint, test year).

| Algorithm | Cold neutral | Cold CVaR | Cold MV | Warm neutral | Warm CVaR | Warm MV |
|---|---|---|---|---|---|---|
| FDPG (central diff.) | 0.26 | 0.40 | 3.0 | 0.87 | 1.10 | 1.9 |
| FDPG (SPSA) | 0.18 | 0.24 | 0.92 | 0.88 | 1.11 | 1.09 |
| A2C | 0.17 | 0.21 | 0.37 | 0.88 | 1.10 | 1.13 |
| REINFORCE | 0.06 | 0.08 | 0.19 | 0.88 | 1.11 | 1.12 |
| TRPO | 0.09 | 0.13 | 0.68 | 0.88 | 1.12 | 1.13 |
| PPO | 0.00 | 0.00 | 0.50 | 0.88 | 1.13 | 1.17 |
| DPG | 0.01 | 0.05 | 1.0 | 0.88 | 1.13 | 1.13 |

Findings:
1. Trained from scratch, no PG method matches the classical agents under the neutral or CVaR objective (best: FDPG at 26 % / 40 %). The signal-to-noise ratio of the gradient is very low (actions are 1–2 % of load), so the finite-difference method (FDPG, which exploits the known cost model) is the most sample-efficient; PPO/DPG/TRPO often diverged (see `diverged` column in `algorithm_summary.csv`).
2. Warm-started from the distilled classical policy, all methods keep ~0.88 skill but do not meaningfully improve it. PG is not a practical route to better procurement here; the classical agents are better and 10-100x cheaper.
3. **Skill above 1 for the unconditional mean-variance/CVaR objectives is an artefact, not a win.** The PG objective is the unconditional cross-day mean + kappa·variance of daily cost, dominated by the known day-ahead price level, whereas the classical agents minimise conditional per-day risk. The reference is therefore not the optimum of the evaluated objective, and skill's denominator is tiny for MV. Do not quote these as business savings; use the classical table above.
4. **Conditional-risk objective (fix for 3).** The `*_cond` objectives (`run_pg.py --objective neutral_cond|cvar_cond|meanvar_cond`) give the PG agent the same quantity the classical agents minimise: per day, mean + lambda·CVaR (or + kappa·variance) of cost over 64 scenario draws from the GBM scenario model, with reference and evaluation on 300 held-out-seed scenarios. Skill is then measured against the optimum of the evaluated objective, so values above 1 are no longer expected. Result (5 seeds, tuned learning rates, median test skill): best method FDPG — neutral 0.27, CVaR 0.18, mean-variance 0.44; every other method <= 0.30. So even with a low-noise model-based reward, PG from scratch recovers well under half of the classical benefit; the gap is an optimisation/sample-budget problem (30k rollouts, tiny action scale), not just reward noise.
5. **Neural (MLP) policies** (`--policy mlp`, 1 tanh hidden layer, 5 seeds, own learning-rate grid) help the model-free methods modestly but do not close the gap: neutral_cond skill A2C 0.16, DPG 0.18 (linear: 0.15 / 0.05), REINFORCE 0.11, TRPO 0.11, PPO 0.13 (linear 0.00); cvar_cond A2C 0.10, TRPO 0.07, DPG 0.03, PPO 0.00. All remain below linear FDPG.
6. **Learning rates per objective.** Grids were re-run for every objective (cvar, meanvar and the three `_cond` objectives; validation only, seeds 100–101). Selected rates matched the neutral grid for most cells; the differences were mean-variance (FDPG and SPSA 8e-3, A2C 1e-3, DPG 1e-4, PPO 1e-3), CVaR A2C (3e-4), cvar_cond FDPG (5e-4) and neutral_cond PPO (1e-3). Re-tuned unconditional runs (5 seeds) give the same ranking as before (cvar: FDPG 0.36, SPSA 0.27, A2C 0.24; meanvar: FDPG 3.8, SPSA 3.6, A2C 1.4), so the earlier reuse of the neutral grid did not change conclusions.

## Caveats
Bootstrap CIs treat days (7-day blocks) as exchangeable; synthetic customer scaled to 10 MW from system-wide load; data are 2015–2018 and market rules have changed since. Rolling quarterly results use out-of-sample scenario residuals but fixed hyper-parameters (no per-quarter tuning). The oracle is not a lower bound (see above).
