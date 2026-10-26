# Walk-forward comparison on FDPG's fold structure

3 expanding folds over the test region (n=4557, fold edges [3645, 3949, 4253, 4557]); each fold retrains on all prior data (last 25% validation); 2 seeds; 3000 training episodes/rollouts per run (equal budget for REINFORCE and the comparison methods; same seeds => same initial parameters across methods of the same policy).

## Pooled out-of-sample (mean and sd over seeds)

| strategy | cum_return_mean | cum_return_std | sharpe_mean | sharpe_std | max_drawdown_mean | max_drawdown_std | turnover_per_year_mean | turnover_per_year_std | exposure_mean | exposure_std | n_seeds |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R-T1(loo_time) | 0.6428 | 0.5767 | 0.6394 | 0.3844 | -0.2746 | 0.0239 | 29.1089 | 14.1141 | 0.8482 | 0.0965 | 2 |
| R-T2(loo_time) | 0.5980 | 0.0805 | 0.7183 | 0.0959 | -0.2353 | 0.0218 | 28.5545 | 1.9603 | 0.7236 | 0.0751 | 2 |
| R-T3-neural(loo_time) | 0.9887 | 0.1263 | 0.9148 | 0.0525 | -0.2285 | 0.0070 | 33.5446 | 31.3647 | 0.8867 | 0.0576 | 2 |
| RSI_rule | 1.0911 | nan | 0.9048 | nan | -0.2915 | nan | 0.8317 | nan | 0.9802 | nan | 1 |
| T1-position SPSA | 0.5860 | 0.2761 | 0.7836 | 0.2567 | -0.2397 | 0.0414 | 20.7609 | 0.3748 | 0.6377 | 0.0191 | 2 |
| T1-position central-FD | 0.7013 | 0.1618 | 0.7361 | 0.0938 | -0.2829 | 0.0070 | 14.3262 | 3.5024 | 0.8924 | 0.0251 | 2 |
| T1-position random-search | 0.4848 | 0.0716 | 0.5858 | 0.0633 | -0.2815 | 0.0229 | 16.3559 | 1.6899 | 0.8545 | 0.0091 | 2 |
| buy_and_hold | 1.0751 | nan | 0.8918 | nan | -0.2915 | nan | 0.2772 | nan | 1.0000 | nan | 1 |
| FDPG FD-T1(forward) (own code, 5 seeds) | 0.8458 | 0.0907 | 0.7990 | 0.0497 | -0.2893 | 0.0010 | 31.6945 | nan | 0.9925 | nan | 5 |
| FDPG FD-T2(regression) (own code, 5 seeds) | 1.0600 | 0.0000 | 0.8981 | 0.0000 | -0.2867 | 0.0000 | 3.5680 | nan | 1.0000 | nan | 5 |

## Per fold (cum. return, mean over seeds)

| strategy | 1 | 2 | 3 |
|---|---|---|---|
| R-T1(loo_time) | 0.1606 | 0.1366 | 0.2252 |
| R-T2(loo_time) | 0.0654 | 0.2336 | 0.2183 |
| R-T3-neural(loo_time) | 0.3131 | 0.2954 | 0.1719 |
| RSI_rule | 0.2941 | 0.2632 | 0.2793 |
| T1-position SPSA | 0.1625 | 0.1812 | 0.1503 |
| T1-position central-FD | 0.1761 | 0.2343 | 0.1708 |
| T1-position random-search | 0.0989 | 0.2244 | 0.1042 |
| buy_and_hold | 0.2941 | 0.2535 | 0.2793 |

## Fold dates: ours vs FDPG's own CSV

ours:

| fold | start | end |
|---|---|---|
| 1 | 2023-01-31 | 2024-04-30 |
| 2 | 2024-05-03 | 2025-07-23 |
| 3 | 2025-07-25 | 2026-10-22 |

FDPG:

| fold | start | end |
|---|---|---|
| 1 | 2023-01-31 | 2024-04-30 |
| 2 | 2024-05-03 | 2025-07-23 |
| 3 | 2025-07-25 | 2026-10-22 |

## Tests vs R-T1 (Mann-Whitney over seeds, pooled cum. return)

| vs | strategy | mean_diff_cum_return | p_mannwhitney |
|---|---|---|---|
| R-T1(loo_time) | R-T2(loo_time) | -0.0448 | 1.0000 |
| R-T1(loo_time) | R-T3-neural(loo_time) | 0.3460 | 0.6667 |
| R-T1(loo_time) | T1-position central-FD | 0.0585 | 1.0000 |
| R-T1(loo_time) | T1-position SPSA | -0.0567 | 1.0000 |
| R-T1(loo_time) | T1-position random-search | -0.1580 | 1.0000 |

Notes: FDPG rows come from FDPG's saved CSVs (its own code, estimators chosen by validation, 40,000 rollouts, seeds 2000+); ours use this project's code and budget, so FDPG rows are indicative, not paired. FDPG's fold-specific policies were not saved, hence not re-evaluated here; the CSVs are sufficient for this comparison.
