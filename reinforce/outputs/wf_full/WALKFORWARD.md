# Walk-forward comparison on FDPG's fold structure

3 expanding folds over the test region (n=4557, fold edges [3645, 3949, 4253, 4557]); each fold retrains on all prior data (last 25% validation); 5 seeds; 20000 training episodes/rollouts per run (equal budget for REINFORCE and the comparison methods; same seeds => same initial parameters across methods of the same policy).

## Pooled out-of-sample (mean and sd over seeds)

| strategy | cum_return_mean | cum_return_std | sharpe_mean | sharpe_std | max_drawdown_mean | max_drawdown_std | turnover_per_year_mean | turnover_per_year_std | exposure_mean | exposure_std | n_seeds |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R-T1(loo_time) | 0.9050 | 0.1263 | 0.8365 | 0.0810 | -0.2588 | 0.0484 | 11.0337 | 4.8273 | 0.9089 | 0.0225 | 5 |
| R-T2(loo_time) | 0.7782 | 0.2255 | 0.7914 | 0.1305 | -0.2704 | 0.0350 | 27.0020 | 8.4072 | 0.8171 | 0.0693 | 5 |
| R-T3-neural(loo_time) | 0.6856 | 0.3041 | 0.6951 | 0.1997 | -0.2847 | 0.0576 | 20.0158 | 6.3654 | 0.8587 | 0.0452 | 5 |
| RSI_rule | 1.0911 | nan | 0.9048 | nan | -0.2915 | nan | 0.8317 | nan | 0.9802 | nan | 1 |
| T1-position SPSA | 0.6286 | 0.1743 | 0.7577 | 0.1096 | -0.2428 | 0.0362 | 17.9441 | 6.0873 | 0.7512 | 0.0833 | 5 |
| T1-position central-FD | 0.7955 | 0.1368 | 0.7724 | 0.0912 | -0.2715 | 0.0421 | 9.0712 | 1.0516 | 0.9379 | 0.0323 | 5 |
| T1-position random-search | 0.7404 | 0.2290 | 0.7358 | 0.1299 | -0.2870 | 0.0277 | 12.3315 | 4.7547 | 0.9054 | 0.0550 | 5 |
| buy_and_hold | 1.0751 | nan | 0.8918 | nan | -0.2915 | nan | 0.2772 | nan | 1.0000 | nan | 1 |
| FDPG FD-T1(forward) (own code, 5 seeds) | 0.8458 | 0.0907 | 0.7990 | 0.0497 | -0.2893 | 0.0010 | 31.6945 | nan | 0.9925 | nan | 5 |
| FDPG FD-T2(regression) (own code, 5 seeds) | 1.0600 | 0.0000 | 0.8981 | 0.0000 | -0.2867 | 0.0000 | 3.5680 | nan | 1.0000 | nan | 5 |

## Per fold (cum. return, mean over seeds)

| strategy | 1 | 2 | 3 |
|---|---|---|---|
| R-T1(loo_time) | 0.2565 | 0.2260 | 0.2402 |
| R-T2(loo_time) | 0.2466 | 0.2870 | 0.1120 |
| R-T3-neural(loo_time) | 0.1956 | 0.2258 | 0.1458 |
| RSI_rule | 0.2941 | 0.2632 | 0.2793 |
| T1-position SPSA | 0.1605 | 0.1997 | 0.1670 |
| T1-position central-FD | 0.2021 | 0.2269 | 0.2174 |
| T1-position random-search | 0.2092 | 0.2074 | 0.1907 |
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
| R-T1(loo_time) | R-T2(loo_time) | -0.1268 | 0.3095 |
| R-T1(loo_time) | R-T3-neural(loo_time) | -0.2194 | 0.0952 |
| R-T1(loo_time) | T1-position central-FD | -0.1095 | 0.5476 |
| R-T1(loo_time) | T1-position SPSA | -0.2764 | 0.0556 |
| R-T1(loo_time) | T1-position random-search | -0.1646 | 0.3095 |

Notes: FDPG rows come from FDPG's saved CSVs (its own code, estimators chosen by validation, 40,000 rollouts, seeds 2000+); ours use this project's code and budget, so FDPG rows are indicative, not paired. FDPG's fold-specific policies were not saved, hence not re-evaluated here; the CSVs are sufficient for this comparison.
