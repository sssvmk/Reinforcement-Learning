# Business-metrics summary (quick)

All costs are assumptions; the test period is never used for fitting or tuning. Trading returns use ADJUSTED `close` (price-scale audit failed; see DATA_NOTES.md).

## Trading
### TECHM / T1: test 2023-01-31 .. 2026-10-22 (best FD estimator by validation: spsa; mean ± sd over 3 seeds; cum CI = moving-block bootstrap, block 20, on the seed-averaged daily series)

| strategy | cum_return | ann_return | ann_vol | sharpe | sortino | max_drawdown | calmar | hit_rate | profit_factor | turnover | cost_drag | exposure | cum CI95 | cum-B&H CI95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FD-T1(spsa) | 0.767 ± 0.029 | 0.170 ± 0.005 | 0.239 ± 0.003 | 0.779 ± 0.012 | 1.177 ± 0.021 | -0.268 ± 0.003 | 0.637 ± 0.015 | 0.499 ± 0.001 | 1.143 ± 0.003 | 34.675 ± 7.993 | 0.017 ± 0.004 | 1.000 | [-0.395, 2.913] | [-0.856, 0.020] |
| buy_and_hold | 1.000 | 0.211 | 0.267 | 0.852 | 1.300 | -0.291 | 0.725 | 0.499 | 1.158 | 1.000 | 0.000 | 1.000 | [-0.330, 3.598] | [0.000, 0.000] |
| flat | 0.000 | 0.000 | 0.000 | nan | nan | 0.000 | nan | 0.000 | nan | 0.000 | 0.000 | 0.000 | [0.000, 0.000] | [-3.707, 0.372] |
| SAR_rule | 0.491 | 0.117 | 0.184 | 0.691 | 1.077 | -0.263 | 0.445 | 0.266 | 1.175 | 89.000 | 0.045 | 0.540 | [-0.331, 1.587] | [-2.341, 0.592] |
| RSI_rule | 1.000 | 0.211 | 0.267 | 0.852 | 1.300 | -0.291 | 0.725 | 0.499 | 1.158 | 1.000 | 0.000 | 1.000 | [-0.387, 3.344] | [0.000, 0.000] |

Ranking sanity check (validation window-return rank vs test rank):
| strategy | val_mean_window_return | test_cum_return | val_rank | test_rank |
|---|---|---|---|---|
| FD-T1(spsa) | 0.3225 | 0.7668 | 1 | 3 |
| buy_and_hold | 0.2887 | 1.0000 | 2 | 1 |
| flat | 0.0000 | 0.0000 | 5 | 5 |
| SAR_rule | 0.1685 | 0.4914 | 4 | 4 |
| RSI_rule | 0.1814 | 1.0000 | 3 | 1 |

Top on validation: **FD-T1(spsa)**; top on test: **buy_and_hold** (DISAGREE). Benchmarks that matched or beat the FD policy's mean test cumulative return: ['buy_and_hold', 'RSI_rule'].

### TECHM / T1: walk-forward (3 expanding folds over the test region, 3 seeds; each fold retrains on all prior data, last 25% of it for validation)

Per fold (mean over seeds):
| fold | strategy | cum_return | sharpe | max_drawdown |
|---|---|---|---|---|
| 1 | FD-T1(spsa) | 0.2556 | 0.9049 | -0.1548 |
| 1 | RSI_rule | 0.2941 | 0.9695 | -0.1622 |
| 1 | SAR_rule | 0.1345 | 0.6966 | -0.1301 |
| 1 | buy_and_hold | 0.2941 | 0.9695 | -0.1622 |
| 1 | flat | 0.0000 | nan | 0.0000 |
| 2 | FD-T1(spsa) | 0.2291 | 0.8234 | -0.2790 |
| 2 | RSI_rule | 0.2632 | 0.8881 | -0.2915 |
| 2 | SAR_rule | 0.0472 | 0.3113 | -0.2059 |
| 2 | buy_and_hold | 0.2535 | 0.8482 | -0.2915 |
| 2 | flat | 0.0000 | nan | 0.0000 |
| 3 | FD-T1(spsa) | 0.2034 | 0.7612 | -0.2144 |
| 3 | RSI_rule | 0.2793 | 0.8625 | -0.2469 |
| 3 | SAR_rule | 0.2713 | 1.0523 | -0.1667 |
| 3 | buy_and_hold | 0.2793 | 0.8625 | -0.2469 |
| 3 | flat | 0.0000 | nan | 0.0000 |

Pooled out-of-sample:
| strategy | cum_return | ann_return | sharpe | sortino | max_drawdown | calmar | exposure |
|---|---|---|---|---|---|---|---|
| FD-T1(spsa) | 0.858 ± 0.092 | 0.187 ± 0.016 | 0.831 ± 0.040 | 1.256 ± 0.066 | -0.279 ± 0.004 | 0.671 ± 0.052 | 0.992 ± 0.007 |
| buy_and_hold | 1.075 | 0.224 | 0.892 | 1.363 | -0.291 | 0.770 | 1.000 |
| flat | 0.000 | 0.000 | nan | nan | 0.000 | nan | 0.000 |
| RSI_rule | 1.091 | 0.227 | 0.905 | 1.387 | -0.291 | 0.778 | 0.980 |
| SAR_rule | 0.510 | 0.121 | 0.712 | 1.109 | -0.254 | 0.478 | 0.540 |

### TECHM / T2: test 2023-01-31 .. 2026-10-22 (best FD estimator by validation: central; mean ± sd over 3 seeds; cum CI = moving-block bootstrap, block 20, on the seed-averaged daily series)

| strategy | cum_return | ann_return | ann_vol | sharpe | sortino | max_drawdown | calmar | hit_rate | profit_factor | turnover | cost_drag | exposure | cum CI95 | cum-B&H CI95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FD-T2(central) | 0.982 ± 0.009 | 0.208 ± 0.002 | 0.259 ± 0.003 | 0.859 ± 0.002 | 1.313 ± 0.005 | -0.285 ± 0.003 | 0.731 ± 0.002 | 0.499 | 1.160 ± 0.000 | 4.664 ± 2.336 | 0.002 ± 0.001 | 1.000 | [-0.315, 3.245] | [-0.136, 0.016] |
| buy_and_hold | 1.000 | 0.211 | 0.267 | 0.852 | 1.300 | -0.291 | 0.725 | 0.499 | 1.158 | 1.000 | 0.000 | 1.000 | [-0.352, 3.210] | [0.000, 0.000] |
| flat | 0.000 | 0.000 | 0.000 | nan | nan | 0.000 | nan | 0.000 | nan | 0.000 | 0.000 | 0.000 | [0.000, 0.000] | [-3.473, 0.321] |
| SAR_rule | 0.491 | 0.117 | 0.184 | 0.691 | 1.077 | -0.263 | 0.445 | 0.266 | 1.175 | 89.000 | 0.045 | 0.540 | [-0.329, 1.495] | [-2.451, 0.651] |
| RSI_rule | 1.000 | 0.211 | 0.267 | 0.852 | 1.300 | -0.291 | 0.725 | 0.499 | 1.158 | 1.000 | 0.000 | 1.000 | [-0.364, 3.447] | [0.000, 0.000] |

Ranking sanity check (validation window-return rank vs test rank):
| strategy | val_mean_window_return | test_cum_return | val_rank | test_rank |
|---|---|---|---|---|
| FD-T2(central) | 0.2726 | 0.9822 | 2 | 3 |
| buy_and_hold | 0.2887 | 1.0000 | 1 | 1 |
| flat | 0.0000 | 0.0000 | 5 | 5 |
| SAR_rule | 0.1685 | 0.4914 | 4 | 4 |
| RSI_rule | 0.1814 | 1.0000 | 3 | 1 |

Top on validation: **buy_and_hold**; top on test: **buy_and_hold** (agree). Benchmarks that matched or beat the FD policy's mean test cumulative return: ['buy_and_hold', 'RSI_rule'].

### TECHM / T2: walk-forward (3 expanding folds over the test region, 3 seeds; each fold retrains on all prior data, last 25% of it for validation)

Per fold (mean over seeds):
| fold | strategy | cum_return | sharpe | max_drawdown |
|---|---|---|---|---|
| 1 | FD-T2(central) | 0.2925 | 0.9834 | -0.1576 |
| 1 | RSI_rule | 0.2941 | 0.9695 | -0.1622 |
| 1 | SAR_rule | 0.1345 | 0.6966 | -0.1301 |
| 1 | buy_and_hold | 0.2941 | 0.9695 | -0.1622 |
| 1 | flat | 0.0000 | nan | 0.0000 |
| 2 | FD-T2(central) | 0.2483 | 0.8499 | -0.2859 |
| 2 | RSI_rule | 0.2632 | 0.8881 | -0.2915 |
| 2 | SAR_rule | 0.0472 | 0.3113 | -0.2059 |
| 2 | buy_and_hold | 0.2535 | 0.8482 | -0.2915 |
| 2 | flat | 0.0000 | nan | 0.0000 |
| 3 | FD-T2(central) | 0.2759 | 0.8686 | -0.2416 |
| 3 | RSI_rule | 0.2793 | 0.8625 | -0.2469 |
| 3 | SAR_rule | 0.2713 | 1.0523 | -0.1667 |
| 3 | buy_and_hold | 0.2793 | 0.8625 | -0.2469 |
| 3 | flat | 0.0000 | nan | 0.0000 |

Pooled out-of-sample:
| strategy | cum_return | ann_return | sharpe | sortino | max_drawdown | calmar | exposure |
|---|---|---|---|---|---|---|---|
| FD-T2(central) | 1.059 ± 0.005 | 0.222 ± 0.001 | 0.899 ± 0.001 | 1.375 ± 0.001 | -0.286 ± 0.001 | 0.775 ± 0.001 | 1.000 |
| buy_and_hold | 1.075 | 0.224 | 0.892 | 1.363 | -0.291 | 0.770 | 1.000 |
| flat | 0.000 | 0.000 | nan | nan | 0.000 | nan | 0.000 |
| RSI_rule | 1.091 | 0.227 | 0.905 | 1.387 | -0.291 | 0.778 | 0.980 |
| SAR_rule | 0.510 | 0.121 | 0.712 | 1.109 | -0.254 | 0.478 | 0.540 |

## Inventory
PENDING: M5 `sell_prices.csv` is missing (see DATA_NOTES.md); no inventory results were produced.
