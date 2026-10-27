# TRPO extension evaluation: neural (quick) - EXPLORATORY

Validation J (normalised) and TEST result, mean over seeds, deterministic policy. Rows differ in tuning effort; seeds are paired with the reference in the optimiser (same theta0) but evaluation here uses seed-level means, so tests are Mann-Whitney over seeds. Diverged = validation J < -1e6.

Medians are reported next to means because a diverged seed ruins a mean.

| domain | key | policy | method | n_seeds | val_J | val_J_mean | test | test_mean | test_sd | n_diverged |
|---|---|---|---|---|---|---|---|---|---|---|
| inventory | FOODS_3_124_CA_3_validation | I4 | TRPO neural | 2 | 2.8047 | 2.8047 | 753.8058 | 753.8058 | 26.0480 | 0 |
| inventory | FOODS_3_124_CA_3_validation | I4 | VPG+GAE neural | 2 | 2.8673 | 2.8673 | 761.4339 | 761.4339 | 10.2990 | 0 |
| inventory | FOODS_3_124_CA_3_validation | I4 | REINFORCE neural (loo_time) | 10 | 3.0433 | 3.0456 | 800.1776 | 798.1903 | 9.5150 | 0 |
| inventory | FOODS_3_124_CA_3_validation | I4 | REINFORCE neural (none) | 10 | 2.9648 | 2.8907 | 779.2848 | 767.1518 | 48.4024 | 0 |
| inventory | FOODS_3_124_CA_3_validation | I1 | TRPO linear | 10 | 2.8992 | 2.9047 | 769.0642 | 771.8002 | 9.6589 | 0 |
| inventory | FOODS_3_124_CA_3_validation | I2 | TRPO linear | 10 | 3.0582 | 3.0575 | 801.3446 | 800.1657 | 7.1053 | 0 |
| inventory | FOODS_3_124_CA_3_validation | I3 | TRPO linear | 10 | 3.1192 | 3.1216 | 807.8980 | 807.2514 | 4.8503 | 0 |
| inventory | FOODS_2_128_WI_2_validation | I4 | TRPO neural | 2 | 0.6556 | 0.6556 | 156.2594 | 156.2594 | 6.5193 | 0 |
| inventory | FOODS_2_128_WI_2_validation | I4 | VPG+GAE neural | 2 | 0.7674 | 0.7674 | 173.1816 | 173.1816 | 3.1085 | 0 |
| inventory | FOODS_2_128_WI_2_validation | I4 | REINFORCE neural (loo_time) | 10 | 0.7682 | 0.7294 | 170.7093 | 163.2760 | 15.7220 | 0 |
| inventory | FOODS_2_128_WI_2_validation | I4 | REINFORCE neural (none) | 10 | 0.7524 | 0.7392 | 166.4390 | 163.9526 | 8.6455 | 0 |
| inventory | FOODS_2_128_WI_2_validation | I1 | TRPO linear | 10 | 0.7939 | 0.7921 | 170.0849 | 169.6494 | 1.4285 | 0 |
| inventory | FOODS_2_128_WI_2_validation | I2 | TRPO linear | 10 | 0.8100 | 0.8176 | 168.9881 | 169.6645 | 3.8100 | 0 |
| inventory | FOODS_2_128_WI_2_validation | I3 | TRPO linear | 10 | 0.8261 | 0.8392 | 165.9329 | 165.7088 | 5.7991 | 0 |
| inventory | HOBBIES_1_371_WI_1_validation | I4 | TRPO neural | 2 | 3.0556 | 3.0556 | 96.6256 | 96.6256 | 8.5364 | 0 |
| inventory | HOBBIES_1_371_WI_1_validation | I4 | VPG+GAE neural | 2 | 3.1639 | 3.1639 | 96.2157 | 96.2157 | 5.9057 | 0 |
| inventory | HOBBIES_1_371_WI_1_validation | I4 | REINFORCE neural (loo_time) | 10 | 3.2903 | 3.3001 | 100.5036 | 101.5191 | 3.7098 | 0 |
| inventory | HOBBIES_1_371_WI_1_validation | I4 | REINFORCE neural (none) | 10 | 3.2749 | 3.2933 | 101.0482 | 100.0766 | 6.3362 | 0 |
| inventory | HOBBIES_1_371_WI_1_validation | I1 | TRPO linear | 10 | 3.3500 | 3.3436 | 106.0134 | 105.4344 | 3.7200 | 0 |
| inventory | HOBBIES_1_371_WI_1_validation | I2 | TRPO linear | 10 | 3.3711 | 3.3767 | 102.2394 | 103.1730 | 2.8241 | 0 |
| inventory | HOBBIES_1_371_WI_1_validation | I3 | TRPO linear | 10 | 3.3021 | 3.2937 | 104.9727 | 105.1640 | 1.9074 | 0 |
| trading | TECHM | T3 | TRPO neural | 2 | 0.6016 | 0.6016 | 0.3079 | 0.3079 | 0.4505 | 0 |
| trading | TECHM | T3 | VPG+GAE neural | 2 | 0.5956 | 0.5956 | 0.4533 | 0.4533 | 0.3981 | 0 |
| trading | TECHM | T3 | REINFORCE neural (loo_time) | 10 | 1.0048 | 1.0332 | 0.5200 | 0.5699 | 0.3354 | 0 |
| trading | TECHM | T3 | REINFORCE neural (none) | 10 | 1.0471 | 1.0475 | 0.6353 | 0.5486 | 0.3259 | 0 |
| trading | TECHM | T1 | TRPO linear | 10 | 1.0355 | 0.9409 | 0.5397 | 0.4609 | 0.3009 | 0 |
| trading | TECHM | T2 | TRPO linear | 10 | 0.8840 | 0.8805 | 0.4165 | 0.4327 | 0.1775 | 0 |
| trading | TECHM | - | buy_and_hold | 1 | 0.6691 | 0.6691 | 0.9884 | 0.9884 | 0.0000 | 0 |
| trading | TECHM | - | flat | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 |
| trading | TECHM | - | RSI_rule | 1 | 0.4463 | 0.4463 | 1.0129 | 1.0129 | 0.0000 | 0 |
| trading | TECHM | - | SAR_rule | 1 | 0.3947 | 0.3947 | 0.4955 | 0.4955 | 0.0000 | 0 |

## Reference vs each other row (Mann-Whitney over seeds, Holm over all rows)

| domain | key | policy | ref | vs | val_p | test_p | test_median_diff | cliffs_test | test_p_holm |
|---|---|---|---|---|---|---|---|---|---|
| inventory | FOODS_3_124_CA_3_validation | I4 | TRPO neural | REINFORCE neural (loo_time) | 0.0303 | 0.0303 | -46.3718 | -1.0000 | 0.2424 |
| inventory | FOODS_3_124_CA_3_validation | I4 | TRPO neural | REINFORCE neural (none) | 0.1818 | 0.2727 | -25.4790 | -0.6000 | 1.0000 |
| inventory | FOODS_2_128_WI_2_validation | I4 | TRPO neural | REINFORCE neural (loo_time) | 0.2727 | 0.3636 | -14.4499 | -0.5000 | 1.0000 |
| inventory | FOODS_2_128_WI_2_validation | I4 | TRPO neural | REINFORCE neural (none) | 0.1212 | 0.1818 | -10.1796 | -0.7000 | 1.0000 |
| inventory | HOBBIES_1_371_WI_1_validation | I4 | TRPO neural | REINFORCE neural (loo_time) | 0.0303 | 0.4848 | -3.8780 | -0.4000 | 1.0000 |
| inventory | HOBBIES_1_371_WI_1_validation | I4 | TRPO neural | REINFORCE neural (none) | 0.1212 | 0.6061 | -4.4226 | -0.3000 | 1.0000 |
| trading | TECHM | T3 | TRPO neural | REINFORCE neural (loo_time) | 0.0303 | 0.4848 | -0.2121 | -0.4000 | 1.0000 |
| trading | TECHM | T3 | TRPO neural | REINFORCE neural (none) | 0.0606 | 0.4848 | -0.3274 | -0.4000 | 1.0000 |
