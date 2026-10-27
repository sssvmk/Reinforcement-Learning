# TRPO extension evaluation: neural (full) - EXPLORATORY

Validation J (normalised) and TEST result, mean over seeds, deterministic policy. Rows differ in tuning effort; seeds are paired with the reference in the optimiser (same theta0) but evaluation here uses seed-level means, so tests are Mann-Whitney over seeds. Diverged = validation J < -1e6.

Medians are reported next to means because a diverged seed ruins a mean.

| domain | key | policy | method | n_seeds | val_J | val_J_mean | test | test_mean | test_sd | n_diverged |
|---|---|---|---|---|---|---|---|---|---|---|
| inventory | FOODS_3_124_CA_3_validation | I4 | TRPO neural | 10 | 2.9593 | 2.9584 | 780.2086 | 781.3424 | 10.7102 | 0 |
| inventory | FOODS_3_124_CA_3_validation | I4 | VPG+GAE neural | 10 | 3.0618 | 3.0564 | 797.9452 | 798.6733 | 8.4474 | 0 |
| inventory | FOODS_3_124_CA_3_validation | I4 | REINFORCE neural (loo_time) | 10 | 3.0433 | 3.0456 | 800.1776 | 798.1903 | 9.5150 | 0 |
| inventory | FOODS_3_124_CA_3_validation | I4 | REINFORCE neural (none) | 10 | 2.9648 | 2.8907 | 779.2848 | 767.1518 | 48.4024 | 0 |
| inventory | FOODS_3_124_CA_3_validation | I1 | TRPO linear | 10 | 2.8992 | 2.9047 | 769.0642 | 771.8002 | 9.6589 | 0 |
| inventory | FOODS_3_124_CA_3_validation | I2 | TRPO linear | 10 | 3.0582 | 3.0575 | 801.3446 | 800.1657 | 7.1053 | 0 |
| inventory | FOODS_3_124_CA_3_validation | I3 | TRPO linear | 10 | 3.1192 | 3.1216 | 807.8980 | 807.2514 | 4.8503 | 0 |
| inventory | FOODS_2_128_WI_2_validation | I4 | TRPO neural | 10 | 0.7506 | 0.7556 | 166.9648 | 165.1872 | 4.3931 | 0 |
| inventory | FOODS_2_128_WI_2_validation | I4 | VPG+GAE neural | 10 | 0.8137 | 0.8073 | 171.2861 | 169.8889 | 5.8816 | 0 |
| inventory | FOODS_2_128_WI_2_validation | I4 | REINFORCE neural (loo_time) | 10 | 0.7682 | 0.7294 | 170.7093 | 163.2760 | 15.7220 | 0 |
| inventory | FOODS_2_128_WI_2_validation | I4 | REINFORCE neural (none) | 10 | 0.7524 | 0.7392 | 166.4390 | 163.9526 | 8.6455 | 0 |
| inventory | FOODS_2_128_WI_2_validation | I1 | TRPO linear | 10 | 0.7939 | 0.7921 | 170.0849 | 169.6494 | 1.4285 | 0 |
| inventory | FOODS_2_128_WI_2_validation | I2 | TRPO linear | 10 | 0.8100 | 0.8176 | 168.9881 | 169.6645 | 3.8100 | 0 |
| inventory | FOODS_2_128_WI_2_validation | I3 | TRPO linear | 10 | 0.8261 | 0.8392 | 165.9329 | 165.7088 | 5.7991 | 0 |
| inventory | HOBBIES_1_371_WI_1_validation | I4 | TRPO neural | 10 | 3.3135 | 3.3186 | 105.3814 | 105.7651 | 1.8934 | 0 |
| inventory | HOBBIES_1_371_WI_1_validation | I4 | VPG+GAE neural | 10 | 3.3474 | 3.3448 | 103.1040 | 101.4103 | 6.1132 | 0 |
| inventory | HOBBIES_1_371_WI_1_validation | I4 | REINFORCE neural (loo_time) | 10 | 3.2903 | 3.3001 | 100.5036 | 101.5191 | 3.7098 | 0 |
| inventory | HOBBIES_1_371_WI_1_validation | I4 | REINFORCE neural (none) | 10 | 3.2749 | 3.2933 | 101.0482 | 100.0766 | 6.3362 | 0 |
| inventory | HOBBIES_1_371_WI_1_validation | I1 | TRPO linear | 10 | 3.3500 | 3.3436 | 106.0134 | 105.4344 | 3.7200 | 0 |
| inventory | HOBBIES_1_371_WI_1_validation | I2 | TRPO linear | 10 | 3.3711 | 3.3767 | 102.2394 | 103.1730 | 2.8241 | 0 |
| inventory | HOBBIES_1_371_WI_1_validation | I3 | TRPO linear | 10 | 3.3021 | 3.2937 | 104.9727 | 105.1640 | 1.9074 | 0 |
| trading | TECHM | T3 | TRPO neural | 10 | 0.6882 | 0.6809 | 0.1346 | 0.1419 | 0.2685 | 0 |
| trading | TECHM | T3 | VPG+GAE neural | 10 | 0.9292 | 0.9714 | 0.6896 | 0.7664 | 0.2232 | 0 |
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
| inventory | FOODS_3_124_CA_3_validation | I4 | TRPO neural | VPG+GAE neural | 0.0036 | 0.0022 | -17.7366 | -0.8200 | 0.0242 |
| inventory | FOODS_3_124_CA_3_validation | I4 | TRPO neural | REINFORCE neural (loo_time) | 0.0036 | 0.0022 | -19.9690 | -0.8200 | 0.0242 |
| inventory | FOODS_3_124_CA_3_validation | I4 | TRPO neural | REINFORCE neural (none) | 0.5708 | 0.7913 | 0.9238 | 0.0800 | 1.0000 |
| inventory | FOODS_2_128_WI_2_validation | I4 | TRPO neural | VPG+GAE neural | 0.0022 | 0.0757 | -4.3213 | -0.4800 | 0.3026 |
| inventory | FOODS_2_128_WI_2_validation | I4 | TRPO neural | REINFORCE neural (loo_time) | 0.8501 | 0.3447 | -3.7445 | -0.2600 | 1.0000 |
| inventory | FOODS_2_128_WI_2_validation | I4 | TRPO neural | REINFORCE neural (none) | 0.6232 | 0.7337 | 0.5259 | -0.1000 | 1.0000 |
| inventory | HOBBIES_1_371_WI_1_validation | I4 | TRPO neural | VPG+GAE neural | 0.1405 | 0.0257 | 2.2774 | 0.6000 | 0.1545 |
| inventory | HOBBIES_1_371_WI_1_validation | I4 | TRPO neural | REINFORCE neural (loo_time) | 0.1859 | 0.0257 | 4.8778 | 0.6000 | 0.1545 |
| inventory | HOBBIES_1_371_WI_1_validation | I4 | TRPO neural | REINFORCE neural (none) | 0.4274 | 0.0140 | 4.3332 | 0.6600 | 0.0981 |
| trading | TECHM | T3 | TRPO neural | VPG+GAE neural | 0.0002 | 0.0002 | -0.5550 | -1.0000 | 0.0022 |
| trading | TECHM | T3 | TRPO neural | REINFORCE neural (loo_time) | 0.0002 | 0.0091 | -0.3854 | -0.7000 | 0.0820 |
| trading | TECHM | T3 | TRPO neural | REINFORCE neural (none) | 0.0002 | 0.0091 | -0.5008 | -0.7000 | 0.0820 |
