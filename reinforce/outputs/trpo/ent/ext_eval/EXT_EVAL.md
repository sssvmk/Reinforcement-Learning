# TRPO extension evaluation: entropy (full) - EXPLORATORY

Validation J (normalised) and TEST result, mean over seeds, deterministic policy. Rows differ in tuning effort; seeds are paired with the reference in the optimiser (same theta0) but evaluation here uses seed-level means, so tests are Mann-Whitney over seeds. Diverged = validation J < -1e6.

Medians are reported next to means because a diverged seed ruins a mean.

| domain | key | policy | method | n_seeds | val_J | val_J_mean | test | test_mean | test_sd | n_diverged |
|---|---|---|---|---|---|---|---|---|---|---|
| inventory | FOODS_3_124_CA_3_validation | I1 | TRPO + entropy bonus | 10 | 2.8967 | 2.9062 | 768.8295 | 770.9894 | 8.5704 | 0 |
| inventory | FOODS_3_124_CA_3_validation | I2 | TRPO + entropy bonus | 10 | 3.0372 | 3.0375 | 801.4368 | 800.2502 | 5.4563 | 0 |
| inventory | FOODS_3_124_CA_3_validation | I3 | TRPO + entropy bonus | 10 | 3.1201 | 3.1213 | 804.5498 | 806.7782 | 7.7034 | 0 |
| inventory | FOODS_3_124_CA_3_validation | I1 | TRPO (plain) | 10 | 2.8992 | 2.9047 | 769.0642 | 771.8002 | 9.6589 | 0 |
| inventory | FOODS_3_124_CA_3_validation | I2 | TRPO (plain) | 10 | 3.0582 | 3.0575 | 801.3446 | 800.1657 | 7.1053 | 0 |
| inventory | FOODS_3_124_CA_3_validation | I3 | TRPO (plain) | 10 | 3.1192 | 3.1216 | 807.8980 | 807.2514 | 4.8503 | 0 |
| inventory | FOODS_2_128_WI_2_validation | I1 | TRPO + entropy bonus | 10 | 0.7881 | 0.7846 | 169.6081 | 169.5347 | 2.2380 | 0 |
| inventory | FOODS_2_128_WI_2_validation | I2 | TRPO + entropy bonus | 10 | 0.8289 | 0.8229 | 170.7467 | 170.1234 | 3.8615 | 0 |
| inventory | FOODS_2_128_WI_2_validation | I3 | TRPO + entropy bonus | 10 | 0.8318 | 0.8276 | 169.1989 | 169.0616 | 3.5264 | 0 |
| inventory | FOODS_2_128_WI_2_validation | I1 | TRPO (plain) | 10 | 0.7939 | 0.7921 | 170.0849 | 169.6494 | 1.4285 | 0 |
| inventory | FOODS_2_128_WI_2_validation | I2 | TRPO (plain) | 10 | 0.8100 | 0.8176 | 168.9881 | 169.6645 | 3.8100 | 0 |
| inventory | FOODS_2_128_WI_2_validation | I3 | TRPO (plain) | 10 | 0.8261 | 0.8392 | 165.9329 | 165.7088 | 5.7991 | 0 |
| inventory | HOBBIES_1_371_WI_1_validation | I1 | TRPO + entropy bonus | 10 | 3.3470 | 3.3417 | 106.9852 | 105.8823 | 4.3336 | 0 |
| inventory | HOBBIES_1_371_WI_1_validation | I2 | TRPO + entropy bonus | 10 | 3.3721 | 3.3586 | 102.7436 | 103.1567 | 2.5757 | 0 |
| inventory | HOBBIES_1_371_WI_1_validation | I3 | TRPO + entropy bonus | 10 | 3.3186 | 3.3177 | 105.2003 | 104.9399 | 2.0476 | 0 |
| inventory | HOBBIES_1_371_WI_1_validation | I1 | TRPO (plain) | 10 | 3.3500 | 3.3436 | 106.0134 | 105.4344 | 3.7200 | 0 |
| inventory | HOBBIES_1_371_WI_1_validation | I2 | TRPO (plain) | 10 | 3.3711 | 3.3767 | 102.2394 | 103.1730 | 2.8241 | 0 |
| inventory | HOBBIES_1_371_WI_1_validation | I3 | TRPO (plain) | 10 | 3.3021 | 3.2937 | 104.9727 | 105.1640 | 1.9074 | 0 |
| trading | TECHM | T1 | TRPO + entropy bonus | 10 | 1.0306 | 1.0100 | 0.5349 | 0.5360 | 0.2419 | 0 |
| trading | TECHM | T2 | TRPO + entropy bonus | 10 | 0.9034 | 0.9210 | 0.4659 | 0.4872 | 0.2734 | 0 |
| trading | TECHM | T1 | TRPO (plain) | 10 | 1.0355 | 0.9409 | 0.5397 | 0.4609 | 0.3009 | 0 |
| trading | TECHM | T2 | TRPO (plain) | 10 | 0.8840 | 0.8805 | 0.4165 | 0.4327 | 0.1775 | 0 |
| trading | TECHM | - | buy_and_hold | 1 | 0.6691 | 0.6691 | 0.9884 | 0.9884 | 0.0000 | 0 |
| trading | TECHM | - | flat | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 |
| trading | TECHM | - | RSI_rule | 1 | 0.4463 | 0.4463 | 1.0129 | 1.0129 | 0.0000 | 0 |
| trading | TECHM | - | SAR_rule | 1 | 0.3947 | 0.3947 | 0.4955 | 0.4955 | 0.0000 | 0 |

## Reference vs each other row (Mann-Whitney over seeds, Holm over all rows)

| domain | key | policy | ref | vs | val_p | test_p | test_median_diff | cliffs_test | test_p_holm |
|---|---|---|---|---|---|---|---|---|---|
| inventory | FOODS_3_124_CA_3_validation | I1 | TRPO + entropy bonus | TRPO (plain) | 0.5708 | 0.8501 | -0.2348 | -0.0600 | 1.0000 |
| inventory | FOODS_3_124_CA_3_validation | I2 | TRPO + entropy bonus | TRPO (plain) | 0.1041 | 0.9097 | 0.0922 | -0.0400 | 1.0000 |
| inventory | FOODS_3_124_CA_3_validation | I3 | TRPO + entropy bonus | TRPO (plain) | 0.5708 | 0.6232 | -3.3481 | -0.1400 | 1.0000 |
| inventory | FOODS_2_128_WI_2_validation | I1 | TRPO + entropy bonus | TRPO (plain) | 0.0539 | 0.6776 | -0.4768 | -0.1200 | 1.0000 |
| inventory | FOODS_2_128_WI_2_validation | I2 | TRPO + entropy bonus | TRPO (plain) | 0.8501 | 0.6776 | 1.7586 | 0.1200 | 1.0000 |
| inventory | FOODS_2_128_WI_2_validation | I3 | TRPO + entropy bonus | TRPO (plain) | 0.8501 | 0.1041 | 3.2660 | 0.4400 | 1.0000 |
| inventory | HOBBIES_1_371_WI_1_validation | I1 | TRPO + entropy bonus | TRPO (plain) | 0.7337 | 0.7337 | 0.9718 | 0.1000 | 1.0000 |
| inventory | HOBBIES_1_371_WI_1_validation | I2 | TRPO + entropy bonus | TRPO (plain) | 0.6232 | 0.7913 | 0.5041 | 0.0800 | 1.0000 |
| inventory | HOBBIES_1_371_WI_1_validation | I3 | TRPO + entropy bonus | TRPO (plain) | 0.2123 | 0.8501 | 0.2275 | 0.0600 | 1.0000 |
| trading | TECHM | T1 | TRPO + entropy bonus | TRPO (plain) | 0.8501 | 0.7337 | -0.0048 | 0.1000 | 1.0000 |
| trading | TECHM | T2 | TRPO + entropy bonus | TRPO (plain) | 0.5708 | 0.7337 | 0.0494 | 0.1000 | 1.0000 |
