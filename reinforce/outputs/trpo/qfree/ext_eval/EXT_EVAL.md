# TRPO extension evaluation: qfree (full) - EXPLORATORY

Validation J (normalised) and TEST result, mean over seeds, deterministic policy. Rows differ in tuning effort; seeds are paired with the reference in the optimiser (same theta0) but evaluation here uses seed-level means, so tests are Mann-Whitney over seeds. Diverged = validation J < -1e6.

Medians are reported next to means because a diverged seed ruins a mean.

| domain | key | policy | method | n_seeds | val_J | val_J_mean | test | test_mean | test_sd | n_diverged | share_orders_above_7mu |
|---|---|---|---|---|---|---|---|---|---|---|---|
| inventory | FOODS_3_124_CA_3_validation | I1 | TRPO Q_max-free | 10 | 2.8992 | 2.9047 | 769.0642 | 771.8003 | 9.6590 | 0 | 0.0000 |
| inventory | FOODS_3_124_CA_3_validation | I2 | TRPO Q_max-free | 10 | 3.0321 | 3.0344 | 800.9654 | 798.7654 | 8.5541 | 0 | 0.0003 |
| inventory | FOODS_3_124_CA_3_validation | I1 | VPG+GAE Q_max-free | 10 | 2.9059 | 2.9142 | 780.5435 | 780.1678 | 6.7067 | 0 | 0.0000 |
| inventory | FOODS_3_124_CA_3_validation | I2 | VPG+GAE Q_max-free | 10 | 3.0123 | 3.0260 | 793.2093 | 790.2714 | 11.0540 | 0 | 0.0039 |
| inventory | FOODS_3_124_CA_3_validation | I1 | REINFORCE Q_max-free (loo_time) | 10 | 2.8748 | 2.8724 | 756.7700 | 757.3654 | 2.5985 | 0 | 0.0000 |
| inventory | FOODS_3_124_CA_3_validation | I2 | REINFORCE Q_max-free (loo_time) | 10 | 3.0115 | 2.9511 | 772.1928 | 768.1764 | 27.2917 | 0 | 0.0033 |
| inventory | FOODS_3_124_CA_3_validation | I1 | REINFORCE Q_max-free (none) | 10 | 2.8748 | 2.8741 | 754.3394 | 754.5057 | 5.2596 | 0 | 0.0000 |
| inventory | FOODS_3_124_CA_3_validation | I2 | REINFORCE Q_max-free (none) | 10 | 2.9401 | 2.9417 | 763.4931 | 772.2889 | 19.4088 | 0 | 0.0011 |
| inventory | FOODS_3_124_CA_3_validation | I1 | TRPO capped 7mu | 10 | 2.8992 | 2.9047 | 769.0642 | 771.8002 | 9.6589 | 0 | 0.0000 |
| inventory | FOODS_3_124_CA_3_validation | I2 | TRPO capped 7mu | 10 | 3.0580 | 3.0558 | 802.0281 | 800.4587 | 6.7371 | 0 | 0.0007 |
| inventory | FOODS_3_124_CA_3_validation | I1 | FDPG (spsa, best on val) | 10 | 2.8845 | 2.8844 | 758.5043 | 758.5631 | 0.1073 | 0 | nan |
| inventory | FOODS_3_124_CA_3_validation | I2 | FDPG (spsa, best on val) | 10 | 3.1113 | 3.1065 | 809.4609 | 808.7173 | 10.1316 | 0 | nan |
| inventory | FOODS_2_128_WI_2_validation | I1 | TRPO Q_max-free | 10 | 0.7939 | 0.7921 | 170.0849 | 169.6494 | 1.4285 | 0 | 0.0000 |
| inventory | FOODS_2_128_WI_2_validation | I2 | TRPO Q_max-free | 10 | 0.8117 | 0.8202 | 168.9881 | 169.7292 | 3.9616 | 0 | 0.0003 |
| inventory | FOODS_2_128_WI_2_validation | I1 | VPG+GAE Q_max-free | 10 | 0.7568 | 0.7491 | 169.1197 | 168.6569 | 1.7510 | 0 | 0.0000 |
| inventory | FOODS_2_128_WI_2_validation | I2 | VPG+GAE Q_max-free | 10 | 0.7396 | 0.7439 | 161.5161 | 160.3863 | 7.4464 | 0 | 0.0026 |
| inventory | FOODS_2_128_WI_2_validation | I1 | REINFORCE Q_max-free (loo_time) | 10 | 0.7480 | 0.7460 | 166.3850 | 165.8785 | 2.6081 | 0 | 0.0000 |
| inventory | FOODS_2_128_WI_2_validation | I2 | REINFORCE Q_max-free (loo_time) | 10 | 0.7577 | 0.7567 | 168.7811 | 168.7656 | 3.4536 | 0 | 0.0007 |
| inventory | FOODS_2_128_WI_2_validation | I1 | REINFORCE Q_max-free (none) | 10 | 0.7403 | 0.7401 | 167.4614 | 166.9751 | 2.2693 | 0 | 0.0000 |
| inventory | FOODS_2_128_WI_2_validation | I2 | REINFORCE Q_max-free (none) | 10 | 0.7599 | 0.7308 | 164.4353 | 159.3263 | 17.3357 | 0 | 0.0004 |
| inventory | FOODS_2_128_WI_2_validation | I1 | TRPO capped 7mu | 10 | 0.7939 | 0.7921 | 170.0849 | 169.6494 | 1.4285 | 0 | 0.0000 |
| inventory | FOODS_2_128_WI_2_validation | I2 | TRPO capped 7mu | 10 | 0.8100 | 0.8169 | 168.9881 | 169.6819 | 3.8503 | 0 | 0.0001 |
| inventory | FOODS_2_128_WI_2_validation | I1 | FDPG (regression, best on val) | 10 | 0.8330 | 0.8330 | 170.5035 | 170.4987 | 0.0272 | 0 | nan |
| inventory | FOODS_2_128_WI_2_validation | I2 | FDPG (regression, best on val) | 10 | 0.8376 | 0.8319 | 171.9034 | 171.6441 | 2.0829 | 0 | nan |
| inventory | HOBBIES_1_371_WI_1_validation | I1 | TRPO Q_max-free | 10 | 3.3519 | 3.3500 | 105.8103 | 105.5354 | 3.8211 | 0 | 0.0017 |
| inventory | HOBBIES_1_371_WI_1_validation | I2 | TRPO Q_max-free | 10 | 3.3822 | 3.3796 | 104.4876 | 104.0319 | 2.4608 | 0 | 0.0071 |
| inventory | HOBBIES_1_371_WI_1_validation | I1 | VPG+GAE Q_max-free | 10 | 3.3349 | 3.3192 | 104.5424 | 104.6538 | 3.9345 | 0 | 0.0000 |
| inventory | HOBBIES_1_371_WI_1_validation | I2 | VPG+GAE Q_max-free | 10 | 3.3598 | 3.3518 | 98.3302 | 98.4092 | 4.3248 | 0 | 0.0041 |
| inventory | HOBBIES_1_371_WI_1_validation | I1 | REINFORCE Q_max-free (loo_time) | 10 | 3.3117 | 3.3107 | 103.9726 | 102.6789 | 4.5103 | 0 | 0.0003 |
| inventory | HOBBIES_1_371_WI_1_validation | I2 | REINFORCE Q_max-free (loo_time) | 10 | 3.3142 | 3.2485 | 99.3752 | 95.2122 | 9.7591 | 0 | 0.0146 |
| inventory | HOBBIES_1_371_WI_1_validation | I1 | REINFORCE Q_max-free (none) | 10 | 3.3085 | 3.2990 | 99.6368 | 100.0379 | 5.6372 | 0 | 0.0011 |
| inventory | HOBBIES_1_371_WI_1_validation | I2 | REINFORCE Q_max-free (none) | 10 | 3.2255 | 3.1784 | 101.6822 | 100.3621 | 6.7383 | 0 | 0.0144 |
| inventory | HOBBIES_1_371_WI_1_validation | I1 | TRPO capped 7mu | 10 | 3.3500 | 3.3436 | 105.9730 | 105.4116 | 3.6963 | 0 | 0.0024 |
| inventory | HOBBIES_1_371_WI_1_validation | I2 | TRPO capped 7mu | 10 | 3.3605 | 3.3727 | 102.1724 | 102.7909 | 2.7436 | 0 | 0.0080 |
| inventory | HOBBIES_1_371_WI_1_validation | I1 | FDPG (forward, best on val) | 10 | 3.0422 | 3.0422 | 94.7999 | 94.8806 | 0.2749 | 0 | nan |
| inventory | HOBBIES_1_371_WI_1_validation | I2 | FDPG (spsa_nocrn, best on val) | 10 | 3.4037 | 3.3830 | 103.1222 | 101.9117 | 5.5678 | 0 | nan |

## Reference vs each other row (Mann-Whitney over seeds, Holm over all rows)

| domain | key | policy | ref | vs | val_p | test_p | test_median_diff | cliffs_test | test_p_holm |
|---|---|---|---|---|---|---|---|---|---|
| inventory | FOODS_3_124_CA_3_validation | I1 | TRPO Q_max-free | VPG+GAE Q_max-free | 0.4727 | 0.0640 | -11.4793 | -0.5000 | 0.8323 |
| inventory | FOODS_3_124_CA_3_validation | I2 | TRPO Q_max-free | VPG+GAE Q_max-free | 0.6232 | 0.1041 | 7.7561 | 0.4400 | 1.0000 |
| inventory | FOODS_3_124_CA_3_validation | I1 | TRPO Q_max-free | REINFORCE Q_max-free (loo_time) | 0.0010 | 0.0006 | 12.2942 | 0.9200 | 0.0134 |
| inventory | FOODS_3_124_CA_3_validation | I2 | TRPO Q_max-free | REINFORCE Q_max-free (loo_time) | 0.1859 | 0.0017 | 28.7726 | 0.8400 | 0.0375 |
| inventory | FOODS_3_124_CA_3_validation | I1 | TRPO Q_max-free | REINFORCE Q_max-free (none) | 0.0046 | 0.0004 | 14.7248 | 0.9400 | 0.0106 |
| inventory | FOODS_3_124_CA_3_validation | I2 | TRPO Q_max-free | REINFORCE Q_max-free (none) | 0.0890 | 0.0046 | 37.4723 | 0.7600 | 0.0826 |
| inventory | FOODS_3_124_CA_3_validation | I1 | TRPO Q_max-free | TRPO capped 7mu | 1.0000 | 1.0000 | 0.0000 | 0.0100 | 1.0000 |
| inventory | FOODS_3_124_CA_3_validation | I2 | TRPO Q_max-free | TRPO capped 7mu | 0.1041 | 0.6776 | -1.0627 | -0.1200 | 1.0000 |
| inventory | FOODS_2_128_WI_2_validation | I1 | TRPO Q_max-free | VPG+GAE Q_max-free | 0.0002 | 0.1405 | 0.9652 | 0.4000 | 1.0000 |
| inventory | FOODS_2_128_WI_2_validation | I2 | TRPO Q_max-free | VPG+GAE Q_max-free | 0.0006 | 0.0022 | 7.4719 | 0.8200 | 0.0440 |
| inventory | FOODS_2_128_WI_2_validation | I1 | TRPO Q_max-free | REINFORCE Q_max-free (loo_time) | 0.0002 | 0.0017 | 3.6999 | 0.8400 | 0.0375 |
| inventory | FOODS_2_128_WI_2_validation | I2 | TRPO Q_max-free | REINFORCE Q_max-free (loo_time) | 0.0002 | 0.9097 | 0.2070 | 0.0400 | 1.0000 |
| inventory | FOODS_2_128_WI_2_validation | I1 | TRPO Q_max-free | REINFORCE Q_max-free (none) | 0.0003 | 0.0173 | 2.6235 | 0.6400 | 0.2761 |
| inventory | FOODS_2_128_WI_2_validation | I2 | TRPO Q_max-free | REINFORCE Q_max-free (none) | 0.0058 | 0.0452 | 4.5528 | 0.5400 | 0.6322 |
| inventory | FOODS_2_128_WI_2_validation | I1 | TRPO Q_max-free | TRPO capped 7mu | 1.0000 | 1.0000 | -0.0000 | 0.0000 | 1.0000 |
| inventory | FOODS_2_128_WI_2_validation | I2 | TRPO Q_max-free | TRPO capped 7mu | 0.7616 | 1.0000 | 0.0000 | 0.0100 | 1.0000 |
| inventory | HOBBIES_1_371_WI_1_validation | I1 | TRPO Q_max-free | VPG+GAE Q_max-free | 0.1405 | 0.5708 | 1.2680 | 0.1600 | 1.0000 |
| inventory | HOBBIES_1_371_WI_1_validation | I2 | TRPO Q_max-free | VPG+GAE Q_max-free | 0.2123 | 0.0036 | 6.1574 | 0.7800 | 0.0686 |
| inventory | HOBBIES_1_371_WI_1_validation | I1 | TRPO Q_max-free | REINFORCE Q_max-free (loo_time) | 0.0073 | 0.2123 | 1.8378 | 0.3400 | 1.0000 |
| inventory | HOBBIES_1_371_WI_1_validation | I2 | TRPO Q_max-free | REINFORCE Q_max-free (loo_time) | 0.0173 | 0.0058 | 5.1125 | 0.7400 | 0.0985 |
| inventory | HOBBIES_1_371_WI_1_validation | I1 | TRPO Q_max-free | REINFORCE Q_max-free (none) | 0.0757 | 0.0173 | 6.1735 | 0.6400 | 0.2761 |
| inventory | HOBBIES_1_371_WI_1_validation | I2 | TRPO Q_max-free | REINFORCE Q_max-free (none) | 0.0028 | 0.3447 | 2.8054 | 0.2600 | 1.0000 |
| inventory | HOBBIES_1_371_WI_1_validation | I1 | TRPO Q_max-free | TRPO capped 7mu | 0.6776 | 1.0000 | -0.1626 | 0.0000 | 1.0000 |
| inventory | HOBBIES_1_371_WI_1_validation | I2 | TRPO Q_max-free | TRPO capped 7mu | 0.6232 | 0.3847 | 2.3152 | 0.2400 | 1.0000 |
