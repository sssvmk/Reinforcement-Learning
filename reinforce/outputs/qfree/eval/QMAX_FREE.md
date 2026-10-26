# Q_max-free REINFORCE (R-I1, R-I2)

All rows are evaluated in the SAME environment (no order cap, same validation / test windows, same normaliser): validation J in norm units, test profit per 90-day window. FDPG rows use FDPG's saved thetas (mean-policy mode) of the estimator that was best on validation here; FDPG's tuning effort and policy parameterisation differ, and seeds are not paired.

| key | policy | method | n_seeds | val_J_nocap_env | val_J_sd | test_profit_nocap_env | test_sd | share_orders_above_7mu |
|---|---|---|---|---|---|---|---|---|
| FOODS_3_124_CA_3_validation | I1 | REINFORCE Q_max-free (loo_time) | 10 | 2.8724 | 0.0089 | 757.3654 | 2.5985 | 0.0000 |
| FOODS_3_124_CA_3_validation | I1 | REINFORCE Q_max-free (none) | 10 | 2.8741 | 0.0180 | 754.5057 | 5.2596 | 0.0000 |
| FOODS_3_124_CA_3_validation | I1 | REINFORCE capped 7mu (loo_time) | 10 | 2.8724 | 0.0089 | 757.3654 | 2.5985 | 0.0000 |
| FOODS_3_124_CA_3_validation | I1 | REINFORCE capped 7mu (none) | 10 | 2.8741 | 0.0180 | 754.5057 | 5.2596 | 0.0000 |
| FOODS_3_124_CA_3_validation | I1 | FDPG F-I1 (spsa, by validation) | 10 | 2.8844 | 0.0002 | 758.5631 | 0.1073 | nan |
| FOODS_3_124_CA_3_validation | I2 | REINFORCE Q_max-free (loo_time) | 10 | 2.9511 | 0.1865 | 768.1764 | 27.2917 | 0.0033 |
| FOODS_3_124_CA_3_validation | I2 | REINFORCE Q_max-free (none) | 10 | 2.9417 | 0.1065 | 772.2889 | 19.4088 | 0.0011 |
| FOODS_3_124_CA_3_validation | I2 | REINFORCE capped 7mu (loo_time) | 10 | 2.9840 | 0.0265 | 780.6462 | 12.2631 | 0.0002 |
| FOODS_3_124_CA_3_validation | I2 | REINFORCE capped 7mu (none) | 10 | 2.9004 | 0.1651 | 767.8455 | 27.1921 | 0.0003 |
| FOODS_3_124_CA_3_validation | I2 | FDPG F-I2 (spsa, by validation) | 10 | 3.1065 | 0.0231 | 808.7173 | 10.1316 | nan |
| FOODS_2_128_WI_2_validation | I1 | REINFORCE Q_max-free (loo_time) | 10 | 0.7460 | 0.0247 | 165.8785 | 2.6081 | 0.0000 |
| FOODS_2_128_WI_2_validation | I1 | REINFORCE Q_max-free (none) | 10 | 0.7401 | 0.0273 | 166.9751 | 2.2693 | 0.0000 |
| FOODS_2_128_WI_2_validation | I1 | REINFORCE capped 7mu (loo_time) | 10 | 0.7460 | 0.0247 | 165.8783 | 2.6084 | 0.0000 |
| FOODS_2_128_WI_2_validation | I1 | REINFORCE capped 7mu (none) | 10 | 0.7401 | 0.0274 | 166.9839 | 2.2558 | 0.0000 |
| FOODS_2_128_WI_2_validation | I1 | FDPG F-I1 (forward, by validation) | 10 | 0.8330 | 0.0000 | 170.4979 | 0.0186 | nan |
| FOODS_2_128_WI_2_validation | I2 | REINFORCE Q_max-free (loo_time) | 10 | 0.7567 | 0.0179 | 168.7656 | 3.4536 | 0.0007 |
| FOODS_2_128_WI_2_validation | I2 | REINFORCE Q_max-free (none) | 10 | 0.7308 | 0.1229 | 159.3263 | 17.3357 | 0.0004 |
| FOODS_2_128_WI_2_validation | I2 | REINFORCE capped 7mu (loo_time) | 10 | 0.7237 | 0.0869 | 164.4056 | 9.3540 | 0.0000 |
| FOODS_2_128_WI_2_validation | I2 | REINFORCE capped 7mu (none) | 10 | 0.7457 | 0.0502 | 163.9962 | 5.0882 | 0.0000 |
| FOODS_2_128_WI_2_validation | I2 | FDPG F-I2 (regression, by validation) | 10 | 0.8319 | 0.0103 | 171.6441 | 2.0829 | nan |
| HOBBIES_1_371_WI_1_validation | I1 | REINFORCE Q_max-free (loo_time) | 10 | 3.3107 | 0.0314 | 102.6789 | 4.5103 | 0.0003 |
| HOBBIES_1_371_WI_1_validation | I1 | REINFORCE Q_max-free (none) | 10 | 3.2990 | 0.0699 | 100.0379 | 5.6372 | 0.0011 |
| HOBBIES_1_371_WI_1_validation | I1 | REINFORCE capped 7mu (loo_time) | 10 | 3.3168 | 0.0186 | 100.0184 | 4.5309 | 0.0000 |
| HOBBIES_1_371_WI_1_validation | I1 | REINFORCE capped 7mu (none) | 10 | 3.3094 | 0.0528 | 101.4993 | 4.7110 | 0.0014 |
| HOBBIES_1_371_WI_1_validation | I1 | FDPG F-I1 (forward, by validation) | 10 | 3.0422 | 0.0002 | 94.8806 | 0.2749 | nan |
| HOBBIES_1_371_WI_1_validation | I2 | REINFORCE Q_max-free (loo_time) | 10 | 3.2485 | 0.1775 | 95.2122 | 9.7591 | 0.0146 |
| HOBBIES_1_371_WI_1_validation | I2 | REINFORCE Q_max-free (none) | 10 | 3.1784 | 0.1793 | 100.3621 | 6.7383 | 0.0144 |
| HOBBIES_1_371_WI_1_validation | I2 | REINFORCE capped 7mu (loo_time) | 10 | 3.1875 | 0.1581 | 96.2553 | 11.5348 | 0.0215 |
| HOBBIES_1_371_WI_1_validation | I2 | REINFORCE capped 7mu (none) | 10 | 3.2147 | 0.1569 | 99.1925 | 4.2701 | 0.0168 |
| HOBBIES_1_371_WI_1_validation | I2 | FDPG F-I2 (spsa_nocrn, by validation) | 10 | 3.3830 | 0.0627 | 101.9117 | 5.5678 | nan |

## Mann-Whitney: Q_max-free REINFORCE (loo_time) vs each other row (over seeds)

| key | policy | vs | val_p | test_p | test_diff_(qfree_minus_other) | cliffs_test | test_p_holm |
|---|---|---|---|---|---|---|---|
| FOODS_3_124_CA_3_validation | I1 | REINFORCE Q_max-free (none) | 0.9097 | 0.1212 | 2.8597 | 0.4200 | 1.0000 |
| FOODS_3_124_CA_3_validation | I1 | REINFORCE capped 7mu (loo_time) | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I1 | REINFORCE capped 7mu (none) | 0.9097 | 0.1212 | 2.8597 | 0.4200 | 1.0000 |
| FOODS_3_124_CA_3_validation | I1 | FDPG best estimator | 0.0002 | 0.4727 | -1.1977 | -0.2000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I2 | REINFORCE Q_max-free (none) | 0.4274 | 0.8501 | -4.1125 | 0.0600 | 1.0000 |
| FOODS_3_124_CA_3_validation | I2 | REINFORCE capped 7mu (loo_time) | 0.3847 | 0.3075 | -12.4698 | -0.2800 | 1.0000 |
| FOODS_3_124_CA_3_validation | I2 | REINFORCE capped 7mu (none) | 0.0820 | 0.8798 | 0.3308 | -0.0500 | 1.0000 |
| FOODS_3_124_CA_3_validation | I2 | FDPG best estimator | 0.0002 | 0.0002 | -40.5410 | -0.9800 | 0.0057 |
| FOODS_2_128_WI_2_validation | I1 | REINFORCE Q_max-free (none) | 0.6776 | 0.4274 | -1.0965 | -0.2200 | 1.0000 |
| FOODS_2_128_WI_2_validation | I1 | REINFORCE capped 7mu (loo_time) | 0.9698 | 1.0000 | 0.0002 | 0.0000 | 1.0000 |
| FOODS_2_128_WI_2_validation | I1 | REINFORCE capped 7mu (none) | 0.6776 | 0.4274 | -1.1054 | -0.2200 | 1.0000 |
| FOODS_2_128_WI_2_validation | I1 | FDPG best estimator | 0.0002 | 0.0002 | -4.6194 | -1.0000 | 0.0044 |
| FOODS_2_128_WI_2_validation | I2 | REINFORCE Q_max-free (none) | 0.9097 | 0.0890 | 9.4392 | 0.4600 | 1.0000 |
| FOODS_2_128_WI_2_validation | I2 | REINFORCE capped 7mu (loo_time) | 0.3847 | 0.2730 | 4.3600 | 0.3000 | 1.0000 |
| FOODS_2_128_WI_2_validation | I2 | REINFORCE capped 7mu (none) | 0.5205 | 0.0376 | 4.7694 | 0.5600 | 0.7903 |
| FOODS_2_128_WI_2_validation | I2 | FDPG best estimator | 0.0002 | 0.0376 | -2.8786 | -0.5600 | 0.7903 |
| HOBBIES_1_371_WI_1_validation | I1 | REINFORCE Q_max-free (none) | 0.9698 | 0.2413 | 2.6410 | 0.3200 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I1 | REINFORCE capped 7mu (loo_time) | 0.8501 | 0.2123 | 2.6605 | 0.3400 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I1 | REINFORCE capped 7mu (none) | 0.7913 | 0.5205 | 1.1796 | 0.1800 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I1 | FDPG best estimator | 0.0002 | 0.0002 | 7.7984 | 0.9800 | 0.0057 |
| HOBBIES_1_371_WI_1_validation | I2 | REINFORCE Q_max-free (none) | 0.1620 | 0.2123 | -5.1498 | -0.3400 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I2 | REINFORCE capped 7mu (loo_time) | 0.2123 | 0.7337 | -1.0431 | -0.1000 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I2 | REINFORCE capped 7mu (none) | 0.3847 | 0.5708 | -3.9803 | -0.1600 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I2 | FDPG best estimator | 0.0257 | 0.0757 | -6.6995 | -0.4800 | 1.0000 |
