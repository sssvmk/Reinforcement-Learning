# Neural REINFORCE vs equal-budget comparison methods and FDPG

Neural rows: R-I4 / R-T3 (PyTorch MLP). `cm-spsa` / `cm-random`: equal-rollout-budget derivative-free methods on the same network (mean policy), same seeds and initial parameters as REINFORCE. FDPG rows: FDPG's saved thetas (best estimator by validation) re-evaluated in this environment (Q_max = 7 mu for inventory, same windows and normaliser). Validation J is in norm units; test is profit per 90-day window (inventory) or contiguous cumulative return (TECHM).

| domain | key | method | n_seeds | val_J | val_sd | test | test_sd | test_unit |
|---|---|---|---|---|---|---|---|---|
| inventory | FOODS_3_124_CA_3_validation | neural reinforce-none | 10 | 2.8907 | 0.2942 | 767.1518 | 48.4024 | profit/window |
| inventory | FOODS_3_124_CA_3_validation | neural reinforce-loo_time | 10 | 3.0456 | 0.0285 | 798.1903 | 9.5150 | profit/window |
| inventory | FOODS_3_124_CA_3_validation | neural reinforce-linear | 10 | 3.0588 | 0.0326 | 802.7944 | 6.7247 | profit/window |
| inventory | FOODS_3_124_CA_3_validation | neural cm-spsa | 10 | 2.4889 | 0.5899 | 668.5910 | 176.2537 | profit/window |
| inventory | FOODS_3_124_CA_3_validation | neural cm-random | 10 | 2.8234 | 0.0445 | 754.8134 | 9.4807 | profit/window |
| inventory | FOODS_3_124_CA_3_validation | FDPG F-I1 (spsa) | 10 | 2.8844 | 0.0002 | 758.5631 | 0.1073 | profit/window |
| inventory | FOODS_3_124_CA_3_validation | FDPG F-I2 (spsa) | 10 | 3.1065 | 0.0231 | 808.7173 | 10.1316 | profit/window |
| inventory | FOODS_3_124_CA_3_validation | FDPG F-I3 (central_nocrn) | 10 | 2.9111 | 0.0268 | 766.9939 | 12.0564 | profit/window |
| inventory | FOODS_2_128_WI_2_validation | neural reinforce-none | 10 | 0.7392 | 0.0492 | 163.9526 | 8.6455 | profit/window |
| inventory | FOODS_2_128_WI_2_validation | neural reinforce-loo_time | 10 | 0.7294 | 0.0909 | 163.2760 | 15.7220 | profit/window |
| inventory | FOODS_2_128_WI_2_validation | neural reinforce-linear | 10 | 0.7569 | 0.0728 | 164.3311 | 9.3137 | profit/window |
| inventory | FOODS_2_128_WI_2_validation | neural cm-spsa | 10 | 0.6611 | 0.0998 | 152.7932 | 8.1374 | profit/window |
| inventory | FOODS_2_128_WI_2_validation | neural cm-random | 10 | 0.6281 | 0.0817 | 147.2333 | 12.7460 | profit/window |
| inventory | FOODS_2_128_WI_2_validation | FDPG F-I1 (forward) | 10 | 0.8330 | 0.0000 | 170.4979 | 0.0186 | profit/window |
| inventory | FOODS_2_128_WI_2_validation | FDPG F-I2 (regression) | 10 | 0.8319 | 0.0103 | 171.6441 | 2.0829 | profit/window |
| inventory | FOODS_2_128_WI_2_validation | FDPG F-I3 (central) | 10 | 0.7790 | 0.0137 | 166.9415 | 0.9844 | profit/window |
| inventory | HOBBIES_1_371_WI_1_validation | neural reinforce-none | 10 | 3.2933 | 0.0966 | 100.0766 | 6.3362 | profit/window |
| inventory | HOBBIES_1_371_WI_1_validation | neural reinforce-loo_time | 10 | 3.3001 | 0.0286 | 101.5191 | 3.7098 | profit/window |
| inventory | HOBBIES_1_371_WI_1_validation | neural reinforce-linear | 10 | 3.3382 | 0.0372 | 101.3597 | 3.5419 | profit/window |
| inventory | HOBBIES_1_371_WI_1_validation | neural cm-spsa | 10 | 3.0584 | 0.2235 | 95.5612 | 13.6949 | profit/window |
| inventory | HOBBIES_1_371_WI_1_validation | neural cm-random | 10 | 3.0861 | 0.0613 | 108.8929 | 1.1070 | profit/window |
| inventory | HOBBIES_1_371_WI_1_validation | FDPG F-I1 (forward) | 10 | 3.0422 | 0.0002 | 94.8806 | 0.2749 | profit/window |
| inventory | HOBBIES_1_371_WI_1_validation | FDPG F-I2 (spsa_nocrn) | 10 | 3.3830 | 0.0627 | 101.9020 | 5.5491 | profit/window |
| inventory | HOBBIES_1_371_WI_1_validation | FDPG F-I3 (spsa_nocrn) | 10 | 3.2839 | 0.0506 | 98.4598 | 3.4940 | profit/window |
| trading | TECHM | neural reinforce-none | 10 | 1.0475 | 0.0735 | 0.5500 | 0.3281 | cum. return |
| trading | TECHM | neural reinforce-loo_time | 10 | 1.0332 | 0.0740 | 0.5706 | 0.3354 | cum. return |
| trading | TECHM | neural reinforce-linear | 10 | 1.0230 | 0.0913 | 0.6208 | 0.2058 | cum. return |
| trading | TECHM | neural cm-spsa | 10 | 0.7425 | 0.0599 | 0.8660 | 0.0562 | cum. return |
| trading | TECHM | neural cm-random | 10 | 0.9501 | 0.0665 | 0.7889 | 0.2526 | cum. return |
| trading | TECHM | FDPG F-T1 (spsa_nocrn) | 10 | 0.8536 | 0.1112 | 0.6891 | 0.1962 | cum. return |
| trading | TECHM | FDPG F-T2 (regression) | 10 | 0.6356 | 0.0000 | 0.9885 | 0.0000 | cum. return |

## Tests: best neural variant on validation vs each other row (Mann-Whitney; paired Wilcoxon where seeds are matched)

| domain | key | reference | vs | val_diff | test_diff | test_p | test_p_paired_wilcoxon | cliffs_test | test_p_holm |
|---|---|---|---|---|---|---|---|---|---|
| inventory | FOODS_3_124_CA_3_validation | neural reinforce-linear (best val) | neural reinforce-none | 0.1681 | 35.6426 | 0.0140 | 0.0371 | 0.6600 | 0.2523 |
| inventory | FOODS_3_124_CA_3_validation | neural reinforce-linear (best val) | neural reinforce-loo_time | 0.0132 | 4.6041 | 0.3847 | 0.2324 | 0.2400 | 1.0000 |
| inventory | FOODS_3_124_CA_3_validation | neural reinforce-linear (best val) | neural cm-spsa | 0.5699 | 134.2034 | 0.0002 | 0.0020 | 1.0000 | 0.0049 |
| inventory | FOODS_3_124_CA_3_validation | neural reinforce-linear (best val) | neural cm-random | 0.2354 | 47.9809 | 0.0002 | 0.0020 | 1.0000 | 0.0049 |
| inventory | FOODS_3_124_CA_3_validation | neural reinforce-linear (best val) | FDPG F-I1 (spsa) | 0.1744 | 44.2313 | 0.0002 | nan | 1.0000 | 0.0049 |
| inventory | FOODS_3_124_CA_3_validation | neural reinforce-linear (best val) | FDPG F-I2 (spsa) | -0.0477 | -5.9230 | 0.0257 | nan | -0.6000 | 0.4120 |
| inventory | FOODS_3_124_CA_3_validation | neural reinforce-linear (best val) | FDPG F-I3 (central_nocrn) | 0.1477 | 35.8005 | 0.0002 | nan | 1.0000 | 0.0049 |
| inventory | FOODS_2_128_WI_2_validation | neural reinforce-linear (best val) | neural reinforce-none | 0.0178 | 0.3785 | 0.7913 | 0.9219 | 0.0800 | 1.0000 |
| inventory | FOODS_2_128_WI_2_validation | neural reinforce-linear (best val) | neural reinforce-loo_time | 0.0276 | 1.0551 | 0.6776 | 0.6953 | -0.1200 | 1.0000 |
| inventory | FOODS_2_128_WI_2_validation | neural reinforce-linear (best val) | neural cm-spsa | 0.0959 | 11.5379 | 0.0211 | 0.0488 | 0.6200 | 0.3593 |
| inventory | FOODS_2_128_WI_2_validation | neural reinforce-linear (best val) | neural cm-random | 0.1289 | 17.0978 | 0.0091 | 0.0020 | 0.7000 | 0.1822 |
| inventory | FOODS_2_128_WI_2_validation | neural reinforce-linear (best val) | FDPG F-I1 (forward) | -0.0761 | -6.1668 | 0.1405 | nan | -0.4000 | 1.0000 |
| inventory | FOODS_2_128_WI_2_validation | neural reinforce-linear (best val) | FDPG F-I2 (regression) | -0.0749 | -7.3131 | 0.0640 | nan | -0.5000 | 0.9603 |
| inventory | FOODS_2_128_WI_2_validation | neural reinforce-linear (best val) | FDPG F-I3 (central) | -0.0221 | -2.6104 | 1.0000 | nan | 0.0000 | 1.0000 |
| inventory | HOBBIES_1_371_WI_1_validation | neural reinforce-linear (best val) | neural reinforce-none | 0.0449 | 1.2831 | 0.9698 | 0.8457 | 0.0200 | 1.0000 |
| inventory | HOBBIES_1_371_WI_1_validation | neural reinforce-linear (best val) | neural reinforce-loo_time | 0.0381 | -0.1595 | 1.0000 | 0.7695 | 0.0000 | 1.0000 |
| inventory | HOBBIES_1_371_WI_1_validation | neural reinforce-linear (best val) | neural cm-spsa | 0.2799 | 5.7985 | 0.4274 | 0.2754 | 0.2200 | 1.0000 |
| inventory | HOBBIES_1_371_WI_1_validation | neural reinforce-linear (best val) | neural cm-random | 0.2521 | -7.5332 | 0.0002 | 0.0020 | -0.9800 | 0.0054 |
| inventory | HOBBIES_1_371_WI_1_validation | neural reinforce-linear (best val) | FDPG F-I1 (forward) | 0.2961 | 6.4791 | 0.0002 | nan | 1.0000 | 0.0049 |
| inventory | HOBBIES_1_371_WI_1_validation | neural reinforce-linear (best val) | FDPG F-I2 (spsa_nocrn) | -0.0447 | -0.5423 | 0.9097 | nan | -0.0400 | 1.0000 |
| inventory | HOBBIES_1_371_WI_1_validation | neural reinforce-linear (best val) | FDPG F-I3 (spsa_nocrn) | 0.0543 | 2.8999 | 0.0757 | nan | 0.4800 | 1.0000 |
| trading | TECHM | neural reinforce-none (best val) | neural reinforce-loo_time | 0.0143 | -0.0206 | 0.8501 | 1.0000 | 0.0600 | 1.0000 |
| trading | TECHM | neural reinforce-none (best val) | neural reinforce-linear | 0.0245 | -0.0708 | 1.0000 | 0.4922 | 0.0000 | 1.0000 |
| trading | TECHM | neural reinforce-none (best val) | neural cm-spsa | 0.3050 | -0.3160 | 0.0091 | 0.0137 | -0.7000 | 0.1822 |
| trading | TECHM | neural reinforce-none (best val) | neural cm-random | 0.0974 | -0.2389 | 0.1212 | 0.0488 | -0.4200 | 1.0000 |
| trading | TECHM | neural reinforce-none (best val) | FDPG F-T1 (spsa_nocrn) | 0.1939 | -0.1391 | 0.5205 | nan | -0.1800 | 1.0000 |
| trading | TECHM | neural reinforce-none (best val) | FDPG F-T2 (regression) | 0.4119 | -0.4386 | 0.0028 | nan | -0.8000 | 0.0594 |

Caveats: FDPG policy families differ (F-I2 is (s,S), F-I3 calendar base-stock, F-T1 sigmoid position); tuning effort is not matched across projects; Nelder-Mead and central FD were dropped for the ~250-parameter network (infeasible at equal budget).
