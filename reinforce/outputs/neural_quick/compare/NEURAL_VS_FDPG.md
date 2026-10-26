# Neural REINFORCE vs equal-budget comparison methods and FDPG

Neural rows: R-I4 / R-T3 (PyTorch MLP). `cm-spsa` / `cm-random`: equal-rollout-budget derivative-free methods on the same network (mean policy), same seeds and initial parameters as REINFORCE. FDPG rows: FDPG's saved thetas (best estimator by validation) re-evaluated in this environment (Q_max = 7 mu for inventory, same windows and normaliser). Validation J is in norm units; test is profit per 90-day window (inventory) or contiguous cumulative return (TECHM).

| domain | key | method | n_seeds | val_J | val_sd | test | test_sd | test_unit |
|---|---|---|---|---|---|---|---|---|
| inventory | FOODS_3_124_CA_3_validation | neural reinforce-none | 2 | 2.7556 | 0.1081 | 734.3031 | 8.1649 | profit/window |
| inventory | FOODS_3_124_CA_3_validation | neural reinforce-loo_time | 2 | 2.9513 | 0.0091 | 778.5885 | 5.1152 | profit/window |
| inventory | FOODS_3_124_CA_3_validation | neural reinforce-linear | 2 | 2.9286 | 0.0304 | 774.6197 | 0.0842 | profit/window |
| inventory | FOODS_3_124_CA_3_validation | neural cm-spsa | 2 | 2.8351 | 0.1308 | 751.4713 | 42.9157 | profit/window |
| inventory | FOODS_3_124_CA_3_validation | neural cm-random | 2 | 2.7527 | 0.0206 | 736.8188 | 21.9744 | profit/window |
| inventory | FOODS_3_124_CA_3_validation | FDPG F-I1 (spsa) | 10 | 2.8844 | 0.0002 | 758.5631 | 0.1073 | profit/window |
| inventory | FOODS_3_124_CA_3_validation | FDPG F-I2 (spsa) | 10 | 3.1065 | 0.0231 | 808.7173 | 10.1316 | profit/window |
| inventory | FOODS_3_124_CA_3_validation | FDPG F-I3 (central_nocrn) | 10 | 2.9111 | 0.0268 | 766.9939 | 12.0564 | profit/window |
| inventory | FOODS_2_128_WI_2_validation | neural reinforce-none | 2 | 0.7156 | 0.0362 | 163.7786 | 4.0170 | profit/window |
| inventory | FOODS_2_128_WI_2_validation | neural reinforce-loo_time | 2 | 0.7251 | 0.1188 | 161.7002 | 15.7078 | profit/window |
| inventory | FOODS_2_128_WI_2_validation | neural reinforce-linear | 2 | 0.7750 | 0.0280 | 166.9320 | 3.5198 | profit/window |
| inventory | FOODS_2_128_WI_2_validation | neural cm-spsa | 2 | 0.6718 | 0.0434 | 151.4150 | 2.9525 | profit/window |
| inventory | FOODS_2_128_WI_2_validation | neural cm-random | 2 | 0.6661 | 0.0471 | 159.0832 | 2.5258 | profit/window |
| inventory | FOODS_2_128_WI_2_validation | FDPG F-I1 (forward) | 10 | 0.8330 | 0.0000 | 170.4979 | 0.0186 | profit/window |
| inventory | FOODS_2_128_WI_2_validation | FDPG F-I2 (regression) | 10 | 0.8319 | 0.0103 | 171.6441 | 2.0829 | profit/window |
| inventory | FOODS_2_128_WI_2_validation | FDPG F-I3 (central) | 10 | 0.7790 | 0.0137 | 166.9415 | 0.9844 | profit/window |
| inventory | HOBBIES_1_371_WI_1_validation | neural reinforce-none | 2 | 3.1250 | 0.2148 | 95.1260 | 0.7648 | profit/window |
| inventory | HOBBIES_1_371_WI_1_validation | neural reinforce-loo_time | 2 | 2.9854 | 0.0404 | 100.0388 | 1.1137 | profit/window |
| inventory | HOBBIES_1_371_WI_1_validation | neural reinforce-linear | 2 | 3.0264 | 0.0097 | 98.1685 | 5.0832 | profit/window |
| inventory | HOBBIES_1_371_WI_1_validation | neural cm-spsa | 2 | 2.8467 | 0.3792 | 76.7113 | 12.0669 | profit/window |
| inventory | HOBBIES_1_371_WI_1_validation | neural cm-random | 2 | 2.9341 | 0.0391 | 107.8826 | 2.6677 | profit/window |
| inventory | HOBBIES_1_371_WI_1_validation | FDPG F-I1 (forward) | 10 | 3.0422 | 0.0002 | 94.8806 | 0.2749 | profit/window |
| inventory | HOBBIES_1_371_WI_1_validation | FDPG F-I2 (spsa_nocrn) | 10 | 3.3830 | 0.0627 | 101.9020 | 5.5491 | profit/window |
| inventory | HOBBIES_1_371_WI_1_validation | FDPG F-I3 (spsa_nocrn) | 10 | 3.2839 | 0.0506 | 98.4598 | 3.4940 | profit/window |
| trading | TECHM | neural reinforce-none | 2 | 0.8774 | 0.1108 | 0.3751 | 0.0620 | cum. return |
| trading | TECHM | neural reinforce-loo_time | 2 | 0.8906 | 0.1445 | 0.2526 | 0.2866 | cum. return |
| trading | TECHM | neural reinforce-linear | 2 | 0.9484 | 0.0668 | 0.6564 | 0.1830 | cum. return |
| trading | TECHM | neural cm-spsa | 2 | 0.6002 | 0.0704 | 0.7624 | 0.2284 | cum. return |
| trading | TECHM | neural cm-random | 2 | 0.8887 | 0.0715 | 0.7008 | 0.0710 | cum. return |
| trading | TECHM | FDPG F-T1 (spsa_nocrn) | 10 | 0.8536 | 0.1112 | 0.6891 | 0.1962 | cum. return |
| trading | TECHM | FDPG F-T2 (regression) | 10 | 0.6356 | 0.0000 | 0.9885 | 0.0000 | cum. return |

## Tests: best neural variant on validation vs each other row (Mann-Whitney; paired Wilcoxon where seeds are matched)

| domain | key | reference | vs | val_diff | test_diff | test_p | test_p_paired_wilcoxon | cliffs_test | test_p_holm |
|---|---|---|---|---|---|---|---|---|---|
| inventory | FOODS_3_124_CA_3_validation | neural reinforce-loo_time (best val) | neural reinforce-none | 0.1957 | 44.2854 | 0.3333 | 0.5000 | 1.0000 | 1.0000 |
| inventory | FOODS_3_124_CA_3_validation | neural reinforce-loo_time (best val) | neural reinforce-linear | 0.0226 | 3.9688 | 0.3333 | 0.5000 | 1.0000 | 1.0000 |
| inventory | FOODS_3_124_CA_3_validation | neural reinforce-loo_time (best val) | neural cm-spsa | 0.1162 | 27.1173 | 0.6667 | 1.0000 | 0.5000 | 1.0000 |
| inventory | FOODS_3_124_CA_3_validation | neural reinforce-loo_time (best val) | neural cm-random | 0.1986 | 41.7698 | 0.3333 | 0.5000 | 1.0000 | 1.0000 |
| inventory | FOODS_3_124_CA_3_validation | neural reinforce-loo_time (best val) | FDPG F-I1 (spsa) | 0.0669 | 20.0255 | 0.0303 | nan | 1.0000 | 0.8182 |
| inventory | FOODS_3_124_CA_3_validation | neural reinforce-loo_time (best val) | FDPG F-I2 (spsa) | -0.1552 | -30.1288 | 0.0303 | nan | -1.0000 | 0.8182 |
| inventory | FOODS_3_124_CA_3_validation | neural reinforce-loo_time (best val) | FDPG F-I3 (central_nocrn) | 0.0401 | 11.5947 | 0.3636 | nan | 0.5000 | 1.0000 |
| inventory | FOODS_2_128_WI_2_validation | neural reinforce-linear (best val) | neural reinforce-none | 0.0594 | 3.1535 | 0.6667 | 0.5000 | 0.5000 | 1.0000 |
| inventory | FOODS_2_128_WI_2_validation | neural reinforce-linear (best val) | neural reinforce-loo_time | 0.0499 | 5.2319 | 1.0000 | 1.0000 | 0.0000 | 1.0000 |
| inventory | FOODS_2_128_WI_2_validation | neural reinforce-linear (best val) | neural cm-spsa | 0.1032 | 15.5170 | 0.3333 | 0.5000 | 1.0000 | 1.0000 |
| inventory | FOODS_2_128_WI_2_validation | neural reinforce-linear (best val) | neural cm-random | 0.1089 | 7.8488 | 0.3333 | 0.5000 | 1.0000 | 1.0000 |
| inventory | FOODS_2_128_WI_2_validation | neural reinforce-linear (best val) | FDPG F-I1 (forward) | -0.0580 | -3.5658 | 0.0303 | nan | -1.0000 | 0.8182 |
| inventory | FOODS_2_128_WI_2_validation | neural reinforce-linear (best val) | FDPG F-I2 (regression) | -0.0569 | -4.7121 | 0.0606 | nan | -0.9000 | 1.0000 |
| inventory | FOODS_2_128_WI_2_validation | neural reinforce-linear (best val) | FDPG F-I3 (central) | -0.0040 | -0.0094 | 1.0000 | nan | 0.0000 | 1.0000 |
| inventory | HOBBIES_1_371_WI_1_validation | neural reinforce-none (best val) | neural reinforce-loo_time | 0.1396 | -4.9128 | 0.3333 | 0.5000 | -1.0000 | 1.0000 |
| inventory | HOBBIES_1_371_WI_1_validation | neural reinforce-none (best val) | neural reinforce-linear | 0.0986 | -3.0424 | 1.0000 | 1.0000 | 0.0000 | 1.0000 |
| inventory | HOBBIES_1_371_WI_1_validation | neural reinforce-none (best val) | neural cm-spsa | 0.2783 | 18.4148 | 0.3333 | 0.5000 | 1.0000 | 1.0000 |
| inventory | HOBBIES_1_371_WI_1_validation | neural reinforce-none (best val) | neural cm-random | 0.1909 | -12.7566 | 0.3333 | 0.5000 | -1.0000 | 1.0000 |
| inventory | HOBBIES_1_371_WI_1_validation | neural reinforce-none (best val) | FDPG F-I1 (forward) | 0.0828 | 0.2455 | 0.9091 | nan | 0.1000 | 1.0000 |
| inventory | HOBBIES_1_371_WI_1_validation | neural reinforce-none (best val) | FDPG F-I2 (spsa_nocrn) | -0.2580 | -6.7759 | 0.1212 | nan | -0.8000 | 1.0000 |
| inventory | HOBBIES_1_371_WI_1_validation | neural reinforce-none (best val) | FDPG F-I3 (spsa_nocrn) | -0.1589 | -3.3337 | 0.1818 | nan | -0.7000 | 1.0000 |
| trading | TECHM | neural reinforce-linear (best val) | neural reinforce-none | 0.0710 | 0.2813 | 0.3333 | 0.5000 | 1.0000 | 1.0000 |
| trading | TECHM | neural reinforce-linear (best val) | neural reinforce-loo_time | 0.0578 | 0.4038 | 0.3333 | 0.5000 | 1.0000 | 1.0000 |
| trading | TECHM | neural reinforce-linear (best val) | neural cm-spsa | 0.3482 | -0.1060 | 0.6667 | 1.0000 | -0.5000 | 1.0000 |
| trading | TECHM | neural reinforce-linear (best val) | neural cm-random | 0.0597 | -0.0445 | 1.0000 | 1.0000 | 0.0000 | 1.0000 |
| trading | TECHM | neural reinforce-linear (best val) | FDPG F-T1 (spsa_nocrn) | 0.0948 | -0.0328 | 0.7576 | nan | -0.2000 | 1.0000 |
| trading | TECHM | neural reinforce-linear (best val) | FDPG F-T2 (regression) | 0.3129 | -0.3322 | 0.0303 | nan | -1.0000 | 0.8182 |

Caveats: FDPG policy families differ (F-I2 is (s,S), F-I3 calendar base-stock, F-T1 sigmoid position); tuning effort is not matched across projects; Nelder-Mead and central FD were dropped for the ~250-parameter network (infeasible at equal budget).
