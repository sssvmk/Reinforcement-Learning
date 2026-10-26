# Neural (R-I4, R-T3) vs linear REINFORCE policies and equal-budget comparison methods on the neural policy

| key | policy | baseline | n_seeds | val_J_mean | val_J_sd | test_mean | test_ci_lo | test_ci_hi | test_unit |
|---|---|---|---|---|---|---|---|---|---|
| FOODS_3_124_CA_3_validation | I4 | cm-random | 10 | 2.8234 | 0.0445 | 754.8134 | 749.3035 | 760.2557 | profit/window |
| FOODS_3_124_CA_3_validation | I4 | cm-spsa | 10 | 2.4889 | 0.5899 | 668.5910 | 553.2258 | 734.9400 | profit/window |
| FOODS_3_124_CA_3_validation | I4 | linear | 10 | 3.0588 | 0.0326 | 802.7944 | 799.0426 | 806.9286 | profit/window |
| FOODS_3_124_CA_3_validation | I4 | loo_time | 10 | 3.0456 | 0.0285 | 798.1903 | 791.9275 | 803.0669 | profit/window |
| FOODS_3_124_CA_3_validation | I4 | none | 10 | 2.8907 | 0.2942 | 767.1518 | 735.2477 | 792.5214 | profit/window |
| FOODS_3_124_CA_3_validation | I2 | linear | 10 | 2.9967 | 0.0272 | 781.7102 | 775.8752 | 787.5360 | profit/window |
| FOODS_3_124_CA_3_validation | I2 | loo_time | 10 | 2.9840 | 0.0265 | 780.6954 | 773.7456 | 787.7051 | profit/window |
| FOODS_3_124_CA_3_validation | I2 | none | 10 | 2.9004 | 0.1651 | 767.8543 | 750.5614 | 780.7869 | profit/window |
| FOODS_3_124_CA_3_validation | I3 | linear | 10 | 3.0483 | 0.0643 | 797.8247 | 790.6538 | 804.4370 | profit/window |
| FOODS_3_124_CA_3_validation | I3 | loo_time | 10 | 3.0502 | 0.0674 | 797.8463 | 788.8554 | 805.3878 | profit/window |
| FOODS_3_124_CA_3_validation | I3 | none | 10 | 3.0403 | 0.0528 | 795.8263 | 788.9792 | 801.8361 | profit/window |
| FOODS_2_128_WI_2_validation | I4 | cm-random | 10 | 0.6281 | 0.0817 | 147.2333 | 139.6374 | 154.4008 | profit/window |
| FOODS_2_128_WI_2_validation | I4 | cm-spsa | 10 | 0.6611 | 0.0998 | 152.7932 | 147.9946 | 157.2036 | profit/window |
| FOODS_2_128_WI_2_validation | I4 | linear | 10 | 0.7569 | 0.0728 | 164.3311 | 158.8004 | 169.5358 | profit/window |
| FOODS_2_128_WI_2_validation | I4 | loo_time | 10 | 0.7294 | 0.0909 | 163.2760 | 153.3536 | 171.3055 | profit/window |
| FOODS_2_128_WI_2_validation | I4 | none | 10 | 0.7392 | 0.0492 | 163.9526 | 158.3120 | 168.2441 | profit/window |
| FOODS_2_128_WI_2_validation | I2 | linear | 10 | 0.7125 | 0.0850 | 164.7041 | 158.0858 | 169.7849 | profit/window |
| FOODS_2_128_WI_2_validation | I2 | loo_time | 10 | 0.7237 | 0.0869 | 164.4056 | 158.3759 | 169.0259 | profit/window |
| FOODS_2_128_WI_2_validation | I2 | none | 10 | 0.7457 | 0.0502 | 163.9962 | 161.2943 | 167.0085 | profit/window |
| FOODS_2_128_WI_2_validation | I3 | linear | 10 | -0.0652 | 1.9264 | 83.2612 | -34.5693 | 164.5342 | profit/window |
| FOODS_2_128_WI_2_validation | I3 | loo_time | 10 | -0.0767 | 1.9374 | 82.8436 | -33.8114 | 162.9411 | profit/window |
| FOODS_2_128_WI_2_validation | I3 | none | 10 | -0.0519 | 1.9317 | 86.9760 | -32.2157 | 168.9924 | profit/window |
| HOBBIES_1_371_WI_1_validation | I4 | cm-random | 10 | 3.0861 | 0.0613 | 108.8929 | 108.2198 | 109.4626 | profit/window |
| HOBBIES_1_371_WI_1_validation | I4 | cm-spsa | 10 | 3.0584 | 0.2235 | 95.5612 | 87.0107 | 102.7800 | profit/window |
| HOBBIES_1_371_WI_1_validation | I4 | linear | 10 | 3.3382 | 0.0372 | 101.3597 | 99.2619 | 103.4429 | profit/window |
| HOBBIES_1_371_WI_1_validation | I4 | loo_time | 10 | 3.3001 | 0.0286 | 101.5191 | 99.4431 | 103.6048 | profit/window |
| HOBBIES_1_371_WI_1_validation | I4 | none | 10 | 3.2933 | 0.0966 | 100.0766 | 95.9673 | 103.4059 | profit/window |
| HOBBIES_1_371_WI_1_validation | I2 | linear | 10 | 3.2406 | 0.0456 | 99.5772 | 94.3239 | 103.4995 | profit/window |
| HOBBIES_1_371_WI_1_validation | I2 | loo_time | 10 | 3.2135 | 0.1597 | 96.0810 | 88.6846 | 101.4588 | profit/window |
| HOBBIES_1_371_WI_1_validation | I2 | none | 10 | 3.2422 | 0.1518 | 98.9337 | 96.6651 | 101.1705 | profit/window |
| HOBBIES_1_371_WI_1_validation | I3 | linear | 10 | 3.2673 | 0.1114 | 103.7958 | 100.8617 | 106.2969 | profit/window |
| HOBBIES_1_371_WI_1_validation | I3 | loo_time | 10 | 3.2361 | 0.1628 | 104.4616 | 102.4152 | 106.3110 | profit/window |
| HOBBIES_1_371_WI_1_validation | I3 | none | 10 | 3.1930 | 0.1852 | 98.1303 | 92.9945 | 102.9854 | profit/window |
| TECHM | T3 | cm-random | 10 | 0.9501 | 0.0665 | 0.7889 | 0.6470 | 0.9389 | cum. return |
| TECHM | T3 | cm-spsa | 10 | 0.7425 | 0.0599 | 0.8660 | 0.8356 | 0.9009 | cum. return |
| TECHM | T3 | linear | 10 | 1.0230 | 0.0913 | 0.6208 | 0.5021 | 0.7453 | cum. return |
| TECHM | T3 | loo_time | 10 | 1.0332 | 0.0740 | 0.5706 | 0.3782 | 0.7647 | cum. return |
| TECHM | T3 | none | 10 | 1.0475 | 0.0735 | 0.5500 | 0.3566 | 0.7350 | cum. return |
| TECHM | T1 | linear | 10 | 0.9773 | 0.1112 | 0.6508 | 0.4819 | 0.8021 | cum. return |
| TECHM | T1 | loo_time | 10 | 0.9318 | 0.1181 | 0.8131 | 0.6874 | 0.9231 | cum. return |
| TECHM | T1 | none | 10 | 0.9382 | 0.0814 | 0.7856 | 0.5979 | 0.9713 | cum. return |
| TECHM | T2 | linear | 10 | 0.9541 | 0.0638 | 0.6883 | 0.5547 | 0.8379 | cum. return |
| TECHM | T2 | loo_time | 10 | 0.9555 | 0.0968 | 0.7559 | 0.6624 | 0.8388 | cum. return |
| TECHM | T2 | none | 10 | 0.9647 | 0.0459 | 0.6158 | 0.5293 | 0.7088 | cum. return |

## Mann-Whitney: neural vs linear (loo_time baseline, over seeds)

| key | neural | vs | val_p | test_p | test_diff | cliffs_test | test_p_holm |
|---|---|---|---|---|---|---|---|
| FOODS_3_124_CA_3_validation | I4 | I2 | 0.0002 | 0.0036 | 17.4949 | 0.7800 | 0.0289 |
| FOODS_3_124_CA_3_validation | I4 | I3 | 0.3847 | 0.9097 | 0.3440 | -0.0400 | 0.9454 |
| FOODS_2_128_WI_2_validation | I4 | I2 | 0.4274 | 0.4727 | -1.1296 | 0.2000 | 0.9454 |
| FOODS_2_128_WI_2_validation | I4 | I3 | 0.7337 | 0.0640 | 80.4324 | 0.5000 | 0.3841 |
| HOBBIES_1_371_WI_1_validation | I4 | I2 | 0.0757 | 0.1859 | 5.4381 | 0.3600 | 0.5576 |
| HOBBIES_1_371_WI_1_validation | I4 | I3 | 0.6776 | 0.0890 | -2.9425 | -0.4600 | 0.4449 |
| TECHM | T3 | T1 | 0.1859 | 0.0539 | -0.2425 | -0.5200 | 0.3773 |
| TECHM | T3 | T2 | 0.1620 | 0.1212 | -0.1853 | -0.4200 | 0.4849 |

## Sensitivity (exploratory; fewer seeds and episodes than the linear study)

| policy | variable | value | n | val_J_mean | val_J_sd |
|---|---|---|---|---|---|
| I4 | clip | 1.0000 | 2 | 3.0284 | 0.0036 |
| I4 | clip | 5.0000 | 2 | 3.0105 | 0.0143 |
| I4 | clip | 25.0000 | 2 | 3.0148 | 0.0361 |
| I4 | gamma | 0.9000 | 2 | 3.0119 | 0.0380 |
| I4 | gamma | 0.9900 | 2 | 3.0105 | 0.0143 |
| I4 | gamma | 1.0000 | 2 | 2.9626 | 0.0982 |
| I4 | horizon | 30.0000 | 2 | 2.6077 | 0.0096 |
| I4 | horizon | 90.0000 | 2 | 3.0105 | 0.0143 |
| I4 | horizon | 180.0000 | 2 | 4.1320 | 0.0317 |
| I4 | lr | 0.0100 | 2 | 2.9510 | 0.0207 |
| I4 | lr | 0.0300 | 2 | 3.0105 | 0.0143 |
| I4 | lr | 0.0900 | 2 | 3.0897 | 0.0154 |
| I4 | sigma0 | 0.2500 | 2 | 3.0097 | 0.0194 |
| I4 | sigma0 | 0.5000 | 2 | 3.0105 | 0.0143 |
| I4 | sigma0 | 1.0000 | 2 | 3.0328 | 0.0517 |
| T3 | clip | 1.0000 | 2 | 1.0700 | 0.1317 |
| T3 | clip | 5.0000 | 2 | 1.0880 | 0.0089 |
| T3 | clip | 25.0000 | 2 | 1.0603 | 0.0015 |
| T3 | gamma | 0.9000 | 2 | 1.0160 | 0.0112 |
| T3 | gamma | 0.9900 | 2 | 1.0209 | 0.0869 |
| T3 | gamma | 1.0000 | 2 | 1.0103 | 0.0170 |
| T3 | horizon | 125.0000 | 2 | 0.4902 | 0.0389 |
| T3 | horizon | 250.0000 | 2 | 1.0880 | 0.0089 |
| T3 | horizon | 500.0000 | 2 | 1.8754 | 0.2536 |
| T3 | lr | 0.0100 | 2 | 1.0049 | 0.0751 |
| T3 | lr | 0.0300 | 2 | 1.0880 | 0.0089 |
| T3 | lr | 0.0900 | 2 | 1.0595 | 0.0245 |
