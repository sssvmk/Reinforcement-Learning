# Neural (R-I4, R-T3) vs linear REINFORCE policies

| key | policy | baseline | n_seeds | val_J_mean | val_J_sd | test_mean | test_ci_lo | test_ci_hi | test_unit |
|---|---|---|---|---|---|---|---|---|---|
| FOODS_3_124_CA_3_validation | I4 | linear | 5 | 3.0567 | 0.0365 | 802.3515 | 796.3886 | 809.9170 | profit/window |
| FOODS_3_124_CA_3_validation | I4 | loo_time | 5 | 3.0521 | 0.0362 | 796.8739 | 787.5293 | 803.7695 | profit/window |
| FOODS_3_124_CA_3_validation | I4 | none | 5 | 3.0214 | 0.0678 | 794.0413 | 779.8247 | 808.4572 | profit/window |
| FOODS_3_124_CA_3_validation | I2 | linear | 10 | 2.9967 | 0.0272 | 781.7102 | 775.8752 | 787.5360 | profit/window |
| FOODS_3_124_CA_3_validation | I2 | loo_time | 10 | 2.9840 | 0.0265 | 780.6954 | 773.7456 | 787.7051 | profit/window |
| FOODS_3_124_CA_3_validation | I2 | none | 10 | 2.9004 | 0.1651 | 767.8543 | 750.5614 | 780.7869 | profit/window |
| FOODS_3_124_CA_3_validation | I3 | linear | 10 | 3.0483 | 0.0643 | 797.8247 | 790.6538 | 804.4370 | profit/window |
| FOODS_3_124_CA_3_validation | I3 | loo_time | 10 | 3.0502 | 0.0674 | 797.8463 | 788.8554 | 805.3878 | profit/window |
| FOODS_3_124_CA_3_validation | I3 | none | 10 | 3.0403 | 0.0528 | 795.8263 | 788.9792 | 801.8361 | profit/window |
| FOODS_2_128_WI_2_validation | I4 | linear | 5 | 0.7443 | 0.0723 | 162.7115 | 154.7284 | 170.4476 | profit/window |
| FOODS_2_128_WI_2_validation | I4 | loo_time | 5 | 0.6933 | 0.1187 | 157.7025 | 139.1048 | 173.5401 | profit/window |
| FOODS_2_128_WI_2_validation | I4 | none | 5 | 0.7082 | 0.0509 | 159.9432 | 152.1322 | 167.7396 | profit/window |
| FOODS_2_128_WI_2_validation | I2 | linear | 10 | 0.7125 | 0.0850 | 164.7041 | 158.0858 | 169.7849 | profit/window |
| FOODS_2_128_WI_2_validation | I2 | loo_time | 10 | 0.7237 | 0.0869 | 164.4056 | 158.3759 | 169.0259 | profit/window |
| FOODS_2_128_WI_2_validation | I2 | none | 10 | 0.7457 | 0.0502 | 163.9962 | 161.2943 | 167.0085 | profit/window |
| FOODS_2_128_WI_2_validation | I3 | linear | 10 | -0.0652 | 1.9264 | 83.2612 | -34.5693 | 164.5342 | profit/window |
| FOODS_2_128_WI_2_validation | I3 | loo_time | 10 | -0.0767 | 1.9374 | 82.8436 | -33.8114 | 162.9411 | profit/window |
| FOODS_2_128_WI_2_validation | I3 | none | 10 | -0.0519 | 1.9317 | 86.9760 | -32.2157 | 168.9924 | profit/window |
| HOBBIES_1_371_WI_1_validation | I4 | linear | 5 | 3.3366 | 0.0228 | 101.9365 | 99.0367 | 104.8364 | profit/window |
| HOBBIES_1_371_WI_1_validation | I4 | loo_time | 5 | 3.2888 | 0.0094 | 100.6109 | 97.3551 | 103.8667 | profit/window |
| HOBBIES_1_371_WI_1_validation | I4 | none | 5 | 3.3324 | 0.0659 | 99.9886 | 96.9470 | 103.0302 | profit/window |
| HOBBIES_1_371_WI_1_validation | I2 | linear | 10 | 3.2406 | 0.0456 | 99.5772 | 94.3239 | 103.4995 | profit/window |
| HOBBIES_1_371_WI_1_validation | I2 | loo_time | 10 | 3.2135 | 0.1597 | 96.0810 | 88.6846 | 101.4588 | profit/window |
| HOBBIES_1_371_WI_1_validation | I2 | none | 10 | 3.2422 | 0.1518 | 98.9337 | 96.6651 | 101.1705 | profit/window |
| HOBBIES_1_371_WI_1_validation | I3 | linear | 10 | 3.2673 | 0.1114 | 103.7958 | 100.8617 | 106.2969 | profit/window |
| HOBBIES_1_371_WI_1_validation | I3 | loo_time | 10 | 3.2361 | 0.1628 | 104.4616 | 102.4152 | 106.3110 | profit/window |
| HOBBIES_1_371_WI_1_validation | I3 | none | 10 | 3.1930 | 0.1852 | 98.1303 | 92.9945 | 102.9854 | profit/window |
| TECHM | T3 | linear | 5 | 1.0234 | 0.1266 | 0.6493 | 0.5173 | 0.7874 | cum. return |
| TECHM | T3 | loo_time | 5 | 1.0685 | 0.0947 | 0.4231 | 0.2467 | 0.5701 | cum. return |
| TECHM | T3 | none | 5 | 1.0789 | 0.0629 | 0.5617 | 0.2373 | 0.8304 | cum. return |
| TECHM | T1 | linear | 10 | 0.9773 | 0.1112 | 0.6508 | 0.4819 | 0.8021 | cum. return |
| TECHM | T1 | loo_time | 10 | 0.9318 | 0.1181 | 0.8131 | 0.6874 | 0.9231 | cum. return |
| TECHM | T1 | none | 10 | 0.9382 | 0.0814 | 0.7856 | 0.5979 | 0.9713 | cum. return |
| TECHM | T2 | linear | 10 | 0.9541 | 0.0638 | 0.6883 | 0.5547 | 0.8379 | cum. return |
| TECHM | T2 | loo_time | 10 | 0.9555 | 0.0968 | 0.7559 | 0.6624 | 0.8388 | cum. return |
| TECHM | T2 | none | 10 | 0.9647 | 0.0459 | 0.6158 | 0.5293 | 0.7088 | cum. return |
