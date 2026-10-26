# REINFORCE evaluation (full)

## Summary (validation J, normalised units; selection by validation only)

| key | policy | method | n_seeds | final_val_mean | final_val_sd | best_val_mean | episodes_to_90 | reached_90_share | optimality_gap_vs_best_known |
|---|---|---|---|---|---|---|---|---|---|
| FOODS_2_128_WI_2_validation | I1 | cm:central | 10 | 0.3643 | 0.0229 | 0.7190 | 640.0000 | 0.2000 | 0.0496 |
| FOODS_2_128_WI_2_validation | I1 | cm:neldermead | 10 | 0.7101 | 0.0026 | 0.7898 | 1820.4444 | 0.9000 | -0.0212 |
| FOODS_2_128_WI_2_validation | I1 | cm:random | 10 | 0.4635 | 0.0543 | 0.7418 | 1877.3333 | 0.6000 | 0.0268 |
| FOODS_2_128_WI_2_validation | I1 | cm:spsa | 10 | 0.7158 | 0.0147 | 0.7789 | 4608.0000 | 1.0000 | -0.0103 |
| FOODS_2_128_WI_2_validation | I1 | reinforce:linear | 10 | 0.7126 | 0.0436 | 0.7457 | 4133.3333 | 0.6000 | 0.0229 |
| FOODS_2_128_WI_2_validation | I1 | reinforce:loo_const | 10 | 0.7243 | 0.0378 | 0.7396 | 4712.0000 | 0.4000 | 0.0290 |
| FOODS_2_128_WI_2_validation | I1 | reinforce:loo_time | 10 | 0.7211 | 0.0401 | 0.7460 | 5952.0000 | 0.5000 | 0.0226 |
| FOODS_2_128_WI_2_validation | I1 | reinforce:loo_time_std | 10 | 0.7312 | 0.0289 | 0.7484 | 6150.4000 | 0.5000 | 0.0202 |
| FOODS_2_128_WI_2_validation | I1 | reinforce:none | 10 | 0.7328 | 0.0348 | 0.7401 | 10118.4000 | 0.5000 | 0.0285 |
| FOODS_2_128_WI_2_validation | I2 | cm:central | 10 | 0.3360 | 0.0434 | 0.6138 | 1316.5714 | 0.7000 | -0.0542 |
| FOODS_2_128_WI_2_validation | I2 | cm:neldermead | 10 | 0.5382 | 0.0639 | 0.6776 | 1945.6000 | 1.0000 | -0.1181 |
| FOODS_2_128_WI_2_validation | I2 | cm:random | 10 | 0.2743 | 0.1377 | 0.5639 | 0.0000 | 0.5000 | -0.0044 |
| FOODS_2_128_WI_2_validation | I2 | cm:spsa | 10 | 0.3092 | 0.1202 | 0.6536 | 4437.3333 | 0.9000 | -0.0941 |
| FOODS_2_128_WI_2_validation | I2 | reinforce:linear | 10 | -10.5035 | 21.6017 | 0.7125 | 1543.1111 | 0.9000 | -0.1529 |
| FOODS_2_128_WI_2_validation | I2 | reinforce:loo_const | 10 | -20.6118 | 26.5365 | 0.6359 | 708.5714 | 0.7000 | -0.0764 |
| FOODS_2_128_WI_2_validation | I2 | reinforce:loo_time | 10 | -5.0272 | 16.3110 | 0.7237 | 661.3333 | 0.9000 | -0.1642 |
| FOODS_2_128_WI_2_validation | I2 | reinforce:loo_time_std | 10 | -5.0373 | 16.3085 | 0.7218 | 661.3333 | 0.9000 | -0.1622 |
| FOODS_2_128_WI_2_validation | I2 | reinforce:none | 10 | -20.2884 | 26.7721 | 0.7457 | 1884.8000 | 1.0000 | -0.1861 |
| FOODS_2_128_WI_2_validation | I3 | reinforce:linear | 10 | -0.3249 | 1.8965 | -0.0652 | 1200.0000 | 0.8000 | -0.0610 |
| FOODS_2_128_WI_2_validation | I3 | reinforce:loo_const | 10 | -0.3039 | 1.8663 | -0.0657 | 1440.0000 | 0.8000 | -0.0605 |
| FOODS_2_128_WI_2_validation | I3 | reinforce:loo_time | 10 | -0.6464 | 2.9266 | -0.0767 | 1200.0000 | 0.8000 | -0.0495 |
| FOODS_2_128_WI_2_validation | I3 | reinforce:loo_time_std | 10 | -0.6462 | 2.9217 | -0.0592 | 1200.0000 | 0.8000 | -0.0671 |
| FOODS_2_128_WI_2_validation | I3 | reinforce:none | 10 | -0.3047 | 1.8592 | -0.0519 | 1200.0000 | 0.8000 | -0.0743 |
| FOODS_3_124_CA_3_validation | I1 | cm:central | 10 | 2.4528 | 0.0258 | 2.8563 | 1920.0000 | 0.8000 | 0.0147 |
| FOODS_3_124_CA_3_validation | I1 | cm:neldermead | 10 | 2.4167 | 0.0767 | 2.7787 | 0.0000 | 0.3000 | 0.0922 |
| FOODS_3_124_CA_3_validation | I1 | cm:random | 10 | 2.4912 | 0.0644 | 2.8723 | 640.0000 | 0.8000 | -0.0014 |
| FOODS_3_124_CA_3_validation | I1 | cm:spsa | 10 | 2.6066 | 0.0227 | 2.8757 | 819.2000 | 1.0000 | -0.0048 |
| FOODS_3_124_CA_3_validation | I1 | reinforce:linear | 10 | 2.7904 | 0.0290 | 2.8760 | 3744.0000 | 1.0000 | -0.0051 |
| FOODS_3_124_CA_3_validation | I1 | reinforce:loo_const | 10 | 2.8429 | 0.0341 | 2.8844 | 4896.0000 | 1.0000 | -0.0135 |
| FOODS_3_124_CA_3_validation | I1 | reinforce:loo_time | 10 | 2.7925 | 0.0276 | 2.8724 | 3744.0000 | 1.0000 | -0.0015 |
| FOODS_3_124_CA_3_validation | I1 | reinforce:loo_time_std | 10 | 2.7883 | 0.0228 | 2.8733 | 3456.0000 | 1.0000 | -0.0024 |
| FOODS_3_124_CA_3_validation | I1 | reinforce:none | 10 | 2.8659 | 0.0195 | 2.8741 | 8544.0000 | 1.0000 | -0.0032 |
| FOODS_3_124_CA_3_validation | I2 | cm:central | 10 | 2.7003 | 0.0513 | 2.7870 | 16896.0000 | 0.1000 | 0.1640 |
| FOODS_3_124_CA_3_validation | I2 | cm:neldermead | 10 | 2.5586 | 0.0992 | 2.6519 | nan | 0.0000 | 0.2991 |
| FOODS_3_124_CA_3_validation | I2 | cm:random | 10 | 2.7123 | 0.1117 | 2.9144 | 17846.8571 | 0.7000 | 0.0366 |
| FOODS_3_124_CA_3_validation | I2 | cm:spsa | 10 | 2.5166 | 0.1161 | 2.8548 | 2048.0000 | 0.7000 | 0.0962 |
| FOODS_3_124_CA_3_validation | I2 | reinforce:linear | 10 | 2.4647 | 0.1450 | 2.9967 | 4265.6000 | 1.0000 | -0.0456 |
| FOODS_3_124_CA_3_validation | I2 | reinforce:loo_const | 10 | 2.9330 | 0.0827 | 2.9934 | 4849.7778 | 0.9000 | -0.0424 |
| FOODS_3_124_CA_3_validation | I2 | reinforce:loo_time | 10 | 2.5529 | 0.1551 | 2.9840 | 4265.6000 | 1.0000 | -0.0330 |
| FOODS_3_124_CA_3_validation | I2 | reinforce:loo_time_std | 10 | 2.9090 | 0.1218 | 3.0190 | 4364.8000 | 1.0000 | -0.0680 |
| FOODS_3_124_CA_3_validation | I2 | reinforce:none | 10 | -7.4836 | 32.5386 | 2.9004 | 8597.3333 | 0.9000 | 0.0506 |
| FOODS_3_124_CA_3_validation | I3 | reinforce:linear | 10 | 2.9015 | 0.0447 | 3.0483 | 1248.0000 | 1.0000 | -0.0258 |
| FOODS_3_124_CA_3_validation | I3 | reinforce:loo_const | 10 | 2.8844 | 0.0700 | 3.0525 | 2304.0000 | 1.0000 | -0.0299 |
| FOODS_3_124_CA_3_validation | I3 | reinforce:loo_time | 10 | 2.8782 | 0.0585 | 3.0502 | 1152.0000 | 1.0000 | -0.0276 |
| FOODS_3_124_CA_3_validation | I3 | reinforce:loo_time_std | 10 | 2.8816 | 0.0405 | 3.0419 | 1152.0000 | 1.0000 | -0.0193 |
| FOODS_3_124_CA_3_validation | I3 | reinforce:none | 10 | 2.8750 | 0.0936 | 3.0403 | 2592.0000 | 1.0000 | -0.0177 |
| HOBBIES_1_371_WI_1_validation | I1 | cm:central | 10 | 2.9776 | 0.0363 | 3.2455 | 5120.0000 | 1.0000 | -0.0154 |
| HOBBIES_1_371_WI_1_validation | I1 | cm:neldermead | 10 | 2.0465 | 0.9653 | 2.2856 | nan | 0.0000 | 0.9445 |
| HOBBIES_1_371_WI_1_validation | I1 | cm:random | 10 | 3.0163 | 0.0344 | 3.2021 | 2730.6667 | 0.9000 | 0.0280 |
| HOBBIES_1_371_WI_1_validation | I1 | cm:spsa | 10 | 3.0089 | 0.0366 | 3.1681 | 2432.0000 | 0.8000 | 0.0619 |
| HOBBIES_1_371_WI_1_validation | I1 | reinforce:linear | 10 | 3.1585 | 0.1179 | 3.3249 | 1190.4000 | 1.0000 | -0.0948 |
| HOBBIES_1_371_WI_1_validation | I1 | reinforce:loo_const | 10 | 3.1709 | 0.0977 | 3.2934 | 992.0000 | 1.0000 | -0.0633 |
| HOBBIES_1_371_WI_1_validation | I1 | reinforce:loo_time | 10 | 3.1526 | 0.0895 | 3.3168 | 1190.4000 | 1.0000 | -0.0867 |
| HOBBIES_1_371_WI_1_validation | I1 | reinforce:loo_time_std | 10 | 1.1309 | 1.9568 | 3.3226 | 1091.2000 | 1.0000 | -0.0925 |
| HOBBIES_1_371_WI_1_validation | I1 | reinforce:none | 10 | 3.0553 | 0.0944 | 3.3095 | 1190.4000 | 1.0000 | -0.0794 |
| HOBBIES_1_371_WI_1_validation | I2 | cm:central | 10 | 3.0212 | 0.0547 | 3.0412 | 28224.0000 | 0.8000 | 0.0087 |
| HOBBIES_1_371_WI_1_validation | I2 | cm:neldermead | 10 | 2.6212 | 0.0698 | 2.8577 | nan | 0.0000 | 0.1922 |
| HOBBIES_1_371_WI_1_validation | I2 | cm:random | 10 | 2.9979 | 0.1072 | 3.1795 | 4608.0000 | 1.0000 | -0.1296 |
| HOBBIES_1_371_WI_1_validation | I2 | cm:spsa | 10 | 2.8846 | 0.0733 | 3.1545 | 7782.4000 | 1.0000 | -0.1046 |
| HOBBIES_1_371_WI_1_validation | I2 | reinforce:linear | 10 | -4.8580 | 23.7198 | 3.2406 | 3372.8000 | 1.0000 | -0.1907 |
| HOBBIES_1_371_WI_1_validation | I2 | reinforce:loo_const | 10 | -34.8937 | 39.2926 | 3.1609 | 1842.2857 | 0.7000 | -0.1111 |
| HOBBIES_1_371_WI_1_validation | I2 | reinforce:loo_time | 10 | -19.6598 | 36.3552 | 3.2135 | 2204.4444 | 0.9000 | -0.1636 |
| HOBBIES_1_371_WI_1_validation | I2 | reinforce:loo_time_std | 10 | -19.5974 | 36.3979 | 3.2043 | 2356.0000 | 0.8000 | -0.1545 |
| HOBBIES_1_371_WI_1_validation | I2 | reinforce:none | 10 | -19.6087 | 36.2535 | 3.2422 | 2424.8889 | 0.9000 | -0.1923 |
| HOBBIES_1_371_WI_1_validation | I3 | reinforce:linear | 10 | 2.9908 | 0.1436 | 3.2673 | 1388.8000 | 1.0000 | -0.1936 |
| HOBBIES_1_371_WI_1_validation | I3 | reinforce:loo_const | 10 | 2.9900 | 0.1696 | 3.2380 | 1785.6000 | 1.0000 | -0.1643 |
| HOBBIES_1_371_WI_1_validation | I3 | reinforce:loo_time | 10 | 2.9870 | 0.1824 | 3.2361 | 3372.8000 | 1.0000 | -0.1624 |
| HOBBIES_1_371_WI_1_validation | I3 | reinforce:loo_time_std | 10 | 2.9266 | 0.1361 | 3.2745 | 1091.2000 | 1.0000 | -0.2008 |
| HOBBIES_1_371_WI_1_validation | I3 | reinforce:none | 10 | 2.9728 | 0.2402 | 3.1930 | 1587.2000 | 1.0000 | -0.1194 |
| TECHM | T1 | cm:central | 10 | 0.7374 | 0.0927 | 0.8340 | 5320.0000 | 0.8000 | 0.0190 |
| TECHM | T1 | cm:neldermead | 10 | 0.6857 | 0.1029 | 0.7537 | 2520.0000 | 0.4000 | 0.0993 |
| TECHM | T1 | cm:random | 10 | 0.6216 | 0.1161 | 0.8552 | 6016.0000 | 0.8000 | -0.0022 |
| TECHM | T1 | cm:spsa | 10 | 0.6731 | 0.0805 | 0.6732 | nan | 0.0000 | 0.1798 |
| TECHM | T1 | reinforce:linear | 10 | 0.8110 | 0.1174 | 0.9773 | 6392.8889 | 0.9000 | -0.1243 |
| TECHM | T1 | reinforce:loo_const | 10 | 0.7853 | 0.1302 | 0.9593 | 3968.0000 | 0.9000 | -0.1064 |
| TECHM | T1 | reinforce:loo_time | 10 | 0.8051 | 0.1401 | 0.9318 | 6392.8889 | 0.9000 | -0.0788 |
| TECHM | T1 | reinforce:loo_time_std | 10 | 0.7354 | 0.0986 | 0.9309 | 4519.1111 | 0.9000 | -0.0780 |
| TECHM | T1 | reinforce:none | 10 | 0.7818 | 0.1313 | 0.9382 | 7164.4444 | 0.9000 | -0.0853 |
| TECHM | T2 | reinforce:linear | 10 | 0.8018 | 0.1080 | 0.9541 | 6528.0000 | 1.0000 | -0.1499 |
| TECHM | T2 | reinforce:loo_const | 10 | 0.7742 | 0.0590 | 0.9492 | 6048.0000 | 1.0000 | -0.1451 |
| TECHM | T2 | reinforce:loo_time | 10 | 0.7853 | 0.0732 | 0.9555 | 6048.0000 | 1.0000 | -0.1514 |
| TECHM | T2 | reinforce:loo_time_std | 10 | 0.7540 | 0.1021 | 0.9441 | 7200.0000 | 1.0000 | -0.1400 |
| TECHM | T2 | reinforce:none | 10 | 0.7701 | 0.0815 | 0.9647 | 5760.0000 | 1.0000 | -0.1605 |

## Stability shares

| key | policy | method | collapsed | converged | diverged | stalled | unconverged |
|---|---|---|---|---|---|---|---|
| FOODS_2_128_WI_2_validation | I1 | cm:central | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_2_128_WI_2_validation | I1 | cm:neldermead | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| FOODS_2_128_WI_2_validation | I1 | cm:random | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_2_128_WI_2_validation | I1 | cm:spsa | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| FOODS_2_128_WI_2_validation | I1 | reinforce:linear | 0.0000 | 0.2000 | 0.0000 | 0.0000 | 0.8000 |
| FOODS_2_128_WI_2_validation | I1 | reinforce:loo_const | 0.0000 | 0.3000 | 0.0000 | 0.0000 | 0.7000 |
| FOODS_2_128_WI_2_validation | I1 | reinforce:loo_time | 0.0000 | 0.4000 | 0.0000 | 0.0000 | 0.6000 |
| FOODS_2_128_WI_2_validation | I1 | reinforce:loo_time_std | 0.0000 | 0.4000 | 0.0000 | 0.0000 | 0.6000 |
| FOODS_2_128_WI_2_validation | I1 | reinforce:none | 0.0000 | 0.5000 | 0.0000 | 0.0000 | 0.5000 |
| FOODS_2_128_WI_2_validation | I2 | cm:central | 0.9000 | 0.0000 | 0.0000 | 0.1000 | 0.0000 |
| FOODS_2_128_WI_2_validation | I2 | cm:neldermead | 0.7000 | 0.3000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_2_128_WI_2_validation | I2 | cm:random | 0.5000 | 0.0000 | 0.0000 | 0.5000 | 0.0000 |
| FOODS_2_128_WI_2_validation | I2 | cm:spsa | 0.8000 | 0.0000 | 0.0000 | 0.1000 | 0.1000 |
| FOODS_2_128_WI_2_validation | I2 | reinforce:linear | 0.6000 | 0.1000 | 0.3000 | 0.0000 | 0.0000 |
| FOODS_2_128_WI_2_validation | I2 | reinforce:loo_const | 0.3000 | 0.1000 | 0.6000 | 0.0000 | 0.0000 |
| FOODS_2_128_WI_2_validation | I2 | reinforce:loo_time | 0.7000 | 0.1000 | 0.2000 | 0.0000 | 0.0000 |
| FOODS_2_128_WI_2_validation | I2 | reinforce:loo_time_std | 0.7000 | 0.1000 | 0.2000 | 0.0000 | 0.0000 |
| FOODS_2_128_WI_2_validation | I2 | reinforce:none | 0.5000 | 0.1000 | 0.4000 | 0.0000 | 0.0000 |
| FOODS_2_128_WI_2_validation | I3 | reinforce:linear | 0.0000 | 0.8000 | 0.0000 | 0.1000 | 0.1000 |
| FOODS_2_128_WI_2_validation | I3 | reinforce:loo_const | 0.0000 | 0.8000 | 0.0000 | 0.1000 | 0.1000 |
| FOODS_2_128_WI_2_validation | I3 | reinforce:loo_time | 0.0000 | 0.8000 | 0.1000 | 0.0000 | 0.1000 |
| FOODS_2_128_WI_2_validation | I3 | reinforce:loo_time_std | 0.0000 | 0.8000 | 0.1000 | 0.0000 | 0.1000 |
| FOODS_2_128_WI_2_validation | I3 | reinforce:none | 0.0000 | 0.8000 | 0.0000 | 0.1000 | 0.1000 |
| FOODS_3_124_CA_3_validation | I1 | cm:central | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_3_124_CA_3_validation | I1 | cm:neldermead | 0.9000 | 0.0000 | 0.0000 | 0.1000 | 0.0000 |
| FOODS_3_124_CA_3_validation | I1 | cm:random | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_3_124_CA_3_validation | I1 | cm:spsa | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_3_124_CA_3_validation | I1 | reinforce:linear | 0.0000 | 0.1000 | 0.0000 | 0.0000 | 0.9000 |
| FOODS_3_124_CA_3_validation | I1 | reinforce:loo_const | 0.0000 | 0.5000 | 0.0000 | 0.0000 | 0.5000 |
| FOODS_3_124_CA_3_validation | I1 | reinforce:loo_time | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I1 | reinforce:loo_time_std | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I1 | reinforce:none | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_3_124_CA_3_validation | I2 | cm:central | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I2 | cm:neldermead | 0.1000 | 0.0000 | 0.0000 | 0.1000 | 0.8000 |
| FOODS_3_124_CA_3_validation | I2 | cm:random | 0.1000 | 0.0000 | 0.0000 | 0.0000 | 0.9000 |
| FOODS_3_124_CA_3_validation | I2 | cm:spsa | 0.6000 | 0.0000 | 0.0000 | 0.0000 | 0.4000 |
| FOODS_3_124_CA_3_validation | I2 | reinforce:linear | 0.8000 | 0.0000 | 0.0000 | 0.0000 | 0.2000 |
| FOODS_3_124_CA_3_validation | I2 | reinforce:loo_const | 0.0000 | 0.7000 | 0.0000 | 0.0000 | 0.3000 |
| FOODS_3_124_CA_3_validation | I2 | reinforce:loo_time | 0.6000 | 0.0000 | 0.0000 | 0.0000 | 0.4000 |
| FOODS_3_124_CA_3_validation | I2 | reinforce:loo_time_std | 0.1000 | 0.6000 | 0.0000 | 0.0000 | 0.3000 |
| FOODS_3_124_CA_3_validation | I2 | reinforce:none | 0.0000 | 0.1000 | 0.1000 | 0.0000 | 0.8000 |
| FOODS_3_124_CA_3_validation | I3 | reinforce:linear | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_3_124_CA_3_validation | I3 | reinforce:loo_const | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_3_124_CA_3_validation | I3 | reinforce:loo_time | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_3_124_CA_3_validation | I3 | reinforce:loo_time_std | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_3_124_CA_3_validation | I3 | reinforce:none | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| HOBBIES_1_371_WI_1_validation | I1 | cm:central | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I1 | cm:neldermead | 0.0000 | 0.0000 | 0.2000 | 0.1000 | 0.7000 |
| HOBBIES_1_371_WI_1_validation | I1 | cm:random | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I1 | cm:spsa | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I1 | reinforce:linear | 0.0000 | 0.6000 | 0.0000 | 0.0000 | 0.4000 |
| HOBBIES_1_371_WI_1_validation | I1 | reinforce:loo_const | 0.0000 | 0.6000 | 0.0000 | 0.0000 | 0.4000 |
| HOBBIES_1_371_WI_1_validation | I1 | reinforce:loo_time | 0.0000 | 0.7000 | 0.0000 | 0.0000 | 0.3000 |
| HOBBIES_1_371_WI_1_validation | I1 | reinforce:loo_time_std | 0.4000 | 0.1000 | 0.4000 | 0.0000 | 0.1000 |
| HOBBIES_1_371_WI_1_validation | I1 | reinforce:none | 0.0000 | 0.1000 | 0.0000 | 0.0000 | 0.9000 |
| HOBBIES_1_371_WI_1_validation | I2 | cm:central | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I2 | cm:neldermead | 0.9000 | 0.0000 | 0.0000 | 0.1000 | 0.0000 |
| HOBBIES_1_371_WI_1_validation | I2 | cm:random | 0.1000 | 0.1000 | 0.0000 | 0.0000 | 0.8000 |
| HOBBIES_1_371_WI_1_validation | I2 | cm:spsa | 0.4000 | 0.0000 | 0.0000 | 0.0000 | 0.6000 |
| HOBBIES_1_371_WI_1_validation | I2 | reinforce:linear | 0.4000 | 0.3000 | 0.2000 | 0.0000 | 0.1000 |
| HOBBIES_1_371_WI_1_validation | I2 | reinforce:loo_const | 0.3000 | 0.1000 | 0.6000 | 0.0000 | 0.0000 |
| HOBBIES_1_371_WI_1_validation | I2 | reinforce:loo_time | 0.2000 | 0.3000 | 0.3000 | 0.0000 | 0.2000 |
| HOBBIES_1_371_WI_1_validation | I2 | reinforce:loo_time_std | 0.2000 | 0.4000 | 0.3000 | 0.0000 | 0.1000 |
| HOBBIES_1_371_WI_1_validation | I2 | reinforce:none | 0.0000 | 0.4000 | 0.4000 | 0.0000 | 0.2000 |
| HOBBIES_1_371_WI_1_validation | I3 | reinforce:linear | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| HOBBIES_1_371_WI_1_validation | I3 | reinforce:loo_const | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| HOBBIES_1_371_WI_1_validation | I3 | reinforce:loo_time | 0.0000 | 0.9000 | 0.0000 | 0.0000 | 0.1000 |
| HOBBIES_1_371_WI_1_validation | I3 | reinforce:loo_time_std | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| HOBBIES_1_371_WI_1_validation | I3 | reinforce:none | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| TECHM | T1 | cm:central | 0.0000 | 0.2000 | 0.0000 | 0.0000 | 0.8000 |
| TECHM | T1 | cm:neldermead | 0.0000 | 0.2000 | 0.0000 | 0.1000 | 0.7000 |
| TECHM | T1 | cm:random | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |
| TECHM | T1 | cm:spsa | 0.0000 | 0.0000 | 0.0000 | 0.2000 | 0.8000 |
| TECHM | T1 | reinforce:linear | 0.1000 | 0.4000 | 0.0000 | 0.0000 | 0.5000 |
| TECHM | T1 | reinforce:loo_const | 0.1000 | 0.4000 | 0.0000 | 0.0000 | 0.5000 |
| TECHM | T1 | reinforce:loo_time | 0.1000 | 0.5000 | 0.0000 | 0.0000 | 0.4000 |
| TECHM | T1 | reinforce:loo_time_std | 0.1000 | 0.2000 | 0.0000 | 0.0000 | 0.7000 |
| TECHM | T1 | reinforce:none | 0.1000 | 0.4000 | 0.0000 | 0.0000 | 0.5000 |
| TECHM | T2 | reinforce:linear | 0.0000 | 0.5000 | 0.0000 | 0.0000 | 0.5000 |
| TECHM | T2 | reinforce:loo_const | 0.0000 | 0.2000 | 0.0000 | 0.0000 | 0.8000 |
| TECHM | T2 | reinforce:loo_time | 0.0000 | 0.3000 | 0.0000 | 0.0000 | 0.7000 |
| TECHM | T2 | reinforce:loo_time_std | 0.1000 | 0.4000 | 0.0000 | 0.0000 | 0.5000 |
| TECHM | T2 | reinforce:none | 0.0000 | 0.2000 | 0.0000 | 0.0000 | 0.8000 |

## Paired tests (Wilcoxon over seeds, Holm within family)

| key | policy | method | vs | mean_diff | p | p_holm |
|---|---|---|---|---|---|---|
| FOODS_2_128_WI_2_validation | I1 | reinforce:linear | none | 0.0056 | 0.1602 | 0.5234 |
| FOODS_2_128_WI_2_validation | I1 | reinforce:loo_const | none | -0.0005 | 0.7695 | 0.7695 |
| FOODS_2_128_WI_2_validation | I1 | reinforce:loo_time | none | 0.0059 | 0.2754 | 0.5508 |
| FOODS_2_128_WI_2_validation | I1 | reinforce:loo_time_std | none | 0.0083 | 0.1309 | 0.5234 |
| FOODS_2_128_WI_2_validation | I1 | reinforce:loo_time | cm:central | 0.0271 | 0.0039 | 0.0117 |
| FOODS_2_128_WI_2_validation | I1 | reinforce:loo_time | cm:neldermead | -0.0438 | 0.0059 | 0.0117 |
| FOODS_2_128_WI_2_validation | I1 | reinforce:loo_time | cm:random | 0.0042 | 0.8457 | 0.8457 |
| FOODS_2_128_WI_2_validation | I1 | reinforce:loo_time | cm:spsa | -0.0328 | 0.0020 | 0.0078 |
| FOODS_2_128_WI_2_validation | I2 | reinforce:linear | none | -0.0332 | 0.3223 | 1.0000 |
| FOODS_2_128_WI_2_validation | I2 | reinforce:loo_const | none | -0.1098 | 0.3223 | 1.0000 |
| FOODS_2_128_WI_2_validation | I2 | reinforce:loo_time | none | -0.0220 | 0.6250 | 1.0000 |
| FOODS_2_128_WI_2_validation | I2 | reinforce:loo_time_std | none | -0.0239 | 0.4922 | 1.0000 |
| FOODS_2_128_WI_2_validation | I2 | reinforce:loo_time | cm:central | 0.1099 | 0.0078 | 0.0312 |
| FOODS_2_128_WI_2_validation | I2 | reinforce:loo_time | cm:neldermead | 0.0461 | 0.1309 | 0.2578 |
| FOODS_2_128_WI_2_validation | I2 | reinforce:loo_time | cm:random | 0.1598 | 0.0078 | 0.0312 |
| FOODS_2_128_WI_2_validation | I2 | reinforce:loo_time | cm:spsa | 0.0701 | 0.1289 | 0.2578 |
| FOODS_2_128_WI_2_validation | I3 | reinforce:linear | none | -0.0133 | 0.4316 | 0.9668 |
| FOODS_2_128_WI_2_validation | I3 | reinforce:loo_const | none | -0.0138 | 0.3223 | 0.9668 |
| FOODS_2_128_WI_2_validation | I3 | reinforce:loo_time | none | -0.0248 | 0.0840 | 0.3359 |
| FOODS_2_128_WI_2_validation | I3 | reinforce:loo_time_std | none | -0.0073 | 0.9219 | 0.9668 |
| FOODS_3_124_CA_3_validation | I1 | reinforce:linear | none | 0.0019 | 1.0000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I1 | reinforce:loo_const | none | 0.0103 | 0.1309 | 0.5234 |
| FOODS_3_124_CA_3_validation | I1 | reinforce:loo_time | none | -0.0017 | 0.8457 | 1.0000 |
| FOODS_3_124_CA_3_validation | I1 | reinforce:loo_time_std | none | -0.0008 | 0.7695 | 1.0000 |
| FOODS_3_124_CA_3_validation | I1 | reinforce:loo_time | cm:central | 0.0162 | 0.0469 | 0.1406 |
| FOODS_3_124_CA_3_validation | I1 | reinforce:loo_time | cm:neldermead | 0.0937 | 0.0156 | 0.0625 |
| FOODS_3_124_CA_3_validation | I1 | reinforce:loo_time | cm:random | 0.0001 | 0.8125 | 1.0000 |
| FOODS_3_124_CA_3_validation | I1 | reinforce:loo_time | cm:spsa | -0.0033 | 0.5781 | 1.0000 |
| FOODS_3_124_CA_3_validation | I2 | reinforce:linear | none | 0.0962 | 0.0098 | 0.0293 |
| FOODS_3_124_CA_3_validation | I2 | reinforce:loo_const | none | 0.0930 | 0.0098 | 0.0293 |
| FOODS_3_124_CA_3_validation | I2 | reinforce:loo_time | none | 0.0836 | 0.0098 | 0.0293 |
| FOODS_3_124_CA_3_validation | I2 | reinforce:loo_time_std | none | 0.1186 | 0.0020 | 0.0078 |
| FOODS_3_124_CA_3_validation | I2 | reinforce:loo_time | cm:central | 0.1970 | 0.0020 | 0.0078 |
| FOODS_3_124_CA_3_validation | I2 | reinforce:loo_time | cm:neldermead | 0.3321 | 0.0020 | 0.0078 |
| FOODS_3_124_CA_3_validation | I2 | reinforce:loo_time | cm:random | 0.0696 | 0.0098 | 0.0195 |
| FOODS_3_124_CA_3_validation | I2 | reinforce:loo_time | cm:spsa | 0.1292 | 0.0137 | 0.0195 |
| FOODS_3_124_CA_3_validation | I3 | reinforce:linear | none | 0.0081 | 1.0000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I3 | reinforce:loo_const | none | 0.0122 | 0.6250 | 1.0000 |
| FOODS_3_124_CA_3_validation | I3 | reinforce:loo_time | none | 0.0099 | 1.0000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I3 | reinforce:loo_time_std | none | 0.0016 | 1.0000 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I1 | reinforce:linear | none | 0.0154 | 0.7695 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I1 | reinforce:loo_const | none | -0.0161 | 0.5566 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I1 | reinforce:loo_time | none | 0.0073 | 0.9219 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I1 | reinforce:loo_time_std | none | 0.0131 | 0.9219 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I1 | reinforce:loo_time | cm:central | 0.0713 | 0.0020 | 0.0078 |
| HOBBIES_1_371_WI_1_validation | I1 | reinforce:loo_time | cm:neldermead | 1.0312 | 0.0020 | 0.0078 |
| HOBBIES_1_371_WI_1_validation | I1 | reinforce:loo_time | cm:random | 0.1147 | 0.0039 | 0.0078 |
| HOBBIES_1_371_WI_1_validation | I1 | reinforce:loo_time | cm:spsa | 0.1487 | 0.0020 | 0.0078 |
| HOBBIES_1_371_WI_1_validation | I2 | reinforce:linear | none | -0.0015 | 0.6250 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I2 | reinforce:loo_const | none | -0.0812 | 0.6953 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I2 | reinforce:loo_time | none | -0.0287 | 0.6953 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I2 | reinforce:loo_time_std | none | -0.0378 | 1.0000 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I2 | reinforce:loo_time | cm:central | 0.1723 | 0.0137 | 0.0410 |
| HOBBIES_1_371_WI_1_validation | I2 | reinforce:loo_time | cm:neldermead | 0.3558 | 0.0039 | 0.0156 |
| HOBBIES_1_371_WI_1_validation | I2 | reinforce:loo_time | cm:random | 0.0340 | 0.4316 | 0.4316 |
| HOBBIES_1_371_WI_1_validation | I2 | reinforce:loo_time | cm:spsa | 0.0590 | 0.1934 | 0.3867 |
| HOBBIES_1_371_WI_1_validation | I3 | reinforce:linear | none | 0.0742 | 0.4922 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I3 | reinforce:loo_const | none | 0.0449 | 1.0000 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I3 | reinforce:loo_time | none | 0.0431 | 1.0000 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I3 | reinforce:loo_time_std | none | 0.0814 | 0.6953 | 1.0000 |
| TECHM | T1 | reinforce:linear | none | 0.0391 | 0.1602 | 0.6406 |
| TECHM | T1 | reinforce:loo_const | none | 0.0211 | 0.4316 | 1.0000 |
| TECHM | T1 | reinforce:loo_time | none | -0.0065 | 1.0000 | 1.0000 |
| TECHM | T1 | reinforce:loo_time_std | none | -0.0073 | 0.9219 | 1.0000 |
| TECHM | T1 | reinforce:loo_time | cm:central | 0.0978 | 0.0137 | 0.0410 |
| TECHM | T1 | reinforce:loo_time | cm:neldermead | 0.1781 | 0.0273 | 0.0547 |
| TECHM | T1 | reinforce:loo_time | cm:random | 0.0766 | 0.1055 | 0.1055 |
| TECHM | T1 | reinforce:loo_time | cm:spsa | 0.2586 | 0.0039 | 0.0156 |
| TECHM | T2 | reinforce:linear | none | -0.0106 | 0.7695 | 1.0000 |
| TECHM | T2 | reinforce:loo_const | none | -0.0154 | 0.6953 | 1.0000 |
| TECHM | T2 | reinforce:loo_time | none | -0.0092 | 0.5566 | 1.0000 |
| TECHM | T2 | reinforce:loo_time_std | none | -0.0206 | 0.6250 | 1.0000 |

## Gradient quality (200 estimates of B=32; reference = loo_time, 50x batch)

| key | policy | theta | baseline | trace_cov | snr | cos_to_ref | rel_l2_to_ref | bias_vs_none_z | var_ratio_vs_none |
|---|---|---|---|---|---|---|---|---|---|
| FOODS_3_124_CA_3_validation | I1 | initial | none | 31.2407 | 0.2493 | 0.3900 | 1.4981 | 0.0000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I1 | initial | loo_const | 16.7780 | 0.5359 | 0.5596 | 1.0815 | 0.8304 | 0.5371 |
| FOODS_3_124_CA_3_validation | I1 | initial | loo_time | 13.9822 | 0.5410 | 0.5445 | 1.0052 | 0.4819 | 0.4476 |
| FOODS_3_124_CA_3_validation | I1 | initial | linear | 7.6006 | 0.9164 | 0.6700 | 0.7631 | 0.5898 | 0.2433 |
| FOODS_3_124_CA_3_validation | I1 | initial | loo_time_std | 30.3197 | 0.7822 | 0.6233 | 1.5026 | 3.7875 | 0.9705 |
| FOODS_3_124_CA_3_validation | I1 | trained | none | 109.8432 | 0.0099 | 0.0551 | 9.5313 | 0.0000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I1 | trained | loo_const | 24.2320 | 0.0300 | 0.1379 | 4.4675 | 0.6188 | 0.2206 |
| FOODS_3_124_CA_3_validation | I1 | trained | loo_time | 3.8923 | 0.1421 | 0.3008 | 1.8035 | 0.6416 | 0.0354 |
| FOODS_3_124_CA_3_validation | I1 | trained | linear | 3.1007 | 0.1989 | 0.3728 | 1.6079 | 0.5905 | 0.0282 |
| FOODS_3_124_CA_3_validation | I1 | trained | loo_time_std | 15.2518 | 0.1880 | 0.3484 | 3.5963 | 1.2554 | 0.1389 |
| FOODS_3_124_CA_3_validation | I2 | initial | none | 1446.0798 | 0.0189 | 0.0897 | 5.1898 | 0.0000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I2 | initial | loo_const | 296.5784 | 0.0609 | 0.2271 | 2.3525 | 0.4388 | 0.2051 |
| FOODS_3_124_CA_3_validation | I2 | initial | loo_time | 67.8126 | 0.2499 | 0.3866 | 1.1385 | 0.4666 | 0.0469 |
| FOODS_3_124_CA_3_validation | I2 | initial | linear | 46.3005 | 0.4058 | 0.4534 | 0.9630 | 0.3890 | 0.0320 |
| FOODS_3_124_CA_3_validation | I2 | initial | loo_time_std | 267.1978 | 0.2652 | 0.3985 | 2.2377 | 1.1477 | 0.1848 |
| FOODS_3_124_CA_3_validation | I2 | trained | none | 165.8717 | 0.0135 | 0.0939 | 10.0638 | 0.0000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I2 | trained | loo_const | 35.7124 | 0.0090 | 0.0518 | 4.6788 | 1.2904 | 0.2153 |
| FOODS_3_124_CA_3_validation | I2 | trained | loo_time | 9.1225 | 0.0210 | 0.1094 | 2.4213 | 1.2173 | 0.0550 |
| FOODS_3_124_CA_3_validation | I2 | trained | linear | 6.5152 | 0.0526 | 0.1477 | 2.0705 | 1.0621 | 0.0393 |
| FOODS_3_124_CA_3_validation | I2 | trained | loo_time_std | 17.8927 | 0.0201 | 0.0996 | 3.3158 | 1.0389 | 0.1079 |
| TECHM | T1 | initial | none | 1.5917 | 0.0178 | 0.0981 | 3.6348 | 0.0000 | 1.0000 |
| TECHM | T1 | initial | loo_const | 1.7088 | 0.0137 | 0.0967 | 3.7514 | 0.6659 | 1.0736 |
| TECHM | T1 | initial | loo_time | 1.5809 | 0.0162 | 0.0145 | 3.7424 | 1.1371 | 0.9932 |
| TECHM | T1 | initial | linear | 1.5417 | 0.0244 | 0.1090 | 3.5360 | 1.0690 | 0.9686 |
| TECHM | T1 | initial | loo_time_std | 11.7933 | 0.0180 | -0.0474 | 10.2649 | 1.6968 | 7.4092 |
| TECHM | T1 | trained | none | 1.4002 | 0.0093 | 0.0198 | 6.0823 | 0.0000 | 1.0000 |
| TECHM | T1 | trained | loo_const | 1.4277 | 0.0056 | 0.0730 | 6.0992 | 1.1347 | 1.0197 |
| TECHM | T1 | trained | loo_time | 1.2317 | 0.0059 | 0.0050 | 5.7934 | 1.2791 | 0.8797 |
| TECHM | T1 | trained | linear | 1.3014 | 0.0027 | -0.0252 | 5.9695 | 1.2008 | 0.9294 |
| TECHM | T1 | trained | loo_time_std | 4.2150 | 0.0082 | 0.0618 | 10.6804 | 0.6481 | 3.0103 |

## Sensitivity (exploratory)

| domain | key | policy | variable | value | mean | std | count |
|---|---|---|---|---|---|---|---|
| inventory | FOODS_3_124_CA_3_validation | I2 | B | 16.0000 | 2.9929 | 0.0330 | 5 |
| inventory | FOODS_3_124_CA_3_validation | I2 | B | 32.0000 | 2.9842 | 0.0327 | 5 |
| inventory | FOODS_3_124_CA_3_validation | I2 | B | 64.0000 | 2.9781 | 0.0323 | 5 |
| inventory | FOODS_3_124_CA_3_validation | I2 | B | 128.0000 | 2.9119 | 0.0389 | 5 |
| inventory | FOODS_3_124_CA_3_validation | I2 | clip | 1.0000 | 3.0033 | 0.0129 | 5 |
| inventory | FOODS_3_124_CA_3_validation | I2 | clip | 5.0000 | 2.9929 | 0.0330 | 5 |
| inventory | FOODS_3_124_CA_3_validation | I2 | clip | 25.0000 | 2.9597 | 0.0364 | 5 |
| inventory | FOODS_3_124_CA_3_validation | I2 | gamma | 0.9000 | 2.9183 | 0.0371 | 5 |
| inventory | FOODS_3_124_CA_3_validation | I2 | gamma | 0.9900 | 2.9929 | 0.0330 | 5 |
| inventory | FOODS_3_124_CA_3_validation | I2 | gamma | 1.0000 | 3.0041 | 0.0357 | 5 |
| inventory | FOODS_3_124_CA_3_validation | I2 | horizon | 30.0000 | 2.5588 | 0.0541 | 5 |
| inventory | FOODS_3_124_CA_3_validation | I2 | horizon | 90.0000 | 2.9929 | 0.0330 | 5 |
| inventory | FOODS_3_124_CA_3_validation | I2 | horizon | 180.0000 | 3.9704 | 0.0717 | 5 |
| inventory | FOODS_3_124_CA_3_validation | I2 | lr | 0.0100 | 2.9664 | 0.0367 | 5 |
| inventory | FOODS_3_124_CA_3_validation | I2 | lr | 0.0300 | 2.9929 | 0.0330 | 5 |
| inventory | FOODS_3_124_CA_3_validation | I2 | lr | 0.0900 | 2.9532 | 0.0536 | 5 |
| inventory | FOODS_3_124_CA_3_validation | I2 | sigma0 | 0.2500 | 2.9918 | 0.0330 | 5 |
| inventory | FOODS_3_124_CA_3_validation | I2 | sigma0 | 0.5000 | 2.9929 | 0.0330 | 5 |
| inventory | FOODS_3_124_CA_3_validation | I2 | sigma0 | 1.0000 | 2.9953 | 0.0249 | 5 |
| trading | TECHM | T1 | B | 16.0000 | 0.9445 | 0.1384 | 5 |
| trading | TECHM | T1 | B | 32.0000 | 0.9397 | 0.0996 | 5 |
| trading | TECHM | T1 | B | 64.0000 | 0.8866 | 0.1318 | 5 |
| trading | TECHM | T1 | B | 128.0000 | 0.8649 | 0.1610 | 5 |
| trading | TECHM | T1 | clip | 1.0000 | 0.9538 | 0.0870 | 5 |
| trading | TECHM | T1 | clip | 5.0000 | 0.9445 | 0.1384 | 5 |
| trading | TECHM | T1 | clip | 25.0000 | 0.9565 | 0.1407 | 5 |
| trading | TECHM | T1 | gamma | 0.9000 | 0.9224 | 0.0604 | 5 |
| trading | TECHM | T1 | gamma | 0.9900 | 0.9369 | 0.1166 | 5 |
| trading | TECHM | T1 | gamma | 1.0000 | 0.9103 | 0.1103 | 5 |
| trading | TECHM | T1 | horizon | 125.0000 | 0.4950 | 0.0314 | 5 |
| trading | TECHM | T1 | horizon | 250.0000 | 0.9445 | 0.1384 | 5 |
| trading | TECHM | T1 | horizon | 500.0000 | 1.8245 | 0.1518 | 5 |
| trading | TECHM | T1 | lr | 0.0100 | 0.8940 | 0.1272 | 5 |
| trading | TECHM | T1 | lr | 0.0300 | 0.9445 | 0.1384 | 5 |
| trading | TECHM | T1 | lr | 0.0900 | 0.8886 | 0.1291 | 5 |