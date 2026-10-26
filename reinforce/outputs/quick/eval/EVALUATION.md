# REINFORCE evaluation (quick)

## Summary (validation J, normalised units; selection by validation only)

| key | policy | method | n_seeds | final_val_mean | final_val_sd | best_val_mean | episodes_to_90 | reached_90_share | optimality_gap_vs_best_known |
|---|---|---|---|---|---|---|---|---|---|
| FOODS_2_128_WI_2_validation | I1 | cm:central | 3 | 0.6929 | 0.0619 | 0.7047 | 960.0000 | 0.6667 | 0.0459 |
| FOODS_2_128_WI_2_validation | I1 | cm:neldermead | 3 | 0.7101 | 0.0007 | 0.8210 | 1333.3333 | 1.0000 | -0.0704 |
| FOODS_2_128_WI_2_validation | I1 | cm:random | 3 | 0.6107 | 0.0331 | 0.7516 | 288.0000 | 0.6667 | -0.0010 |
| FOODS_2_128_WI_2_validation | I1 | cm:spsa | 3 | 0.7506 | 0.0485 | 0.7541 | 576.0000 | 0.6667 | -0.0035 |
| FOODS_2_128_WI_2_validation | I1 | reinforce:linear | 3 | 0.6320 | 0.0861 | 0.7329 | 64.0000 | 0.6667 | 0.0178 |
| FOODS_2_128_WI_2_validation | I1 | reinforce:loo_const | 3 | 0.6099 | 0.1021 | 0.7289 | 128.0000 | 0.6667 | 0.0217 |
| FOODS_2_128_WI_2_validation | I1 | reinforce:loo_time | 3 | 0.6272 | 0.0865 | 0.7346 | 128.0000 | 0.6667 | 0.0161 |
| FOODS_2_128_WI_2_validation | I1 | reinforce:loo_time_std | 3 | 0.6421 | 0.0668 | 0.7269 | 64.0000 | 0.6667 | 0.0237 |
| FOODS_2_128_WI_2_validation | I1 | reinforce:none | 3 | 0.6413 | 0.0866 | 0.7333 | 64.0000 | 0.6667 | 0.0173 |
| FOODS_2_128_WI_2_validation | I2 | cm:central | 3 | 0.3667 | 0.3355 | 0.6887 | 1536.0000 | 0.3333 | 0.0215 |
| FOODS_2_128_WI_2_validation | I2 | cm:neldermead | 3 | 0.4137 | 0.0947 | 0.7176 | 720.0000 | 0.6667 | -0.0075 |
| FOODS_2_128_WI_2_validation | I2 | cm:random | 3 | 0.2483 | 0.1284 | 0.6597 | nan | 0.0000 | 0.0504 |
| FOODS_2_128_WI_2_validation | I2 | cm:spsa | 3 | 0.3118 | 0.3710 | 0.7203 | 576.0000 | 0.3333 | -0.0102 |
| FOODS_2_128_WI_2_validation | I2 | reinforce:linear | 3 | -34.1666 | 29.9006 | 0.6825 | nan | 0.0000 | 0.0276 |
| FOODS_2_128_WI_2_validation | I2 | reinforce:loo_const | 3 | 0.2439 | 0.6024 | 0.7170 | 448.0000 | 0.6667 | -0.0069 |
| FOODS_2_128_WI_2_validation | I2 | reinforce:loo_time | 3 | -17.1535 | 29.3517 | 0.6934 | 1920.0000 | 0.3333 | 0.0167 |
| FOODS_2_128_WI_2_validation | I2 | reinforce:loo_time_std | 3 | -17.1716 | 29.3600 | 0.6952 | 1536.0000 | 0.3333 | 0.0150 |
| FOODS_2_128_WI_2_validation | I2 | reinforce:none | 3 | -16.7992 | 29.9335 | 0.6783 | nan | 0.0000 | 0.0318 |
| FOODS_2_128_WI_2_validation | I3 | reinforce:linear | 3 | -2.3047 | 2.8855 | -1.0853 | 128.0000 | 0.3333 | -0.0043 |
| FOODS_2_128_WI_2_validation | I3 | reinforce:loo_const | 3 | -2.1490 | 0.0305 | -1.8924 | nan | 0.0000 | 0.8028 |
| FOODS_2_128_WI_2_validation | I3 | reinforce:loo_time | 3 | -3.1381 | 1.7152 | -1.0672 | 512.0000 | 0.3333 | -0.0224 |
| FOODS_2_128_WI_2_validation | I3 | reinforce:loo_time_std | 3 | -1.3514 | 1.6227 | -1.0211 | 2048.0000 | 0.3333 | -0.0685 |
| FOODS_2_128_WI_2_validation | I3 | reinforce:none | 3 | -3.2904 | 1.7807 | -1.9664 | nan | 0.0000 | 0.8768 |
| FOODS_3_124_CA_3_validation | I1 | cm:central | 3 | 2.7826 | 0.0433 | 2.8317 | 1280.0000 | 0.3333 | 0.0738 |
| FOODS_3_124_CA_3_validation | I1 | cm:neldermead | 3 | 2.4440 | 0.0760 | 2.8120 | 1440.0000 | 0.3333 | 0.0935 |
| FOODS_3_124_CA_3_validation | I1 | cm:random | 3 | 2.4599 | 0.0308 | 2.9028 | 960.0000 | 1.0000 | 0.0027 |
| FOODS_3_124_CA_3_validation | I1 | cm:spsa | 3 | 2.7823 | 0.0482 | 2.9007 | 832.0000 | 1.0000 | 0.0049 |
| FOODS_3_124_CA_3_validation | I1 | reinforce:linear | 3 | 2.8391 | 0.0122 | 2.8933 | 1109.3333 | 1.0000 | 0.0123 |
| FOODS_3_124_CA_3_validation | I1 | reinforce:loo_const | 3 | 2.9029 | 0.0290 | 2.9102 | 1450.6667 | 1.0000 | -0.0047 |
| FOODS_3_124_CA_3_validation | I1 | reinforce:loo_time | 3 | 2.8507 | 0.0361 | 2.8962 | 1066.6667 | 1.0000 | 0.0093 |
| FOODS_3_124_CA_3_validation | I1 | reinforce:loo_time_std | 3 | 2.8520 | 0.0331 | 2.8979 | 1024.0000 | 1.0000 | 0.0076 |
| FOODS_3_124_CA_3_validation | I1 | reinforce:none | 3 | 2.8450 | 0.0615 | 2.8484 | 2688.0000 | 0.6667 | 0.0571 |
| FOODS_3_124_CA_3_validation | I2 | cm:central | 3 | 2.6509 | 0.0705 | 2.6509 | nan | 0.0000 | 0.3128 |
| FOODS_3_124_CA_3_validation | I2 | cm:neldermead | 3 | 2.5308 | 0.0629 | 2.5878 | nan | 0.0000 | 0.3759 |
| FOODS_3_124_CA_3_validation | I2 | cm:random | 3 | 2.6654 | 0.0869 | 2.7343 | nan | 0.0000 | 0.2294 |
| FOODS_3_124_CA_3_validation | I2 | cm:spsa | 3 | 2.7456 | 0.0428 | 3.0078 | 1152.0000 | 1.0000 | -0.0440 |
| FOODS_3_124_CA_3_validation | I2 | reinforce:linear | 3 | -31.4000 | 59.3449 | 2.8094 | 2688.0000 | 0.6667 | 0.1544 |
| FOODS_3_124_CA_3_validation | I2 | reinforce:loo_const | 3 | -31.7301 | 59.1314 | 2.7917 | 1152.0000 | 0.3333 | 0.1720 |
| FOODS_3_124_CA_3_validation | I2 | reinforce:loo_time | 3 | 2.8085 | 0.1402 | 2.9080 | 1920.0000 | 0.6667 | 0.0558 |
| FOODS_3_124_CA_3_validation | I2 | reinforce:loo_time_std | 3 | 2.7607 | 0.2650 | 2.8363 | 1664.0000 | 0.3333 | 0.1274 |
| FOODS_3_124_CA_3_validation | I2 | reinforce:none | 3 | 2.8819 | 0.1472 | 2.9751 | 2432.0000 | 0.6667 | -0.0113 |
| FOODS_3_124_CA_3_validation | I3 | reinforce:linear | 3 | 2.6221 | 0.2000 | 2.9947 | 256.0000 | 1.0000 | -0.0583 |
| FOODS_3_124_CA_3_validation | I3 | reinforce:loo_const | 3 | 2.8479 | 0.0724 | 3.0315 | 1322.6667 | 1.0000 | -0.0951 |
| FOODS_3_124_CA_3_validation | I3 | reinforce:loo_time | 3 | 2.7922 | 0.0766 | 3.0142 | 298.6667 | 1.0000 | -0.0779 |
| FOODS_3_124_CA_3_validation | I3 | reinforce:loo_time_std | 3 | 2.8181 | 0.0387 | 3.0695 | 298.6667 | 1.0000 | -0.1332 |
| FOODS_3_124_CA_3_validation | I3 | reinforce:none | 3 | -0.2928 | 5.4749 | -0.1810 | 128.0000 | 0.6667 | 3.1173 |
| HOBBIES_1_371_WI_1_validation | I1 | cm:central | 3 | 3.1890 | 0.0338 | 3.1890 | 5120.0000 | 0.6667 | 0.0457 |
| HOBBIES_1_371_WI_1_validation | I1 | cm:neldermead | 3 | 2.6340 | 0.0164 | 3.0052 | nan | 0.0000 | 0.2295 |
| HOBBIES_1_371_WI_1_validation | I1 | cm:random | 3 | 2.9698 | 0.1175 | 3.2136 | 576.0000 | 0.6667 | 0.0211 |
| HOBBIES_1_371_WI_1_validation | I1 | cm:spsa | 3 | 3.1294 | 0.0813 | 3.1653 | 2688.0000 | 0.3333 | 0.0694 |
| HOBBIES_1_371_WI_1_validation | I1 | reinforce:linear | 3 | 3.1340 | 0.0700 | 3.2067 | 853.3333 | 1.0000 | 0.0280 |
| HOBBIES_1_371_WI_1_validation | I1 | reinforce:loo_const | 3 | 3.1661 | 0.0566 | 3.1956 | 1578.6667 | 1.0000 | 0.0391 |
| HOBBIES_1_371_WI_1_validation | I1 | reinforce:loo_time | 3 | 3.2201 | 0.0763 | 3.2420 | 853.3333 | 1.0000 | -0.0073 |
| HOBBIES_1_371_WI_1_validation | I1 | reinforce:loo_time_std | 3 | 3.2182 | 0.0712 | 3.2316 | 832.0000 | 0.6667 | 0.0031 |
| HOBBIES_1_371_WI_1_validation | I1 | reinforce:none | 3 | 3.2343 | 0.0972 | 3.2820 | 853.3333 | 1.0000 | -0.0473 |
| HOBBIES_1_371_WI_1_validation | I2 | cm:central | 3 | 2.5499 | 0.4309 | 2.8621 | nan | 0.0000 | 0.3523 |
| HOBBIES_1_371_WI_1_validation | I2 | cm:neldermead | 3 | 2.4790 | 0.1487 | 2.9402 | nan | 0.0000 | 0.2742 |
| HOBBIES_1_371_WI_1_validation | I2 | cm:random | 3 | 3.0324 | 0.0901 | 3.0938 | 1344.0000 | 0.3333 | 0.1206 |
| HOBBIES_1_371_WI_1_validation | I2 | cm:spsa | 3 | 2.9850 | 0.0749 | 3.2067 | 2976.0000 | 0.6667 | 0.0077 |
| HOBBIES_1_371_WI_1_validation | I2 | reinforce:linear | 3 | -21.9308 | 43.6561 | 3.1551 | 960.0000 | 0.6667 | 0.0593 |
| HOBBIES_1_371_WI_1_validation | I2 | reinforce:loo_const | 3 | 3.0438 | 0.1895 | 3.2623 | 1536.0000 | 1.0000 | -0.0479 |
| HOBBIES_1_371_WI_1_validation | I2 | reinforce:loo_time | 3 | 2.9392 | 0.2122 | 3.1628 | 1664.0000 | 0.6667 | 0.0516 |
| HOBBIES_1_371_WI_1_validation | I2 | reinforce:loo_time_std | 3 | 3.0160 | 0.2444 | 3.2035 | 2474.6667 | 1.0000 | 0.0109 |
| HOBBIES_1_371_WI_1_validation | I2 | reinforce:none | 3 | 3.1402 | 0.0697 | 3.2592 | 1408.0000 | 1.0000 | -0.0449 |
| HOBBIES_1_371_WI_1_validation | I3 | reinforce:linear | 3 | 1.3960 | 2.2942 | 3.1216 | 170.6667 | 1.0000 | -0.1334 |
| HOBBIES_1_371_WI_1_validation | I3 | reinforce:loo_const | 3 | -2.3630 | 8.8276 | 3.0425 | 170.6667 | 1.0000 | -0.0543 |
| HOBBIES_1_371_WI_1_validation | I3 | reinforce:loo_time | 3 | -6.4446 | 15.6463 | 3.0144 | 170.6667 | 1.0000 | -0.0262 |
| HOBBIES_1_371_WI_1_validation | I3 | reinforce:loo_time_std | 3 | -0.3064 | 5.4306 | 3.0187 | 128.0000 | 1.0000 | -0.0305 |
| HOBBIES_1_371_WI_1_validation | I3 | reinforce:none | 3 | 1.2746 | 2.0259 | 1.6824 | 192.0000 | 0.6667 | 1.3058 |
| TECHM | T1 | cm:central | 3 | 0.8371 | 0.0328 | 0.8379 | 4928.0000 | 1.0000 | -0.0008 |
| TECHM | T1 | cm:neldermead | 3 | 0.6864 | 0.1706 | 0.7429 | 1120.0000 | 0.3333 | 0.0942 |
| TECHM | T1 | cm:random | 3 | 0.6098 | 0.1932 | 0.8736 | 1973.3333 | 1.0000 | -0.0364 |
| TECHM | T1 | cm:spsa | 3 | 0.4522 | 0.1139 | 0.4522 | nan | 0.0000 | 0.3850 |
| TECHM | T1 | reinforce:linear | 3 | 0.5730 | 0.3020 | 0.7471 | 1088.0000 | 0.6667 | 0.0900 |
| TECHM | T1 | reinforce:loo_const | 3 | 0.4314 | 0.3513 | 0.5438 | 768.0000 | 0.3333 | 0.2933 |
| TECHM | T1 | reinforce:loo_time | 3 | 0.6545 | 0.3688 | 0.7397 | 1856.0000 | 0.6667 | 0.0975 |
| TECHM | T1 | reinforce:loo_time_std | 3 | 0.5196 | 0.3108 | 0.6821 | 1536.0000 | 0.3333 | 0.1551 |
| TECHM | T1 | reinforce:none | 3 | 0.5596 | 0.2413 | 0.7187 | 768.0000 | 0.3333 | 0.1185 |
| TECHM | T2 | reinforce:linear | 3 | 0.7209 | 0.0878 | 0.8035 | 2389.3333 | 1.0000 | -0.0101 |
| TECHM | T2 | reinforce:loo_const | 3 | 0.7413 | 0.1573 | 0.9205 | 1792.0000 | 1.0000 | -0.1271 |
| TECHM | T2 | reinforce:loo_time | 3 | 0.7506 | 0.2062 | 0.8446 | 1920.0000 | 1.0000 | -0.0512 |
| TECHM | T2 | reinforce:loo_time_std | 3 | 0.6005 | 0.0772 | 0.7137 | 1408.0000 | 0.3333 | 0.0797 |
| TECHM | T2 | reinforce:none | 3 | 0.7863 | 0.0495 | 0.9120 | 2005.3333 | 1.0000 | -0.1186 |

## Stability shares

| key | policy | method | collapsed | converged | diverged | stalled | unconverged |
|---|---|---|---|---|---|---|---|
| FOODS_2_128_WI_2_validation | I1 | cm:central | 0.0000 | 0.6667 | 0.0000 | 0.3333 | 0.0000 |
| FOODS_2_128_WI_2_validation | I1 | cm:neldermead | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| FOODS_2_128_WI_2_validation | I1 | cm:random | 0.6667 | 0.0000 | 0.0000 | 0.0000 | 0.3333 |
| FOODS_2_128_WI_2_validation | I1 | cm:spsa | 0.0000 | 0.6667 | 0.0000 | 0.0000 | 0.3333 |
| FOODS_2_128_WI_2_validation | I1 | reinforce:linear | 0.3333 | 0.0000 | 0.0000 | 0.0000 | 0.6667 |
| FOODS_2_128_WI_2_validation | I1 | reinforce:loo_const | 0.3333 | 0.0000 | 0.0000 | 0.0000 | 0.6667 |
| FOODS_2_128_WI_2_validation | I1 | reinforce:loo_time | 0.3333 | 0.0000 | 0.0000 | 0.0000 | 0.6667 |
| FOODS_2_128_WI_2_validation | I1 | reinforce:loo_time_std | 0.3333 | 0.0000 | 0.0000 | 0.0000 | 0.6667 |
| FOODS_2_128_WI_2_validation | I1 | reinforce:none | 0.3333 | 0.0000 | 0.0000 | 0.0000 | 0.6667 |
| FOODS_2_128_WI_2_validation | I2 | cm:central | 0.6667 | 0.0000 | 0.0000 | 0.3333 | 0.0000 |
| FOODS_2_128_WI_2_validation | I2 | cm:neldermead | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_2_128_WI_2_validation | I2 | cm:random | 0.6667 | 0.0000 | 0.0000 | 0.3333 | 0.0000 |
| FOODS_2_128_WI_2_validation | I2 | cm:spsa | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_2_128_WI_2_validation | I2 | reinforce:linear | 0.3333 | 0.0000 | 0.6667 | 0.0000 | 0.0000 |
| FOODS_2_128_WI_2_validation | I2 | reinforce:loo_const | 0.3333 | 0.3333 | 0.3333 | 0.0000 | 0.0000 |
| FOODS_2_128_WI_2_validation | I2 | reinforce:loo_time | 0.3333 | 0.0000 | 0.6667 | 0.0000 | 0.0000 |
| FOODS_2_128_WI_2_validation | I2 | reinforce:loo_time_std | 0.3333 | 0.0000 | 0.6667 | 0.0000 | 0.0000 |
| FOODS_2_128_WI_2_validation | I2 | reinforce:none | 0.6667 | 0.0000 | 0.3333 | 0.0000 | 0.0000 |
| FOODS_2_128_WI_2_validation | I3 | reinforce:linear | 0.3333 | 0.3333 | 0.0000 | 0.0000 | 0.3333 |
| FOODS_2_128_WI_2_validation | I3 | reinforce:loo_const | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| FOODS_2_128_WI_2_validation | I3 | reinforce:loo_time | 0.3333 | 0.0000 | 0.0000 | 0.0000 | 0.6667 |
| FOODS_2_128_WI_2_validation | I3 | reinforce:loo_time_std | 0.0000 | 0.3333 | 0.0000 | 0.0000 | 0.6667 |
| FOODS_2_128_WI_2_validation | I3 | reinforce:none | 0.3333 | 0.0000 | 0.0000 | 0.0000 | 0.6667 |
| FOODS_3_124_CA_3_validation | I1 | cm:central | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I1 | cm:neldermead | 0.3333 | 0.0000 | 0.0000 | 0.0000 | 0.6667 |
| FOODS_3_124_CA_3_validation | I1 | cm:random | 0.3333 | 0.0000 | 0.0000 | 0.0000 | 0.6667 |
| FOODS_3_124_CA_3_validation | I1 | cm:spsa | 0.0000 | 0.3333 | 0.0000 | 0.0000 | 0.6667 |
| FOODS_3_124_CA_3_validation | I1 | reinforce:linear | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_3_124_CA_3_validation | I1 | reinforce:loo_const | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_3_124_CA_3_validation | I1 | reinforce:loo_time | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_3_124_CA_3_validation | I1 | reinforce:loo_time_std | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_3_124_CA_3_validation | I1 | reinforce:none | 0.0000 | 0.3333 | 0.0000 | 0.0000 | 0.6667 |
| FOODS_3_124_CA_3_validation | I2 | cm:central | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| FOODS_3_124_CA_3_validation | I2 | cm:neldermead | 0.3333 | 0.0000 | 0.0000 | 0.6667 | 0.0000 |
| FOODS_3_124_CA_3_validation | I2 | cm:random | 0.3333 | 0.0000 | 0.0000 | 0.3333 | 0.3333 |
| FOODS_3_124_CA_3_validation | I2 | cm:spsa | 0.6667 | 0.0000 | 0.0000 | 0.0000 | 0.3333 |
| FOODS_3_124_CA_3_validation | I2 | reinforce:linear | 0.0000 | 0.3333 | 0.3333 | 0.0000 | 0.3333 |
| FOODS_3_124_CA_3_validation | I2 | reinforce:loo_const | 0.3333 | 0.0000 | 0.3333 | 0.0000 | 0.3333 |
| FOODS_3_124_CA_3_validation | I2 | reinforce:loo_time | 0.3333 | 0.3333 | 0.0000 | 0.0000 | 0.3333 |
| FOODS_3_124_CA_3_validation | I2 | reinforce:loo_time_std | 0.3333 | 0.0000 | 0.0000 | 0.0000 | 0.6667 |
| FOODS_3_124_CA_3_validation | I2 | reinforce:none | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I3 | reinforce:linear | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_3_124_CA_3_validation | I3 | reinforce:loo_const | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_3_124_CA_3_validation | I3 | reinforce:loo_time | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_3_124_CA_3_validation | I3 | reinforce:loo_time_std | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_3_124_CA_3_validation | I3 | reinforce:none | 0.0000 | 0.6667 | 0.0000 | 0.0000 | 0.3333 |
| HOBBIES_1_371_WI_1_validation | I1 | cm:central | 0.0000 | 0.3333 | 0.0000 | 0.0000 | 0.6667 |
| HOBBIES_1_371_WI_1_validation | I1 | cm:neldermead | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| HOBBIES_1_371_WI_1_validation | I1 | cm:random | 0.3333 | 0.0000 | 0.0000 | 0.0000 | 0.6667 |
| HOBBIES_1_371_WI_1_validation | I1 | cm:spsa | 0.0000 | 0.3333 | 0.0000 | 0.0000 | 0.6667 |
| HOBBIES_1_371_WI_1_validation | I1 | reinforce:linear | 0.0000 | 0.3333 | 0.0000 | 0.0000 | 0.6667 |
| HOBBIES_1_371_WI_1_validation | I1 | reinforce:loo_const | 0.0000 | 0.3333 | 0.0000 | 0.0000 | 0.6667 |
| HOBBIES_1_371_WI_1_validation | I1 | reinforce:loo_time | 0.0000 | 0.6667 | 0.0000 | 0.0000 | 0.3333 |
| HOBBIES_1_371_WI_1_validation | I1 | reinforce:loo_time_std | 0.0000 | 0.6667 | 0.0000 | 0.0000 | 0.3333 |
| HOBBIES_1_371_WI_1_validation | I1 | reinforce:none | 0.0000 | 0.6667 | 0.0000 | 0.0000 | 0.3333 |
| HOBBIES_1_371_WI_1_validation | I2 | cm:central | 0.3333 | 0.0000 | 0.0000 | 0.3333 | 0.3333 |
| HOBBIES_1_371_WI_1_validation | I2 | cm:neldermead | 0.6667 | 0.0000 | 0.0000 | 0.0000 | 0.3333 |
| HOBBIES_1_371_WI_1_validation | I2 | cm:random | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I2 | cm:spsa | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I2 | reinforce:linear | 0.0000 | 0.6667 | 0.3333 | 0.0000 | 0.0000 |
| HOBBIES_1_371_WI_1_validation | I2 | reinforce:loo_const | 0.0000 | 0.3333 | 0.0000 | 0.0000 | 0.6667 |
| HOBBIES_1_371_WI_1_validation | I2 | reinforce:loo_time | 0.3333 | 0.3333 | 0.0000 | 0.0000 | 0.3333 |
| HOBBIES_1_371_WI_1_validation | I2 | reinforce:loo_time_std | 0.3333 | 0.3333 | 0.0000 | 0.0000 | 0.3333 |
| HOBBIES_1_371_WI_1_validation | I2 | reinforce:none | 0.0000 | 0.6667 | 0.0000 | 0.0000 | 0.3333 |
| HOBBIES_1_371_WI_1_validation | I3 | reinforce:linear | 0.0000 | 0.3333 | 0.0000 | 0.0000 | 0.6667 |
| HOBBIES_1_371_WI_1_validation | I3 | reinforce:loo_const | 0.0000 | 0.6667 | 0.3333 | 0.0000 | 0.0000 |
| HOBBIES_1_371_WI_1_validation | I3 | reinforce:loo_time | 0.0000 | 0.6667 | 0.3333 | 0.0000 | 0.0000 |
| HOBBIES_1_371_WI_1_validation | I3 | reinforce:loo_time_std | 0.3333 | 0.6667 | 0.0000 | 0.0000 | 0.0000 |
| HOBBIES_1_371_WI_1_validation | I3 | reinforce:none | 0.0000 | 0.6667 | 0.0000 | 0.0000 | 0.3333 |
| TECHM | T1 | cm:central | 0.0000 | 0.3333 | 0.0000 | 0.0000 | 0.6667 |
| TECHM | T1 | cm:neldermead | 0.0000 | 0.3333 | 0.0000 | 0.0000 | 0.6667 |
| TECHM | T1 | cm:random | 0.3333 | 0.0000 | 0.0000 | 0.0000 | 0.6667 |
| TECHM | T1 | cm:spsa | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| TECHM | T1 | reinforce:linear | 0.3333 | 0.0000 | 0.0000 | 0.0000 | 0.6667 |
| TECHM | T1 | reinforce:loo_const | 0.3333 | 0.3333 | 0.0000 | 0.3333 | 0.0000 |
| TECHM | T1 | reinforce:loo_time | 0.3333 | 0.6667 | 0.0000 | 0.0000 | 0.0000 |
| TECHM | T1 | reinforce:loo_time_std | 0.6667 | 0.0000 | 0.0000 | 0.0000 | 0.3333 |
| TECHM | T1 | reinforce:none | 0.3333 | 0.3333 | 0.0000 | 0.0000 | 0.3333 |
| TECHM | T2 | reinforce:linear | 0.0000 | 0.3333 | 0.0000 | 0.0000 | 0.6667 |
| TECHM | T2 | reinforce:loo_const | 0.0000 | 0.3333 | 0.0000 | 0.0000 | 0.6667 |
| TECHM | T2 | reinforce:loo_time | 0.0000 | 0.6667 | 0.0000 | 0.0000 | 0.3333 |
| TECHM | T2 | reinforce:loo_time_std | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| TECHM | T2 | reinforce:none | 0.0000 | 0.6667 | 0.0000 | 0.0000 | 0.3333 |

## Paired tests (Wilcoxon over seeds, Holm within family)

| key | policy | method | vs | mean_diff | p | p_holm |
|---|---|---|---|---|---|---|
| FOODS_2_128_WI_2_validation | I1 | reinforce:linear | none | -0.0005 | 1.0000 | 1.0000 |
| FOODS_2_128_WI_2_validation | I1 | reinforce:loo_const | none | -0.0044 | 0.7500 | 1.0000 |
| FOODS_2_128_WI_2_validation | I1 | reinforce:loo_time | none | 0.0012 | 0.7500 | 1.0000 |
| FOODS_2_128_WI_2_validation | I1 | reinforce:loo_time_std | none | -0.0064 | 0.7500 | 1.0000 |
| FOODS_2_128_WI_2_validation | I1 | reinforce:loo_time | cm:central | 0.0298 | 0.5000 | 1.0000 |
| FOODS_2_128_WI_2_validation | I1 | reinforce:loo_time | cm:neldermead | -0.0864 | 0.2500 | 1.0000 |
| FOODS_2_128_WI_2_validation | I1 | reinforce:loo_time | cm:random | -0.0171 | 0.5000 | 1.0000 |
| FOODS_2_128_WI_2_validation | I1 | reinforce:loo_time | cm:spsa | -0.0196 | 0.5000 | 1.0000 |
| FOODS_2_128_WI_2_validation | I2 | reinforce:linear | none | 0.0042 | 0.5000 | 1.0000 |
| FOODS_2_128_WI_2_validation | I2 | reinforce:loo_const | none | 0.0387 | 0.5000 | 1.0000 |
| FOODS_2_128_WI_2_validation | I2 | reinforce:loo_time | none | 0.0151 | 1.0000 | 1.0000 |
| FOODS_2_128_WI_2_validation | I2 | reinforce:loo_time_std | none | 0.0168 | 1.0000 | 1.0000 |
| FOODS_2_128_WI_2_validation | I2 | reinforce:loo_time | cm:central | 0.0047 | 1.0000 | 1.0000 |
| FOODS_2_128_WI_2_validation | I2 | reinforce:loo_time | cm:neldermead | -0.0242 | 0.2500 | 1.0000 |
| FOODS_2_128_WI_2_validation | I2 | reinforce:loo_time | cm:random | 0.0337 | 0.5000 | 1.0000 |
| FOODS_2_128_WI_2_validation | I2 | reinforce:loo_time | cm:spsa | -0.0269 | 1.0000 | 1.0000 |
| FOODS_2_128_WI_2_validation | I3 | reinforce:linear | none | 0.8810 | 1.0000 | 1.0000 |
| FOODS_2_128_WI_2_validation | I3 | reinforce:loo_const | none | 0.0740 | 0.2500 | 1.0000 |
| FOODS_2_128_WI_2_validation | I3 | reinforce:loo_time | none | 0.8991 | 0.5000 | 1.0000 |
| FOODS_2_128_WI_2_validation | I3 | reinforce:loo_time_std | none | 0.9453 | 0.2500 | 1.0000 |
| FOODS_3_124_CA_3_validation | I1 | reinforce:linear | none | 0.0449 | 0.5000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I1 | reinforce:loo_const | none | 0.0618 | 0.2500 | 1.0000 |
| FOODS_3_124_CA_3_validation | I1 | reinforce:loo_time | none | 0.0478 | 0.2500 | 1.0000 |
| FOODS_3_124_CA_3_validation | I1 | reinforce:loo_time_std | none | 0.0495 | 0.2500 | 1.0000 |
| FOODS_3_124_CA_3_validation | I1 | reinforce:loo_time | cm:central | 0.0645 | 0.2500 | 1.0000 |
| FOODS_3_124_CA_3_validation | I1 | reinforce:loo_time | cm:neldermead | 0.0842 | 0.2500 | 1.0000 |
| FOODS_3_124_CA_3_validation | I1 | reinforce:loo_time | cm:random | -0.0066 | 1.0000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I1 | reinforce:loo_time | cm:spsa | -0.0045 | 1.0000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I2 | reinforce:linear | none | -0.1657 | 0.5000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I2 | reinforce:loo_const | none | -0.1834 | 0.5000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I2 | reinforce:loo_time | none | -0.0671 | 0.5000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I2 | reinforce:loo_time_std | none | -0.1388 | 0.2500 | 1.0000 |
| FOODS_3_124_CA_3_validation | I2 | reinforce:loo_time | cm:central | 0.2570 | 0.2500 | 1.0000 |
| FOODS_3_124_CA_3_validation | I2 | reinforce:loo_time | cm:neldermead | 0.3201 | 0.2500 | 1.0000 |
| FOODS_3_124_CA_3_validation | I2 | reinforce:loo_time | cm:random | 0.1736 | 0.2500 | 1.0000 |
| FOODS_3_124_CA_3_validation | I2 | reinforce:loo_time | cm:spsa | -0.0998 | 0.2500 | 1.0000 |
| FOODS_3_124_CA_3_validation | I3 | reinforce:linear | none | 3.1757 | 0.5000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I3 | reinforce:loo_const | none | 3.2125 | 0.2500 | 1.0000 |
| FOODS_3_124_CA_3_validation | I3 | reinforce:loo_time | none | 3.1952 | 0.7500 | 1.0000 |
| FOODS_3_124_CA_3_validation | I3 | reinforce:loo_time_std | none | 3.2505 | 0.2500 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I1 | reinforce:linear | none | -0.0753 | 0.2500 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I1 | reinforce:loo_const | none | -0.0864 | 0.2500 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I1 | reinforce:loo_time | none | -0.0400 | 0.2500 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I1 | reinforce:loo_time_std | none | -0.0504 | 0.2500 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I1 | reinforce:loo_time | cm:central | 0.0530 | 0.5000 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I1 | reinforce:loo_time | cm:neldermead | 0.2368 | 0.2500 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I1 | reinforce:loo_time | cm:random | 0.0284 | 0.7500 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I1 | reinforce:loo_time | cm:spsa | 0.0766 | 0.5000 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I2 | reinforce:linear | none | -0.1042 | 0.7500 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I2 | reinforce:loo_const | none | 0.0030 | 1.0000 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I2 | reinforce:loo_time | none | -0.0964 | 0.2500 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I2 | reinforce:loo_time_std | none | -0.0558 | 0.5000 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I2 | reinforce:loo_time | cm:central | 0.3007 | 0.2500 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I2 | reinforce:loo_time | cm:neldermead | 0.2226 | 0.2500 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I2 | reinforce:loo_time | cm:random | 0.0691 | 0.2500 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I2 | reinforce:loo_time | cm:spsa | -0.0438 | 0.7500 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I3 | reinforce:linear | none | 1.4392 | 0.2500 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I3 | reinforce:loo_const | none | 1.3601 | 0.2500 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I3 | reinforce:loo_time | none | 1.3320 | 0.2500 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I3 | reinforce:loo_time_std | none | 1.3363 | 0.5000 | 1.0000 |
| TECHM | T1 | reinforce:linear | none | 0.0284 | 0.7500 | 1.0000 |
| TECHM | T1 | reinforce:loo_const | none | -0.1748 | 0.5000 | 1.0000 |
| TECHM | T1 | reinforce:loo_time | none | 0.0210 | 1.0000 | 1.0000 |
| TECHM | T1 | reinforce:loo_time_std | none | -0.0366 | 1.0000 | 1.0000 |
| TECHM | T1 | reinforce:loo_time | cm:central | -0.0983 | 1.0000 | 1.0000 |
| TECHM | T1 | reinforce:loo_time | cm:neldermead | -0.0033 | 1.0000 | 1.0000 |
| TECHM | T1 | reinforce:loo_time | cm:random | -0.1339 | 1.0000 | 1.0000 |
| TECHM | T1 | reinforce:loo_time | cm:spsa | 0.2875 | 0.5000 | 1.0000 |
| TECHM | T2 | reinforce:linear | none | -0.1085 | 0.2500 | 1.0000 |
| TECHM | T2 | reinforce:loo_const | none | 0.0085 | 1.0000 | 1.0000 |
| TECHM | T2 | reinforce:loo_time | none | -0.0674 | 0.2500 | 1.0000 |
| TECHM | T2 | reinforce:loo_time_std | none | -0.1983 | 0.2500 | 1.0000 |

## Gradient quality (200 estimates of B=32; reference = loo_time, 50x batch)

| key | policy | theta | baseline | trace_cov | snr | cos_to_ref | rel_l2_to_ref | bias_vs_none_z | var_ratio_vs_none |
|---|---|---|---|---|---|---|---|---|---|
| FOODS_3_124_CA_3_validation | I1 | initial | none | 31.2407 | 0.2493 | 0.3900 | 1.4981 | 0.0000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I1 | initial | loo_const | 16.7780 | 0.5359 | 0.5596 | 1.0815 | 0.8304 | 0.5371 |
| FOODS_3_124_CA_3_validation | I1 | initial | loo_time | 13.9822 | 0.5410 | 0.5445 | 1.0052 | 0.4819 | 0.4476 |
| FOODS_3_124_CA_3_validation | I1 | initial | linear | 7.6006 | 0.9164 | 0.6700 | 0.7631 | 0.5898 | 0.2433 |
| FOODS_3_124_CA_3_validation | I1 | initial | loo_time_std | 30.3197 | 0.7822 | 0.6233 | 1.5026 | 3.7875 | 0.9705 |
| FOODS_3_124_CA_3_validation | I1 | trained | none | 177.3705 | 0.0112 | 0.0606 | 8.2741 | 0.0000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I1 | trained | loo_const | 39.2858 | 0.0358 | 0.1487 | 3.8909 | 0.6333 | 0.2215 |
| FOODS_3_124_CA_3_validation | I1 | trained | loo_time | 7.1028 | 0.1564 | 0.3065 | 1.6719 | 0.6452 | 0.0400 |
| FOODS_3_124_CA_3_validation | I1 | trained | linear | 4.9756 | 0.2214 | 0.3815 | 1.4093 | 0.6423 | 0.0281 |
| FOODS_3_124_CA_3_validation | I1 | trained | loo_time_std | 28.5547 | 0.2061 | 0.3553 | 3.3673 | 1.3310 | 0.1610 |
| FOODS_3_124_CA_3_validation | I2 | initial | none | 1446.0798 | 0.0189 | 0.0897 | 5.1898 | 0.0000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I2 | initial | loo_const | 296.5784 | 0.0609 | 0.2271 | 2.3525 | 0.4388 | 0.2051 |
| FOODS_3_124_CA_3_validation | I2 | initial | loo_time | 67.8126 | 0.2499 | 0.3866 | 1.1385 | 0.4666 | 0.0469 |
| FOODS_3_124_CA_3_validation | I2 | initial | linear | 46.3005 | 0.4058 | 0.4534 | 0.9630 | 0.3890 | 0.0320 |
| FOODS_3_124_CA_3_validation | I2 | initial | loo_time_std | 267.1978 | 0.2652 | 0.3985 | 2.2377 | 1.1477 | 0.1848 |
| FOODS_3_124_CA_3_validation | I2 | trained | none | 1635.2793 | 0.0063 | 0.0505 | 20.3386 | 0.0000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I2 | trained | loo_const | 319.7800 | 0.0022 | 0.0348 | 9.0095 | 0.9841 | 0.1956 |
| FOODS_3_124_CA_3_validation | I2 | trained | loo_time | 60.0811 | 0.0297 | 0.1540 | 3.7799 | 0.7060 | 0.0367 |
| FOODS_3_124_CA_3_validation | I2 | trained | linear | 52.8859 | 0.0598 | 0.2108 | 3.6935 | 0.5851 | 0.0323 |
| FOODS_3_124_CA_3_validation | I2 | trained | loo_time_std | 187.8043 | 0.0371 | 0.1391 | 6.9734 | 0.5784 | 0.1148 |
| TECHM | T1 | initial | none | 1.5917 | 0.0178 | 0.0981 | 3.6348 | 0.0000 | 1.0000 |
| TECHM | T1 | initial | loo_const | 1.7088 | 0.0137 | 0.0967 | 3.7514 | 0.6659 | 1.0736 |
| TECHM | T1 | initial | loo_time | 1.5809 | 0.0162 | 0.0145 | 3.7424 | 1.1371 | 0.9932 |
| TECHM | T1 | initial | linear | 1.5417 | 0.0244 | 0.1090 | 3.5360 | 1.0690 | 0.9686 |
| TECHM | T1 | initial | loo_time_std | 11.7933 | 0.0180 | -0.0474 | 10.2649 | 1.6968 | 7.4092 |
| TECHM | T1 | trained | none | 1.5917 | 0.0178 | 0.0981 | 3.6348 | 0.0000 | 1.0000 |
| TECHM | T1 | trained | loo_const | 1.7088 | 0.0137 | 0.0967 | 3.7514 | 0.6659 | 1.0736 |
| TECHM | T1 | trained | loo_time | 1.5809 | 0.0162 | 0.0145 | 3.7424 | 1.1371 | 0.9932 |
| TECHM | T1 | trained | linear | 1.5417 | 0.0244 | 0.1090 | 3.5360 | 1.0690 | 0.9686 |
| TECHM | T1 | trained | loo_time_std | 11.7933 | 0.0180 | -0.0474 | 10.2649 | 1.6968 | 7.4092 |

## Sensitivity (exploratory)

| domain | key | policy | variable | value | mean | std | count |
|---|---|---|---|---|---|---|---|
| inventory | FOODS_3_124_CA_3_validation | I2 | B | 16.0000 | 2.8318 | 0.1020 | 2 |
| inventory | FOODS_3_124_CA_3_validation | I2 | B | 32.0000 | 2.7185 | 0.3450 | 2 |
| inventory | FOODS_3_124_CA_3_validation | I2 | B | 64.0000 | 2.8639 | 0.0907 | 2 |
| inventory | FOODS_3_124_CA_3_validation | I2 | B | 128.0000 | 2.6360 | 0.1238 | 2 |
| inventory | FOODS_3_124_CA_3_validation | I2 | clip | 1.0000 | 2.7071 | 0.3287 | 2 |
| inventory | FOODS_3_124_CA_3_validation | I2 | clip | 5.0000 | 2.7185 | 0.3450 | 2 |
| inventory | FOODS_3_124_CA_3_validation | I2 | clip | 25.0000 | 2.9537 | 0.0264 | 2 |
| inventory | FOODS_3_124_CA_3_validation | I2 | gamma | 0.9000 | 2.9355 | 0.0080 | 2 |
| inventory | FOODS_3_124_CA_3_validation | I2 | gamma | 0.9900 | 2.7185 | 0.3450 | 2 |
| inventory | FOODS_3_124_CA_3_validation | I2 | gamma | 1.0000 | 2.6657 | 0.2703 | 2 |
| inventory | FOODS_3_124_CA_3_validation | I2 | horizon | 30.0000 | 2.4580 | 0.0076 | 2 |
| inventory | FOODS_3_124_CA_3_validation | I2 | horizon | 90.0000 | 2.7185 | 0.3450 | 2 |
| inventory | FOODS_3_124_CA_3_validation | I2 | horizon | 180.0000 | 3.7222 | 0.4230 | 2 |
| inventory | FOODS_3_124_CA_3_validation | I2 | lr | 0.0333 | 2.7059 | 0.0086 | 2 |
| inventory | FOODS_3_124_CA_3_validation | I2 | lr | 0.1000 | 2.7185 | 0.3450 | 2 |
| inventory | FOODS_3_124_CA_3_validation | I2 | lr | 0.3000 | 2.7197 | 0.3466 | 2 |
| inventory | FOODS_3_124_CA_3_validation | I2 | sigma0 | 0.2500 | 2.7185 | 0.3449 | 2 |
| inventory | FOODS_3_124_CA_3_validation | I2 | sigma0 | 0.5000 | 2.7185 | 0.3450 | 2 |
| inventory | FOODS_3_124_CA_3_validation | I2 | sigma0 | 1.0000 | 2.7417 | 0.3778 | 2 |
| trading | TECHM | T1 | B | 16.0000 | 0.8232 | 0.2689 | 2 |
| trading | TECHM | T1 | B | 32.0000 | 0.6933 | 0.3825 | 2 |
| trading | TECHM | T1 | B | 64.0000 | 0.8192 | 0.0715 | 2 |
| trading | TECHM | T1 | B | 128.0000 | 0.8643 | 0.0077 | 2 |
| trading | TECHM | T1 | clip | 1.0000 | 0.6443 | 0.3188 | 2 |
| trading | TECHM | T1 | clip | 5.0000 | 0.6933 | 0.3825 | 2 |
| trading | TECHM | T1 | clip | 25.0000 | 0.6933 | 0.3825 | 2 |
| trading | TECHM | T1 | gamma | 0.9000 | 0.8971 | 0.1156 | 2 |
| trading | TECHM | T1 | gamma | 0.9900 | 0.8522 | 0.1072 | 2 |
| trading | TECHM | T1 | gamma | 1.0000 | 0.7312 | 0.3301 | 2 |
| trading | TECHM | T1 | horizon | 125.0000 | 0.4083 | 0.0172 | 2 |
| trading | TECHM | T1 | horizon | 250.0000 | 0.6933 | 0.3825 | 2 |
| trading | TECHM | T1 | horizon | 500.0000 | 1.9164 | 0.1487 | 2 |
| trading | TECHM | T1 | lr | 0.0333 | 0.8821 | 0.1782 | 2 |
| trading | TECHM | T1 | lr | 0.1000 | 0.6933 | 0.3825 | 2 |
| trading | TECHM | T1 | lr | 0.3000 | 0.5706 | 0.4230 | 2 |