# TRPO algorithm evaluation (quick mode)

Return units: validation return / reward normaliser (higher is better). `final` = last checkpoint, deterministic policy.

## Per problem / policy / variant

| key | policy | method | n | final | ci_lo | ci_hi | eps95_median | reached95 | worst_drop_median | seed_sd | converge | stall | collapse | diverge |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FOODS_2_128_WI_2_validation | I1 | trpo:ng_fixed | 3 | 0.5417 | 0.3151 | 0.6821 | 32.0000 | 1.0000 | -0.0504 | 0.1981 | 0.6667 | 0.0000 | 0.3333 | 0.0000 |
| FOODS_2_128_WI_2_validation | I1 | trpo:trpo | 3 | 0.4517 | 0.3266 | 0.5209 | 32.0000 | 1.0000 | -0.0650 | 0.1086 | 0.6667 | 0.0000 | 0.3333 | 0.0000 |
| FOODS_2_128_WI_2_validation | I1 | trpo:trpo_mc | 3 | 0.1655 | -0.1669 | 0.4883 | 32.0000 | 1.0000 | -0.0770 | 0.3277 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| FOODS_2_128_WI_2_validation | I1 | trpo:trpo_nostd | 3 | 0.4695 | 0.3384 | 0.5495 | 32.0000 | 1.0000 | -0.0829 | 0.1145 | 0.6667 | 0.0000 | 0.3333 | 0.0000 |
| FOODS_2_128_WI_2_validation | I1 | trpo:vpg | 3 | 0.5864 | 0.5556 | 0.6322 | 32.0000 | 1.0000 | -0.0263 | 0.0405 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_2_128_WI_2_validation | I2 | trpo:ng_fixed | 3 | 0.7385 | 0.7111 | 0.7677 | 2208.0000 | 1.0000 | -0.0773 | 0.0283 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_2_128_WI_2_validation | I2 | trpo:trpo | 3 | 0.7585 | 0.7403 | 0.7776 | 800.0000 | 1.0000 | -0.0613 | 0.0187 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_2_128_WI_2_validation | I2 | trpo:trpo_mc | 3 | 0.5110 | 0.3315 | 0.6575 | 1504.0000 | 0.6667 | -0.1042 | 0.1655 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| FOODS_2_128_WI_2_validation | I2 | trpo:trpo_nostd | 3 | 0.7404 | 0.7184 | 0.7720 | 160.0000 | 1.0000 | -0.0544 | 0.0281 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_2_128_WI_2_validation | I2 | trpo:vpg | 3 | 0.6897 | 0.6086 | 0.7490 | 544.0000 | 0.6667 | -0.3672 | 0.0727 | 0.6667 | 0.0000 | 0.3333 | 0.0000 |
| FOODS_2_128_WI_2_validation | I3 | trpo:ng_fixed | 3 | 0.5942 | 0.5237 | 0.6351 | 1568.0000 | 1.0000 | -17.4802 | 0.0614 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_2_128_WI_2_validation | I3 | trpo:trpo | 3 | 0.5833 | 0.4811 | 0.6608 | 928.0000 | 1.0000 | -0.4560 | 0.0924 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_2_128_WI_2_validation | I3 | trpo:trpo_mc | 3 | 0.1205 | -0.0300 | 0.2461 | 1952.0000 | 1.0000 | -0.5994 | 0.1397 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_2_128_WI_2_validation | I3 | trpo:trpo_nostd | 3 | 0.6233 | 0.5312 | 0.7085 | 800.0000 | 1.0000 | -0.4955 | 0.0889 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_2_128_WI_2_validation | I3 | trpo:vpg | 3 | -3.3659 | -8.6101 | 0.7597 | 160.0000 | 0.3333 | -0.2460 | 4.7840 | 0.6667 | 0.0000 | 0.3333 | 0.0000 |
| FOODS_3_124_CA_3_validation | I1 | trpo:ng_fixed | 3 | 2.8568 | 2.8229 | 2.8930 | 544.0000 | 1.0000 | -0.0778 | 0.0351 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_3_124_CA_3_validation | I1 | trpo:trpo | 3 | 2.8123 | 2.7880 | 2.8284 | 672.0000 | 1.0000 | -0.0797 | 0.0214 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_3_124_CA_3_validation | I1 | trpo:trpo_mc | 3 | 2.3503 | 2.2742 | 2.4547 | 672.0000 | 1.0000 | -0.0834 | 0.0935 | 0.3333 | 0.6667 | 0.0000 | 0.0000 |
| FOODS_3_124_CA_3_validation | I1 | trpo:trpo_nostd | 3 | 2.8182 | 2.7903 | 2.8517 | 544.0000 | 1.0000 | -0.0960 | 0.0311 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_3_124_CA_3_validation | I1 | trpo:vpg | 3 | 2.8774 | 2.8605 | 2.8876 | 800.0000 | 1.0000 | -0.0006 | 0.0147 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_3_124_CA_3_validation | I2 | trpo:ng_fixed | 3 | 2.9598 | 2.9010 | 3.0294 | 1440.0000 | 1.0000 | -0.0438 | 0.0649 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_3_124_CA_3_validation | I2 | trpo:trpo | 3 | 2.9633 | 2.9296 | 3.0212 | 1056.0000 | 1.0000 | -0.0632 | 0.0504 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_3_124_CA_3_validation | I2 | trpo:trpo_mc | 3 | 2.7170 | 2.4974 | 2.8324 | 1696.0000 | 1.0000 | -0.0627 | 0.1903 | 0.6667 | 0.0000 | 0.3333 | 0.0000 |
| FOODS_3_124_CA_3_validation | I2 | trpo:trpo_nostd | 3 | 2.9526 | 2.9168 | 2.9785 | 928.0000 | 1.0000 | -0.0575 | 0.0320 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_3_124_CA_3_validation | I2 | trpo:vpg | 3 | 2.8885 | 2.8452 | 2.9243 | 1440.0000 | 1.0000 | -1.7187 | 0.0401 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_3_124_CA_3_validation | I3 | trpo:ng_fixed | 3 | 2.4489 | 2.3632 | 2.5724 | 288.0000 | 1.0000 | -0.0606 | 0.1096 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_3_124_CA_3_validation | I3 | trpo:trpo | 3 | 2.5847 | 2.3033 | 2.7711 | 288.0000 | 1.0000 | -0.0652 | 0.2479 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_3_124_CA_3_validation | I3 | trpo:trpo_mc | 3 | 0.3279 | -3.6328 | 2.6145 | 416.0000 | 1.0000 | -0.0641 | 3.4437 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_3_124_CA_3_validation | I3 | trpo:trpo_nostd | 3 | 2.6704 | 2.3115 | 2.9366 | 288.0000 | 1.0000 | -0.0458 | 0.3227 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_3_124_CA_3_validation | I3 | trpo:vpg | 3 | 2.9327 | 2.9228 | 2.9401 | 160.0000 | 1.0000 | -0.5246 | 0.0089 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| HOBBIES_1_371_WI_1_validation | I1 | trpo:ng_fixed | 3 | 3.3233 | 3.2951 | 3.3643 | 1312.0000 | 1.0000 | -0.0628 | 0.0364 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| HOBBIES_1_371_WI_1_validation | I1 | trpo:trpo | 3 | 3.2484 | 3.2272 | 3.2823 | 1440.0000 | 1.0000 | -0.0769 | 0.0297 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| HOBBIES_1_371_WI_1_validation | I1 | trpo:trpo_mc | 3 | 3.1286 | 3.0891 | 3.1926 | 1440.0000 | 1.0000 | -0.0797 | 0.0560 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| HOBBIES_1_371_WI_1_validation | I1 | trpo:trpo_nostd | 3 | 3.2426 | 3.2121 | 3.2893 | 1312.0000 | 1.0000 | -0.0765 | 0.0410 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| HOBBIES_1_371_WI_1_validation | I1 | trpo:vpg | 3 | 3.1792 | 3.0824 | 3.3070 | 928.0000 | 0.6667 | -0.0780 | 0.1155 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| HOBBIES_1_371_WI_1_validation | I2 | trpo:ng_fixed | 3 | 3.1201 | 2.7784 | 3.2953 | 1056.0000 | 0.6667 | -0.1033 | 0.2959 | 0.6667 | 0.3333 | 0.0000 | 0.0000 |
| HOBBIES_1_371_WI_1_validation | I2 | trpo:trpo | 3 | 3.2374 | 3.1873 | 3.2887 | 1056.0000 | 1.0000 | -0.1153 | 0.0507 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| HOBBIES_1_371_WI_1_validation | I2 | trpo:trpo_mc | 3 | 2.9492 | 2.8536 | 3.0891 | 1056.0000 | 0.6667 | -0.1095 | 0.1238 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| HOBBIES_1_371_WI_1_validation | I2 | trpo:trpo_nostd | 3 | 3.2454 | 3.2062 | 3.2874 | 1184.0000 | 1.0000 | -0.1082 | 0.0407 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| HOBBIES_1_371_WI_1_validation | I2 | trpo:vpg | 3 | 3.2920 | 3.2369 | 3.3371 | 672.0000 | 1.0000 | -0.9585 | 0.0508 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| HOBBIES_1_371_WI_1_validation | I3 | trpo:ng_fixed | 3 | 3.2446 | 3.1766 | 3.3086 | 672.0000 | 1.0000 | -0.1058 | 0.0661 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| HOBBIES_1_371_WI_1_validation | I3 | trpo:trpo | 3 | 3.2545 | 3.1934 | 3.3356 | 672.0000 | 1.0000 | -0.0952 | 0.0732 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| HOBBIES_1_371_WI_1_validation | I3 | trpo:trpo_mc | 3 | 2.9537 | 2.8342 | 3.0714 | 2208.0000 | 1.0000 | -0.1713 | 0.1186 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| HOBBIES_1_371_WI_1_validation | I3 | trpo:trpo_nostd | 3 | 3.2161 | 3.1839 | 3.2787 | 672.0000 | 1.0000 | -0.0671 | 0.0543 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| HOBBIES_1_371_WI_1_validation | I3 | trpo:vpg | 3 | 1.8345 | -0.5984 | 3.2953 | 544.0000 | 1.0000 | -0.6541 | 2.1211 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| TECHM | T1 | trpo:ng_fixed | 3 | 0.8892 | 0.7169 | 1.0002 | 1056.0000 | 1.0000 | -0.1057 | 0.1512 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| TECHM | T1 | trpo:trpo | 3 | 0.6100 | 0.4107 | 0.8502 | 352.0000 | 0.6667 | -0.0549 | 0.2226 | 0.6667 | 0.3333 | 0.0000 | 0.0000 |
| TECHM | T1 | trpo:trpo_mc | 3 | 0.7230 | 0.4590 | 0.9000 | 1568.0000 | 0.6667 | -0.0622 | 0.2330 | 0.6667 | 0.3333 | 0.0000 | 0.0000 |
| TECHM | T1 | trpo:trpo_nostd | 3 | 0.5181 | 0.2163 | 0.7440 | 1632.0000 | 0.6667 | -0.0691 | 0.2719 | 0.6667 | 0.0000 | 0.3333 | 0.0000 |
| TECHM | T1 | trpo:vpg | 3 | 0.7868 | 0.5511 | 1.0180 | 672.0000 | 1.0000 | -0.0080 | 0.2335 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| TECHM | T2 | trpo:ng_fixed | 3 | 0.6270 | 0.2609 | 0.9205 | 928.0000 | 1.0000 | -0.0407 | 0.3357 | 0.6667 | 0.3333 | 0.0000 | 0.0000 |
| TECHM | T2 | trpo:trpo | 3 | 0.6225 | 0.4232 | 0.9843 | 544.0000 | 1.0000 | -0.0317 | 0.3139 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| TECHM | T2 | trpo:trpo_mc | 3 | 0.6752 | 0.6286 | 0.7429 | 544.0000 | 1.0000 | -0.0270 | 0.0600 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| TECHM | T2 | trpo:trpo_nostd | 3 | 0.5210 | 0.3527 | 0.6519 | 544.0000 | 1.0000 | -0.0276 | 0.1530 | 0.6667 | 0.3333 | 0.0000 | 0.0000 |
| TECHM | T2 | trpo:vpg | 3 | 0.4981 | 0.3374 | 0.7145 | 1568.0000 | 0.6667 | -0.0041 | 0.1946 | 0.6667 | 0.3333 | 0.0000 | 0.0000 |

## Update quality (trust-region check)

| key | policy | method | frac_improved | mean_KL | max_KL | backtracks | t_update_ms | t_collect_ms |
|---|---|---|---|---|---|---|---|---|
| FOODS_2_128_WI_2_validation | I1 | trpo:ng_fixed | 1.0000 | 0.0301 | 0.0354 | 0.0000 | 2.2731 | 6.3575 |
| FOODS_2_128_WI_2_validation | I1 | trpo:trpo | 1.0000 | 0.0173 | 0.0300 | 0.5251 | 2.1952 | 6.1745 |
| FOODS_2_128_WI_2_validation | I1 | trpo:trpo_mc | 1.0000 | 0.0178 | 0.0300 | 0.4946 | 2.1359 | 6.0978 |
| FOODS_2_128_WI_2_validation | I1 | trpo:trpo_nostd | 1.0000 | 0.0170 | 0.0300 | 0.5412 | 2.1508 | 6.1848 |
| FOODS_2_128_WI_2_validation | I1 | trpo:vpg | 0.8763 | 0.0058 | 0.1053 | 0.0000 | 1.3759 | 5.9265 |
| FOODS_2_128_WI_2_validation | I2 | trpo:ng_fixed | 1.0000 | 0.0304 | 0.0355 | 0.0000 | 2.1762 | 6.2360 |
| FOODS_2_128_WI_2_validation | I2 | trpo:trpo | 1.0000 | 0.0161 | 0.0300 | 0.5860 | 2.2844 | 6.6710 |
| FOODS_2_128_WI_2_validation | I2 | trpo:trpo_mc | 1.0000 | 0.0173 | 0.0300 | 0.5215 | 1.9396 | 5.7295 |
| FOODS_2_128_WI_2_validation | I2 | trpo:trpo_nostd | 1.0000 | 0.0159 | 0.0300 | 0.5950 | 2.0547 | 6.0259 |
| FOODS_2_128_WI_2_validation | I2 | trpo:vpg | 0.7240 | 0.1969 | 4.7503 | 0.0000 | 1.4260 | 6.3862 |
| FOODS_2_128_WI_2_validation | I3 | trpo:ng_fixed | 1.0000 | 0.0408 | 3.2404 | 0.0000 | 23.9294 | 13.4085 |
| FOODS_2_128_WI_2_validation | I3 | trpo:trpo | 1.0000 | 0.0198 | 0.0300 | 0.4158 | 23.2054 | 13.5276 |
| FOODS_2_128_WI_2_validation | I3 | trpo:trpo_mc | 1.0000 | 0.0188 | 0.0300 | 0.4606 | 23.7071 | 13.7226 |
| FOODS_2_128_WI_2_validation | I3 | trpo:trpo_nostd | 1.0000 | 0.0194 | 0.0300 | 0.4355 | 26.2254 | 15.5351 |
| FOODS_2_128_WI_2_validation | I3 | trpo:vpg | 0.8441 | 0.0238 | 3.8044 | 0.0000 | 6.7666 | 13.6834 |
| FOODS_3_124_CA_3_validation | I1 | trpo:ng_fixed | 1.0000 | 0.0302 | 0.0355 | 0.0000 | 2.4897 | 7.2441 |
| FOODS_3_124_CA_3_validation | I1 | trpo:trpo | 1.0000 | 0.0165 | 0.0300 | 0.5681 | 2.5881 | 7.6591 |
| FOODS_3_124_CA_3_validation | I1 | trpo:trpo_mc | 1.0000 | 0.0172 | 0.0300 | 0.5287 | 2.3270 | 6.6817 |
| FOODS_3_124_CA_3_validation | I1 | trpo:trpo_nostd | 1.0000 | 0.0166 | 0.0300 | 0.5645 | 2.3390 | 6.9034 |
| FOODS_3_124_CA_3_validation | I1 | trpo:vpg | 0.9068 | 0.0006 | 0.0074 | 0.0000 | 1.6825 | 7.5482 |
| FOODS_3_124_CA_3_validation | I2 | trpo:ng_fixed | 1.0000 | 0.0303 | 0.0355 | 0.0000 | 2.4355 | 7.0240 |
| FOODS_3_124_CA_3_validation | I2 | trpo:trpo | 1.0000 | 0.0167 | 0.0300 | 0.5538 | 2.5015 | 7.2319 |
| FOODS_3_124_CA_3_validation | I2 | trpo:trpo_mc | 1.0000 | 0.0169 | 0.0300 | 0.5394 | 2.4145 | 6.9878 |
| FOODS_3_124_CA_3_validation | I2 | trpo:trpo_nostd | 1.0000 | 0.0168 | 0.0300 | 0.5520 | 2.3060 | 6.6639 |
| FOODS_3_124_CA_3_validation | I2 | trpo:vpg | 0.7616 | 0.1635 | 6.3868 | 0.0000 | 1.5938 | 7.2368 |
| FOODS_3_124_CA_3_validation | I3 | trpo:ng_fixed | 1.0000 | 0.0300 | 0.1492 | 0.0000 | 28.6059 | 15.9569 |
| FOODS_3_124_CA_3_validation | I3 | trpo:trpo | 1.0000 | 0.0199 | 0.0300 | 0.4140 | 26.6058 | 15.4655 |
| FOODS_3_124_CA_3_validation | I3 | trpo:trpo_mc | 1.0000 | 0.0199 | 0.0300 | 0.4122 | 25.7172 | 15.0815 |
| FOODS_3_124_CA_3_validation | I3 | trpo:trpo_nostd | 1.0000 | 0.0186 | 0.0300 | 0.4839 | 28.0628 | 14.3598 |
| FOODS_3_124_CA_3_validation | I3 | trpo:vpg | 0.8692 | 0.0340 | 6.2841 | 0.0000 | 7.8030 | 16.0090 |
| HOBBIES_1_371_WI_1_validation | I1 | trpo:ng_fixed | 1.0000 | 0.0100 | 0.0110 | 0.0000 | 3.5336 | 10.7554 |
| HOBBIES_1_371_WI_1_validation | I1 | trpo:trpo | 1.0000 | 0.0060 | 0.0100 | 0.5054 | 2.5072 | 7.0679 |
| HOBBIES_1_371_WI_1_validation | I1 | trpo:trpo_mc | 1.0000 | 0.0061 | 0.0100 | 0.4892 | 3.4696 | 10.3298 |
| HOBBIES_1_371_WI_1_validation | I1 | trpo:trpo_nostd | 1.0000 | 0.0061 | 0.0100 | 0.5000 | 3.1867 | 9.6493 |
| HOBBIES_1_371_WI_1_validation | I1 | trpo:vpg | 0.8674 | 0.0046 | 0.1012 | 0.0000 | 2.0892 | 9.4545 |
| HOBBIES_1_371_WI_1_validation | I2 | trpo:ng_fixed | 1.0000 | 0.0303 | 0.0355 | 0.0000 | 3.3381 | 10.3308 |
| HOBBIES_1_371_WI_1_validation | I2 | trpo:trpo | 1.0000 | 0.0163 | 0.0300 | 0.5753 | 3.3855 | 10.0879 |
| HOBBIES_1_371_WI_1_validation | I2 | trpo:trpo_mc | 1.0000 | 0.0176 | 0.0300 | 0.5054 | 3.2463 | 9.7177 |
| HOBBIES_1_371_WI_1_validation | I2 | trpo:trpo_nostd | 1.0000 | 0.0165 | 0.0300 | 0.5663 | 3.1928 | 9.7693 |
| HOBBIES_1_371_WI_1_validation | I2 | trpo:vpg | 0.7814 | 1.8977 | 912.3394 | 0.0000 | 2.1352 | 10.3741 |
| HOBBIES_1_371_WI_1_validation | I3 | trpo:ng_fixed | 1.0000 | 0.0104 | 0.2984 | 0.0000 | 35.4729 | 20.6742 |
| HOBBIES_1_371_WI_1_validation | I3 | trpo:trpo | 1.0000 | 0.0072 | 0.0100 | 0.3387 | 39.7141 | 20.2017 |
| HOBBIES_1_371_WI_1_validation | I3 | trpo:trpo_mc | 1.0000 | 0.0069 | 0.0100 | 0.3907 | 34.9040 | 20.8989 |
| HOBBIES_1_371_WI_1_validation | I3 | trpo:trpo_nostd | 1.0000 | 0.0070 | 0.0100 | 0.3746 | 41.9869 | 22.1854 |
| HOBBIES_1_371_WI_1_validation | I3 | trpo:vpg | 0.8208 | 0.0353 | 6.4631 | 0.0000 | 9.4740 | 20.2839 |
| TECHM | T1 | trpo:ng_fixed | 0.9749 | 0.0307 | 0.0678 | 0.0000 | 10.6945 | 19.7704 |
| TECHM | T1 | trpo:trpo | 1.0000 | 0.0172 | 0.0300 | 0.5287 | 10.6839 | 19.9274 |
| TECHM | T1 | trpo:trpo_mc | 1.0000 | 0.0173 | 0.0300 | 0.5215 | 10.8548 | 19.8451 |
| TECHM | T1 | trpo:trpo_nostd | 1.0000 | 0.0165 | 0.0300 | 0.5609 | 10.2485 | 19.6918 |
| TECHM | T1 | trpo:vpg | 0.8226 | 0.0000 | 0.0014 | 0.0000 | 6.6871 | 20.6053 |
| TECHM | T2 | trpo:ng_fixed | 1.0000 | 0.0100 | 0.0130 | 0.0000 | 43.1780 | 36.9626 |
| TECHM | T2 | trpo:trpo | 1.0000 | 0.0063 | 0.0100 | 0.4677 | 43.0857 | 38.4547 |
| TECHM | T2 | trpo:trpo_mc | 1.0000 | 0.0061 | 0.0100 | 0.4964 | 40.2168 | 34.8638 |
| TECHM | T2 | trpo:trpo_nostd | 1.0000 | 0.0064 | 0.0100 | 0.4391 | 36.5941 | 34.0739 |
| TECHM | T2 | trpo:vpg | 0.8728 | 0.0009 | 0.0500 | 0.0000 | 14.0047 | 38.4829 |

## TRPO update diagnostics

| key | policy | corr_surrogate_vs_realised | cos_CG_vs_exact | cg_residual | vf_r2 | frac_accepted |
|---|---|---|---|---|---|---|
| FOODS_2_128_WI_2_validation | I1 | 0.2203 | 1.0000 | 0.0001 | 0.3474 | 1.0000 |
| FOODS_2_128_WI_2_validation | I2 | 0.0146 | 1.0000 | 0.0000 | 0.4886 | 1.0000 |
| FOODS_2_128_WI_2_validation | I3 | 0.0214 | 0.9995 | 0.0063 | 0.4244 | 1.0000 |
| FOODS_3_124_CA_3_validation | I1 | 0.3348 | 1.0000 | 0.0001 | 0.8220 | 1.0000 |
| FOODS_3_124_CA_3_validation | I2 | -0.0659 | 1.0000 | 0.0000 | 0.8251 | 1.0000 |
| FOODS_3_124_CA_3_validation | I3 | 0.0930 | 0.9989 | 0.0119 | 0.7660 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I1 | 0.1997 | 1.0000 | 0.0001 | 0.3375 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I2 | 0.1151 | 1.0000 | 0.0000 | 0.4818 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I3 | -0.0130 | 0.9990 | 0.0065 | 0.4176 | 1.0000 |
| TECHM | T1 | 0.0855 | 0.9942 | 0.0016 | 0.0337 | 1.0000 |
| TECHM | T2 | 0.1034 | 1.0000 | 0.0003 | 0.0294 | 1.0000 |

## TRPO vs ablations (paired by seed, Wilcoxon, Holm-adjusted over all rows)

| key | policy | vs | diff_trpo_minus | p_wilcoxon | cliffs_delta | p_holm |
|---|---|---|---|---|---|---|
| FOODS_2_128_WI_2_validation | I1 | trpo:ng_fixed | -0.0900 | 0.5000 | -0.3333 | 1.0000 |
| FOODS_2_128_WI_2_validation | I1 | trpo:trpo_mc | 0.2862 | 0.2500 | 0.7778 | 1.0000 |
| FOODS_2_128_WI_2_validation | I1 | trpo:trpo_nostd | -0.0179 | 0.2500 | -0.3333 | 1.0000 |
| FOODS_2_128_WI_2_validation | I1 | trpo:vpg | -0.1347 | 0.2500 | -1.0000 | 1.0000 |
| FOODS_2_128_WI_2_validation | I2 | trpo:ng_fixed | 0.0200 | 0.7500 | 0.5556 | 1.0000 |
| FOODS_2_128_WI_2_validation | I2 | trpo:trpo_mc | 0.2475 | 0.2500 | 1.0000 | 1.0000 |
| FOODS_2_128_WI_2_validation | I2 | trpo:trpo_nostd | 0.0181 | 0.2500 | 0.5556 | 1.0000 |
| FOODS_2_128_WI_2_validation | I2 | trpo:vpg | 0.0688 | 0.5000 | 0.7778 | 1.0000 |
| FOODS_2_128_WI_2_validation | I3 | trpo:ng_fixed | -0.0109 | 0.7500 | -0.1111 | 1.0000 |
| FOODS_2_128_WI_2_validation | I3 | trpo:trpo_mc | 0.4629 | 0.2500 | 1.0000 | 1.0000 |
| FOODS_2_128_WI_2_validation | I3 | trpo:trpo_nostd | -0.0400 | 0.2500 | -0.3333 | 1.0000 |
| FOODS_2_128_WI_2_validation | I3 | trpo:vpg | 3.9493 | 0.5000 | 0.3333 | 1.0000 |
| FOODS_3_124_CA_3_validation | I1 | trpo:ng_fixed | -0.0446 | 0.2500 | -0.7778 | 1.0000 |
| FOODS_3_124_CA_3_validation | I1 | trpo:trpo_mc | 0.4620 | 0.2500 | 1.0000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I1 | trpo:trpo_nostd | -0.0059 | 0.7500 | -0.1111 | 1.0000 |
| FOODS_3_124_CA_3_validation | I1 | trpo:vpg | -0.0651 | 0.2500 | -1.0000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I2 | trpo:ng_fixed | 0.0035 | 1.0000 | -0.1111 | 1.0000 |
| FOODS_3_124_CA_3_validation | I2 | trpo:trpo_mc | 0.2463 | 0.2500 | 1.0000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I2 | trpo:trpo_nostd | 0.0107 | 0.7500 | 0.1111 | 1.0000 |
| FOODS_3_124_CA_3_validation | I2 | trpo:vpg | 0.0748 | 0.2500 | 1.0000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I3 | trpo:ng_fixed | 0.1358 | 0.5000 | 0.3333 | 1.0000 |
| FOODS_3_124_CA_3_validation | I3 | trpo:trpo_mc | 2.2568 | 0.2500 | 0.7778 | 1.0000 |
| FOODS_3_124_CA_3_validation | I3 | trpo:trpo_nostd | -0.0857 | 1.0000 | -0.3333 | 1.0000 |
| FOODS_3_124_CA_3_validation | I3 | trpo:vpg | -0.3480 | 0.2500 | -1.0000 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I1 | trpo:ng_fixed | -0.0749 | 0.2500 | -1.0000 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I1 | trpo:trpo_mc | 0.1198 | 0.2500 | 1.0000 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I1 | trpo:trpo_nostd | 0.0058 | 0.7500 | 0.3333 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I1 | trpo:vpg | 0.0692 | 0.5000 | 0.3333 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I2 | trpo:ng_fixed | 0.1174 | 0.7500 | -0.1111 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I2 | trpo:trpo_mc | 0.2883 | 0.2500 | 1.0000 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I2 | trpo:trpo_nostd | -0.0079 | 0.5000 | -0.1111 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I2 | trpo:vpg | -0.0546 | 0.2500 | -0.7778 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I3 | trpo:ng_fixed | 0.0099 | 0.5000 | 0.1111 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I3 | trpo:trpo_mc | 0.3008 | 0.2500 | 1.0000 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I3 | trpo:trpo_nostd | 0.0385 | 0.2500 | 0.5556 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I3 | trpo:vpg | 1.4200 | 0.5000 | 0.5556 | 1.0000 |
| TECHM | T1 | trpo:ng_fixed | -0.2792 | 0.2500 | -0.7778 | 1.0000 |
| TECHM | T1 | trpo:trpo_mc | -0.1131 | 0.7500 | -0.3333 | 1.0000 |
| TECHM | T1 | trpo:trpo_nostd | 0.0918 | 0.5000 | 0.1111 | 1.0000 |
| TECHM | T1 | trpo:vpg | -0.1768 | 0.5000 | -0.3333 | 1.0000 |
| TECHM | T2 | trpo:ng_fixed | -0.0045 | 1.0000 | 0.1111 | 1.0000 |
| TECHM | T2 | trpo:trpo_mc | -0.0527 | 1.0000 | -0.3333 | 1.0000 |
| TECHM | T2 | trpo:trpo_nostd | 0.1015 | 1.0000 | 0.1111 | 1.0000 |
| TECHM | T2 | trpo:vpg | 0.1244 | 0.2500 | 0.3333 | 1.0000 |

## Sensitivity (one at a time; best validation return, mean over seeds)

| domain | key | policy | variable | value | mean | std | count |
|---|---|---|---|---|---|---|---|
| inventory | FOODS_3_124_CA_3_validation | I2 | B | 128 | 2.9549 | 0.0711 | 2 |
| inventory | FOODS_3_124_CA_3_validation | I2 | B | 16 | 2.9251 | 0.0511 | 2 |
| inventory | FOODS_3_124_CA_3_validation | I2 | B | 32 | 2.9785 | 0.0550 | 2 |
| inventory | FOODS_3_124_CA_3_validation | I2 | B | 64 | 2.9492 | 0.0669 | 2 |
| inventory | FOODS_3_124_CA_3_validation | I2 | B | 8 | 2.8929 | 0.0459 | 2 |
| inventory | FOODS_3_124_CA_3_validation | I2 | cg_iters | 10 | 2.9713 | 0.0662 | 2 |
| inventory | FOODS_3_124_CA_3_validation | I2 | cg_iters | 2 | 2.8945 | 0.0404 | 2 |
| inventory | FOODS_3_124_CA_3_validation | I2 | cg_iters | 20 | 2.9785 | 0.0550 | 2 |
| inventory | FOODS_3_124_CA_3_validation | I2 | cg_iters | 5 | 2.9701 | 0.0465 | 2 |
| inventory | FOODS_3_124_CA_3_validation | I2 | damp | 0.0001 | 2.9525 | 0.0076 | 2 |
| inventory | FOODS_3_124_CA_3_validation | I2 | damp | 0.01 | 2.9785 | 0.0550 | 2 |
| inventory | FOODS_3_124_CA_3_validation | I2 | damp | 0.1 | 2.8978 | 0.0050 | 2 |
| inventory | FOODS_3_124_CA_3_validation | I2 | damp | 1.0 | 2.8712 | 0.0097 | 2 |
| inventory | FOODS_3_124_CA_3_validation | I2 | delta | 0.003 | 2.9319 | 0.0515 | 2 |
| inventory | FOODS_3_124_CA_3_validation | I2 | delta | 0.01 | 2.9518 | 0.0336 | 2 |
| inventory | FOODS_3_124_CA_3_validation | I2 | delta | 0.03 | 2.9785 | 0.0550 | 2 |
| inventory | FOODS_3_124_CA_3_validation | I2 | delta | 0.1 | 2.9497 | 0.0423 | 2 |
| inventory | FOODS_3_124_CA_3_validation | I2 | gamma | 0.9 | 2.9824 | 0.0387 | 2 |
| inventory | FOODS_3_124_CA_3_validation | I2 | gamma | 0.99 | 2.9785 | 0.0550 | 2 |
| inventory | FOODS_3_124_CA_3_validation | I2 | gamma | 1.0 | 2.8906 | 0.0711 | 2 |
| inventory | FOODS_3_124_CA_3_validation | I2 | horizon | 180 | 4.0934 | 0.0832 | 2 |
| inventory | FOODS_3_124_CA_3_validation | I2 | horizon | 30 | 2.5058 | 0.0144 | 2 |
| inventory | FOODS_3_124_CA_3_validation | I2 | horizon | 90 | 2.9785 | 0.0550 | 2 |
| inventory | FOODS_3_124_CA_3_validation | I2 | lam | 0.0 | 2.7045 | 0.2225 | 2 |
| inventory | FOODS_3_124_CA_3_validation | I2 | lam | 0.9 | 3.0191 | 0.0349 | 2 |
| inventory | FOODS_3_124_CA_3_validation | I2 | lam | 0.97 | 2.9785 | 0.0550 | 2 |
| inventory | FOODS_3_124_CA_3_validation | I2 | lam | 1.0 | 2.8854 | 0.1093 | 2 |
| inventory | FOODS_3_124_CA_3_validation | I2 | sigma0 | 0.25 | 2.9696 | 0.0591 | 2 |
| inventory | FOODS_3_124_CA_3_validation | I2 | sigma0 | 0.5 | 2.9785 | 0.0550 | 2 |
| inventory | FOODS_3_124_CA_3_validation | I2 | sigma0 | 1.0 | 2.9032 | 0.0594 | 2 |
| inventory | FOODS_3_124_CA_3_validation | I2 | std | False | 2.9130 | 0.0617 | 2 |
| inventory | FOODS_3_124_CA_3_validation | I2 | std | True | 2.9785 | 0.0550 | 2 |
| trading | TECHM | T1 | B | 128 | 0.9534 | 0.0087 | 2 |
| trading | TECHM | T1 | B | 16 | 1.0616 | 0.0228 | 2 |
| trading | TECHM | T1 | B | 32 | 1.0669 | 0.0306 | 2 |
| trading | TECHM | T1 | B | 64 | 1.0841 | 0.1506 | 2 |
| trading | TECHM | T1 | B | 8 | 1.0966 | 0.0019 | 2 |
| trading | TECHM | T1 | cg_iters | 10 | 1.0669 | 0.0306 | 2 |
| trading | TECHM | T1 | cg_iters | 2 | 1.0685 | 0.0271 | 2 |
| trading | TECHM | T1 | cg_iters | 20 | 1.0669 | 0.0306 | 2 |
| trading | TECHM | T1 | cg_iters | 5 | 1.0795 | 0.0561 | 2 |
| trading | TECHM | T1 | damp | 0.0001 | 1.0679 | 0.1340 | 2 |
| trading | TECHM | T1 | damp | 0.01 | 1.0669 | 0.0306 | 2 |
| trading | TECHM | T1 | damp | 0.1 | 1.0276 | 0.1164 | 2 |
| trading | TECHM | T1 | damp | 1.0 | 0.9468 | 0.1218 | 2 |
| trading | TECHM | T1 | delta | 0.003 | 0.9312 | 0.0033 | 2 |
| trading | TECHM | T1 | delta | 0.01 | 0.9549 | 0.0114 | 2 |
| trading | TECHM | T1 | delta | 0.03 | 1.0669 | 0.0306 | 2 |
| trading | TECHM | T1 | delta | 0.1 | 0.9401 | 0.1691 | 2 |
| trading | TECHM | T1 | gamma | 0.9 | 0.9726 | 0.0857 | 2 |
| trading | TECHM | T1 | gamma | 0.995 | 1.0669 | 0.0306 | 2 |
| trading | TECHM | T1 | gamma | 1.0 | 0.9602 | 0.0317 | 2 |
| trading | TECHM | T1 | horizon | 125 | 0.3831 | 0.0184 | 2 |
| trading | TECHM | T1 | horizon | 250 | 1.0669 | 0.0306 | 2 |
| trading | TECHM | T1 | horizon | 500 | 1.7739 | 0.0562 | 2 |
| trading | TECHM | T1 | lam | 0.0 | 0.9689 | 0.1402 | 2 |
| trading | TECHM | T1 | lam | 0.9 | 0.9710 | 0.1991 | 2 |
| trading | TECHM | T1 | lam | 0.97 | 1.0669 | 0.0306 | 2 |
| trading | TECHM | T1 | lam | 1.0 | 1.0054 | 0.0685 | 2 |
| trading | TECHM | T1 | std | False | 0.9531 | 0.0465 | 2 |
| trading | TECHM | T1 | std | True | 1.0669 | 0.0306 | 2 |
