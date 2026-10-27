# TRPO algorithm evaluation (full mode)

Return units: validation return / reward normaliser (higher is better). `final` = last checkpoint, deterministic policy.

## Per problem / policy / variant

| key | policy | method | n | final | ci_lo | ci_hi | eps95_median | reached95 | worst_drop_median | seed_sd | converge | stall | collapse | diverge |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FOODS_2_128_WI_2_validation | I1 | trpo:ng_fixed | 10 | 0.7634 | 0.7535 | 0.7729 | 32.0000 | 1.0000 | -0.0291 | 0.0165 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_2_128_WI_2_validation | I1 | trpo:trpo | 10 | 0.6775 | 0.6594 | 0.6971 | 32.0000 | 1.0000 | -0.0427 | 0.0320 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_2_128_WI_2_validation | I1 | trpo:trpo_mc | 10 | -0.3298 | -0.4994 | -0.2012 | 32.0000 | 1.0000 | -0.3094 | 0.2549 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| FOODS_2_128_WI_2_validation | I1 | trpo:trpo_nostd | 10 | 0.6790 | 0.6611 | 0.6986 | 32.0000 | 1.0000 | -0.0436 | 0.0320 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_2_128_WI_2_validation | I1 | trpo:vpg | 10 | 0.7471 | 0.7339 | 0.7593 | 32.0000 | 1.0000 | -0.0060 | 0.0216 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_2_128_WI_2_validation | I2 | trpo:ng_fixed | 10 | 0.7411 | 0.7163 | 0.7659 | 3504.0000 | 1.0000 | -0.0462 | 0.0425 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_2_128_WI_2_validation | I2 | trpo:trpo | 10 | 0.7544 | 0.7239 | 0.7803 | 5488.0000 | 1.0000 | -0.0605 | 0.0477 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_2_128_WI_2_validation | I2 | trpo:trpo_mc | 10 | 0.5435 | 0.4613 | 0.6125 | 2016.0000 | 0.8000 | -0.0937 | 0.1281 | 0.5000 | 0.1000 | 0.4000 | 0.0000 |
| FOODS_2_128_WI_2_validation | I2 | trpo:trpo_nostd | 10 | 0.7515 | 0.7201 | 0.7787 | 5488.0000 | 1.0000 | -0.0586 | 0.0500 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_2_128_WI_2_validation | I2 | trpo:vpg | 10 | 0.6383 | 0.5739 | 0.6901 | 2016.0000 | 1.0000 | -0.5570 | 0.0964 | 0.8000 | 0.1000 | 0.1000 | 0.0000 |
| FOODS_2_128_WI_2_validation | I3 | trpo:ng_fixed | 10 | 0.5059 | 0.1046 | 0.7311 | 1024.0000 | 1.0000 | -4.2962 | 0.6118 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_2_128_WI_2_validation | I3 | trpo:trpo | 10 | 0.7031 | 0.6648 | 0.7426 | 1024.0000 | 1.0000 | -0.1041 | 0.0657 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_2_128_WI_2_validation | I3 | trpo:trpo_mc | 10 | 0.3610 | 0.2512 | 0.4573 | 5984.0000 | 1.0000 | -0.3942 | 0.1782 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_2_128_WI_2_validation | I3 | trpo:trpo_nostd | 10 | 0.6953 | 0.6716 | 0.7240 | 1024.0000 | 1.0000 | -0.0929 | 0.0451 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_2_128_WI_2_validation | I3 | trpo:vpg | 10 | 0.7732 | 0.7472 | 0.7969 | 1024.0000 | 1.0000 | -0.1474 | 0.0430 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_3_124_CA_3_validation | I1 | trpo:ng_fixed | 10 | 2.8421 | 2.8310 | 2.8526 | 528.0000 | 1.0000 | -0.0251 | 0.0188 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_3_124_CA_3_validation | I1 | trpo:trpo | 10 | 2.8145 | 2.8031 | 2.8266 | 528.0000 | 1.0000 | -0.0329 | 0.0195 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_3_124_CA_3_validation | I1 | trpo:trpo_mc | 10 | 2.6500 | 2.6285 | 2.6710 | 1024.0000 | 1.0000 | -0.0350 | 0.0371 | 0.5000 | 0.5000 | 0.0000 | 0.0000 |
| FOODS_3_124_CA_3_validation | I1 | trpo:trpo_nostd | 10 | 2.8033 | 2.7916 | 2.8164 | 528.0000 | 1.0000 | -0.0332 | 0.0215 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_3_124_CA_3_validation | I1 | trpo:vpg | 10 | 2.8618 | 2.8436 | 2.8785 | 528.0000 | 1.0000 | -0.0162 | 0.0308 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_3_124_CA_3_validation | I2 | trpo:ng_fixed | 10 | 2.9944 | 2.9671 | 3.0226 | 1024.0000 | 1.0000 | -0.0881 | 0.0476 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_3_124_CA_3_validation | I2 | trpo:trpo | 10 | 2.8979 | 2.8608 | 2.9349 | 1024.0000 | 1.0000 | -0.1651 | 0.0642 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_3_124_CA_3_validation | I2 | trpo:trpo_mc | 10 | 2.2364 | 2.2040 | 2.2622 | 1024.0000 | 1.0000 | -1.1391 | 0.0508 | 0.0000 | 0.8000 | 0.2000 | 0.0000 |
| FOODS_3_124_CA_3_validation | I2 | trpo:trpo_nostd | 10 | 2.8340 | 2.7779 | 2.8897 | 1024.0000 | 1.0000 | -0.1595 | 0.0969 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_3_124_CA_3_validation | I2 | trpo:vpg | 10 | 2.8780 | 2.8157 | 2.9408 | 1024.0000 | 1.0000 | -0.6484 | 0.1065 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_3_124_CA_3_validation | I3 | trpo:ng_fixed | 10 | 3.0354 | 3.0054 | 3.0630 | 1024.0000 | 1.0000 | -0.0324 | 0.0498 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_3_124_CA_3_validation | I3 | trpo:trpo | 10 | 3.0481 | 3.0224 | 3.0739 | 1024.0000 | 1.0000 | -0.0311 | 0.0446 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_3_124_CA_3_validation | I3 | trpo:trpo_mc | 10 | 2.8009 | 2.7444 | 2.8444 | 4000.0000 | 1.0000 | -0.0338 | 0.0850 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_3_124_CA_3_validation | I3 | trpo:trpo_nostd | 10 | 3.0715 | 3.0468 | 3.0921 | 1024.0000 | 1.0000 | -0.0317 | 0.0391 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_3_124_CA_3_validation | I3 | trpo:vpg | 10 | 3.0170 | 3.0035 | 3.0335 | 1024.0000 | 1.0000 | -0.0585 | 0.0255 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| HOBBIES_1_371_WI_1_validation | I1 | trpo:ng_fixed | 10 | 3.3122 | 3.2976 | 3.3245 | 2016.0000 | 1.0000 | -0.0319 | 0.0232 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| HOBBIES_1_371_WI_1_validation | I1 | trpo:trpo | 10 | 3.2546 | 3.2258 | 3.2831 | 2512.0000 | 1.0000 | -0.0611 | 0.0496 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| HOBBIES_1_371_WI_1_validation | I1 | trpo:trpo_mc | 10 | 2.9686 | 2.8216 | 3.0879 | 3008.0000 | 1.0000 | -0.0594 | 0.2325 | 0.9000 | 0.1000 | 0.0000 | 0.0000 |
| HOBBIES_1_371_WI_1_validation | I1 | trpo:trpo_nostd | 10 | 3.2565 | 3.2218 | 3.2910 | 2016.0000 | 1.0000 | -0.0575 | 0.0593 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| HOBBIES_1_371_WI_1_validation | I1 | trpo:vpg | 10 | 3.1010 | 3.0300 | 3.1759 | 1024.0000 | 1.0000 | -0.0904 | 0.1251 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| HOBBIES_1_371_WI_1_validation | I2 | trpo:ng_fixed | 10 | 3.1931 | 3.1145 | 3.2531 | 1520.0000 | 1.0000 | -0.1117 | 0.1196 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| HOBBIES_1_371_WI_1_validation | I2 | trpo:trpo | 10 | 3.1495 | 3.0733 | 3.2166 | 1520.0000 | 1.0000 | -0.2442 | 0.1234 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| HOBBIES_1_371_WI_1_validation | I2 | trpo:trpo_mc | 10 | 2.4618 | 2.2976 | 2.6766 | 3504.0000 | 1.0000 | -0.4916 | 0.3382 | 0.2000 | 0.0000 | 0.8000 | 0.0000 |
| HOBBIES_1_371_WI_1_validation | I2 | trpo:trpo_nostd | 10 | 3.1602 | 3.0960 | 3.2157 | 1520.0000 | 1.0000 | -0.2389 | 0.1050 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| HOBBIES_1_371_WI_1_validation | I2 | trpo:vpg | 10 | 3.2758 | 3.2340 | 3.3164 | 2016.0000 | 1.0000 | -0.2613 | 0.0711 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| HOBBIES_1_371_WI_1_validation | I3 | trpo:ng_fixed | 10 | 3.2675 | 3.2477 | 3.2853 | 1024.0000 | 1.0000 | -1.1373 | 0.0335 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| HOBBIES_1_371_WI_1_validation | I3 | trpo:trpo | 10 | 3.2290 | 3.1982 | 3.2562 | 1024.0000 | 1.0000 | -0.1099 | 0.0499 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| HOBBIES_1_371_WI_1_validation | I3 | trpo:trpo_mc | 10 | 3.1034 | 3.0650 | 3.1399 | 3008.0000 | 1.0000 | -0.1880 | 0.0644 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| HOBBIES_1_371_WI_1_validation | I3 | trpo:trpo_nostd | 10 | 3.2341 | 3.2140 | 3.2524 | 1024.0000 | 1.0000 | -0.0870 | 0.0346 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| HOBBIES_1_371_WI_1_validation | I3 | trpo:vpg | 10 | 3.2335 | 3.0809 | 3.3514 | 1024.0000 | 1.0000 | -0.1428 | 0.2401 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| TECHM | T1 | trpo:ng_fixed | 10 | 0.7791 | 0.6010 | 0.9213 | 1024.0000 | 1.0000 | -0.0715 | 0.2749 | 0.8000 | 0.1000 | 0.1000 | 0.0000 |
| TECHM | T1 | trpo:trpo | 10 | 0.7694 | 0.6133 | 0.8935 | 1024.0000 | 0.8000 | -0.0579 | 0.2326 | 0.8000 | 0.1000 | 0.1000 | 0.0000 |
| TECHM | T1 | trpo:trpo_mc | 10 | 0.6399 | 0.5021 | 0.7766 | 1520.0000 | 0.8000 | -0.0587 | 0.2345 | 0.7000 | 0.1000 | 0.2000 | 0.0000 |
| TECHM | T1 | trpo:trpo_nostd | 10 | 0.7062 | 0.5519 | 0.8340 | 1024.0000 | 0.9000 | -0.0578 | 0.2390 | 0.7000 | 0.2000 | 0.1000 | 0.0000 |
| TECHM | T1 | trpo:vpg | 10 | 0.6663 | 0.5076 | 0.8129 | 1024.0000 | 0.8000 | -0.0232 | 0.2647 | 0.7000 | 0.1000 | 0.2000 | 0.0000 |
| TECHM | T2 | trpo:ng_fixed | 10 | 0.5304 | 0.4245 | 0.6123 | 1024.0000 | 1.0000 | -0.0887 | 0.1578 | 0.8000 | 0.1000 | 0.1000 | 0.0000 |
| TECHM | T2 | trpo:trpo | 10 | 0.6804 | 0.5689 | 0.7776 | 1024.0000 | 0.9000 | -0.0695 | 0.1778 | 0.8000 | 0.2000 | 0.0000 | 0.0000 |
| TECHM | T2 | trpo:trpo_mc | 10 | 0.5819 | 0.4779 | 0.6929 | 2512.0000 | 1.0000 | -0.0607 | 0.1881 | 0.7000 | 0.3000 | 0.0000 | 0.0000 |
| TECHM | T2 | trpo:trpo_nostd | 10 | 0.6682 | 0.5146 | 0.8177 | 1024.0000 | 0.8000 | -0.0746 | 0.2610 | 0.7000 | 0.3000 | 0.0000 | 0.0000 |
| TECHM | T2 | trpo:vpg | 10 | 0.6123 | 0.4791 | 0.7408 | 1024.0000 | 0.9000 | -0.0170 | 0.2311 | 0.8000 | 0.2000 | 0.0000 | 0.0000 |

## Update quality (trust-region check)

| key | policy | method | frac_improved | mean_KL | max_KL | backtracks | t_update_ms | t_collect_ms |
|---|---|---|---|---|---|---|---|---|
| FOODS_2_128_WI_2_validation | I1 | trpo:ng_fixed | 1.0000 | 0.0100 | 0.0110 | 0.0000 | 3.2462 | 9.7238 |
| FOODS_2_128_WI_2_validation | I1 | trpo:trpo | 1.0000 | 0.0053 | 0.0100 | 0.6098 | 3.4532 | 10.1710 |
| FOODS_2_128_WI_2_validation | I1 | trpo:trpo_mc | 1.0000 | 0.0055 | 0.0100 | 0.5791 | 2.9431 | 9.7177 |
| FOODS_2_128_WI_2_validation | I1 | trpo:trpo_nostd | 1.0000 | 0.0053 | 0.0100 | 0.6113 | 3.0718 | 9.5977 |
| FOODS_2_128_WI_2_validation | I1 | trpo:vpg | 0.8815 | 0.0001 | 0.0014 | 0.0000 | 2.0835 | 9.7944 |
| FOODS_2_128_WI_2_validation | I2 | trpo:ng_fixed | 1.0000 | 0.0100 | 0.0110 | 0.0000 | 3.3070 | 10.0088 |
| FOODS_2_128_WI_2_validation | I2 | trpo:trpo | 1.0000 | 0.0053 | 0.0100 | 0.6150 | 3.3108 | 10.1329 |
| FOODS_2_128_WI_2_validation | I2 | trpo:trpo_mc | 1.0000 | 0.0055 | 0.0100 | 0.5672 | 2.9601 | 10.1960 |
| FOODS_2_128_WI_2_validation | I2 | trpo:trpo_nostd | 1.0000 | 0.0053 | 0.0100 | 0.6166 | 3.4574 | 10.8593 |
| FOODS_2_128_WI_2_validation | I2 | trpo:vpg | 0.6604 | 670.5243 | 936092.6141 | 0.0000 | 1.9999 | 10.1255 |
| FOODS_2_128_WI_2_validation | I3 | trpo:ng_fixed | 0.9994 | 0.0934 | 463.2961 | 0.0000 | 26.9557 | 15.8813 |
| FOODS_2_128_WI_2_validation | I3 | trpo:trpo | 1.0000 | 0.0166 | 0.0300 | 0.5655 | 31.5427 | 19.0416 |
| FOODS_2_128_WI_2_validation | I3 | trpo:trpo_mc | 1.0000 | 0.0167 | 0.0300 | 0.5648 | 25.9109 | 15.4886 |
| FOODS_2_128_WI_2_validation | I3 | trpo:trpo_nostd | 1.0000 | 0.0166 | 0.0300 | 0.5637 | 25.4334 | 15.2387 |
| FOODS_2_128_WI_2_validation | I3 | trpo:vpg | 0.8344 | 0.0052 | 5.6899 | 0.0000 | 6.6362 | 14.9914 |
| FOODS_3_124_CA_3_validation | I1 | trpo:ng_fixed | 1.0000 | 0.0030 | 0.0032 | 0.0000 | 3.5924 | 10.7174 |
| FOODS_3_124_CA_3_validation | I1 | trpo:trpo | 1.0000 | 0.0016 | 0.0030 | 0.6023 | 3.6007 | 10.6323 |
| FOODS_3_124_CA_3_validation | I1 | trpo:trpo_mc | 1.0000 | 0.0017 | 0.0030 | 0.5725 | 3.4819 | 10.3823 |
| FOODS_3_124_CA_3_validation | I1 | trpo:trpo_nostd | 1.0000 | 0.0016 | 0.0030 | 0.6006 | 3.3558 | 10.2816 |
| FOODS_3_124_CA_3_validation | I1 | trpo:vpg | 0.8801 | 0.0008 | 0.0278 | 0.0000 | 2.3122 | 10.7469 |
| FOODS_3_124_CA_3_validation | I2 | trpo:ng_fixed | 0.9997 | 0.1010 | 0.1366 | 0.0000 | 3.7122 | 11.0584 |
| FOODS_3_124_CA_3_validation | I2 | trpo:trpo | 1.0000 | 0.0509 | 0.1000 | 0.6067 | 3.5780 | 10.9117 |
| FOODS_3_124_CA_3_validation | I2 | trpo:trpo_mc | 1.0000 | 0.0464 | 0.1000 | 0.6869 | 3.0079 | 10.9968 |
| FOODS_3_124_CA_3_validation | I2 | trpo:trpo_nostd | 1.0000 | 0.0509 | 0.1000 | 0.6037 | 3.3542 | 10.8890 |
| FOODS_3_124_CA_3_validation | I2 | trpo:vpg | 0.7185 | 1.2831 | 1528.0780 | 0.0000 | 2.2965 | 11.0690 |
| FOODS_3_124_CA_3_validation | I3 | trpo:ng_fixed | 1.0000 | 0.0030 | 0.0167 | 0.0000 | 41.5164 | 23.3155 |
| FOODS_3_124_CA_3_validation | I3 | trpo:trpo | 1.0000 | 0.0019 | 0.0030 | 0.4541 | 40.0151 | 22.6962 |
| FOODS_3_124_CA_3_validation | I3 | trpo:trpo_mc | 1.0000 | 0.0019 | 0.0030 | 0.4560 | 39.7046 | 22.6788 |
| FOODS_3_124_CA_3_validation | I3 | trpo:trpo_nostd | 1.0000 | 0.0019 | 0.0030 | 0.4688 | 41.3255 | 22.4014 |
| FOODS_3_124_CA_3_validation | I3 | trpo:vpg | 0.9014 | 0.0003 | 0.2601 | 0.0000 | 10.3031 | 25.3430 |
| HOBBIES_1_371_WI_1_validation | I1 | trpo:ng_fixed | 1.0000 | 0.0030 | 0.0032 | 0.0000 | 2.6663 | 7.8496 |
| HOBBIES_1_371_WI_1_validation | I1 | trpo:trpo | 1.0000 | 0.0017 | 0.0030 | 0.5820 | 2.8165 | 8.3286 |
| HOBBIES_1_371_WI_1_validation | I1 | trpo:trpo_mc | 1.0000 | 0.0017 | 0.0030 | 0.5571 | 2.1037 | 6.1000 |
| HOBBIES_1_371_WI_1_validation | I1 | trpo:trpo_nostd | 1.0000 | 0.0017 | 0.0030 | 0.5811 | 2.5469 | 7.5809 |
| HOBBIES_1_371_WI_1_validation | I1 | trpo:vpg | 0.8542 | 0.0834 | 6.2905 | 0.0000 | 2.0995 | 9.8613 |
| HOBBIES_1_371_WI_1_validation | I2 | trpo:ng_fixed | 1.0000 | 0.0301 | 0.0355 | 0.0000 | 3.4271 | 10.4530 |
| HOBBIES_1_371_WI_1_validation | I2 | trpo:trpo | 1.0000 | 0.0157 | 0.0300 | 0.6042 | 3.4403 | 10.7360 |
| HOBBIES_1_371_WI_1_validation | I2 | trpo:trpo_mc | 1.0000 | 0.0156 | 0.0300 | 0.6032 | 2.8667 | 10.2218 |
| HOBBIES_1_371_WI_1_validation | I2 | trpo:trpo_nostd | 1.0000 | 0.0158 | 0.0300 | 0.6002 | 3.2844 | 10.4295 |
| HOBBIES_1_371_WI_1_validation | I2 | trpo:vpg | 0.7664 | 0.0358 | 20.1437 | 0.0000 | 2.2365 | 10.9586 |
| HOBBIES_1_371_WI_1_validation | I3 | trpo:ng_fixed | 1.0000 | 0.0104 | 1.6310 | 0.0000 | 38.8935 | 23.4585 |
| HOBBIES_1_371_WI_1_validation | I3 | trpo:trpo | 1.0000 | 0.0062 | 0.0100 | 0.4769 | 38.6443 | 22.5423 |
| HOBBIES_1_371_WI_1_validation | I3 | trpo:trpo_mc | 1.0000 | 0.0060 | 0.0100 | 0.5143 | 37.3343 | 22.0975 |
| HOBBIES_1_371_WI_1_validation | I3 | trpo:trpo_nostd | 1.0000 | 0.0060 | 0.0100 | 0.5074 | 35.8673 | 21.8693 |
| HOBBIES_1_371_WI_1_validation | I3 | trpo:vpg | 0.8889 | 0.0011 | 1.9391 | 0.0000 | 10.3443 | 24.6764 |
| TECHM | T1 | trpo:ng_fixed | 0.9602 | 0.0108 | 0.1936 | 0.0000 | 9.2206 | 17.2747 |
| TECHM | T1 | trpo:trpo | 1.0000 | 0.0055 | 0.0100 | 0.5638 | 10.2434 | 19.3663 |
| TECHM | T1 | trpo:trpo_mc | 1.0000 | 0.0053 | 0.0100 | 0.5939 | 9.2902 | 17.3278 |
| TECHM | T1 | trpo:trpo_nostd | 1.0000 | 0.0055 | 0.0100 | 0.5573 | 9.1424 | 17.9028 |
| TECHM | T1 | trpo:vpg | 0.7805 | 0.0001 | 0.0174 | 0.0000 | 6.1737 | 18.8440 |
| TECHM | T2 | trpo:ng_fixed | 0.9769 | 0.0338 | 0.5646 | 0.0000 | 36.2742 | 33.0490 |
| TECHM | T2 | trpo:trpo | 1.0000 | 0.0155 | 0.0300 | 0.6267 | 39.0008 | 36.1661 |
| TECHM | T2 | trpo:trpo_mc | 1.0000 | 0.0157 | 0.0300 | 0.6127 | 33.8802 | 31.3469 |
| TECHM | T2 | trpo:trpo_nostd | 1.0000 | 0.0158 | 0.0300 | 0.6069 | 26.8318 | 26.9928 |
| TECHM | T2 | trpo:vpg | 0.8078 | 0.0002 | 0.0584 | 0.0000 | 12.2733 | 34.1602 |

## TRPO update diagnostics

| key | policy | corr_surrogate_vs_realised | cos_CG_vs_exact | cg_residual | vf_r2 | frac_accepted |
|---|---|---|---|---|---|---|
| FOODS_2_128_WI_2_validation | I1 | 0.0574 | 1.0000 | 0.0000 | 0.3840 | 1.0000 |
| FOODS_2_128_WI_2_validation | I2 | -0.0387 | 1.0000 | 0.0000 | 0.4791 | 1.0000 |
| FOODS_2_128_WI_2_validation | I3 | 0.0034 | 0.9979 | 0.0131 | 0.4829 | 1.0000 |
| FOODS_3_124_CA_3_validation | I1 | 0.1155 | 1.0000 | 0.0000 | 0.8311 | 1.0000 |
| FOODS_3_124_CA_3_validation | I2 | -0.0615 | 1.0000 | 0.0001 | 0.8207 | 1.0000 |
| FOODS_3_124_CA_3_validation | I3 | 0.2587 | 0.9981 | 0.0138 | 0.8443 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I1 | 0.0686 | 1.0000 | 0.0001 | 0.3180 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I2 | -0.0242 | 1.0000 | 0.0001 | 0.4395 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I3 | 0.0266 | 0.9972 | 0.0155 | 0.4795 | 1.0000 |
| TECHM | T1 | -0.0060 | 0.9819 | 0.0033 | 0.0575 | 1.0000 |
| TECHM | T2 | -0.0232 | 0.9960 | 0.0025 | 0.0569 | 1.0000 |

## TRPO vs ablations (paired by seed, Wilcoxon, Holm-adjusted over all rows)

| key | policy | vs | diff_trpo_minus | p_wilcoxon | cliffs_delta | p_holm |
|---|---|---|---|---|---|---|
| FOODS_2_128_WI_2_validation | I1 | trpo:ng_fixed | -0.0859 | 0.0020 | -1.0000 | 0.0859 |
| FOODS_2_128_WI_2_validation | I1 | trpo:trpo_mc | 1.0073 | 0.0020 | 1.0000 | 0.0859 |
| FOODS_2_128_WI_2_validation | I1 | trpo:trpo_nostd | -0.0015 | 0.8457 | -0.0400 | 1.0000 |
| FOODS_2_128_WI_2_validation | I1 | trpo:vpg | -0.0696 | 0.0020 | -0.9400 | 0.0859 |
| FOODS_2_128_WI_2_validation | I2 | trpo:ng_fixed | 0.0132 | 0.2754 | 0.2400 | 1.0000 |
| FOODS_2_128_WI_2_validation | I2 | trpo:trpo_mc | 0.2109 | 0.0020 | 0.9600 | 0.0859 |
| FOODS_2_128_WI_2_validation | I2 | trpo:trpo_nostd | 0.0029 | 0.1934 | 0.0600 | 1.0000 |
| FOODS_2_128_WI_2_validation | I2 | trpo:vpg | 0.1160 | 0.0098 | 0.8000 | 0.2930 |
| FOODS_2_128_WI_2_validation | I3 | trpo:ng_fixed | 0.1972 | 0.6250 | 0.0800 | 1.0000 |
| FOODS_2_128_WI_2_validation | I3 | trpo:trpo_mc | 0.3421 | 0.0020 | 1.0000 | 0.0859 |
| FOODS_2_128_WI_2_validation | I3 | trpo:trpo_nostd | 0.0078 | 1.0000 | 0.1600 | 1.0000 |
| FOODS_2_128_WI_2_validation | I3 | trpo:vpg | -0.0701 | 0.0488 | -0.5600 | 1.0000 |
| FOODS_3_124_CA_3_validation | I1 | trpo:ng_fixed | -0.0275 | 0.0195 | -0.6800 | 0.5469 |
| FOODS_3_124_CA_3_validation | I1 | trpo:trpo_mc | 0.1645 | 0.0020 | 1.0000 | 0.0859 |
| FOODS_3_124_CA_3_validation | I1 | trpo:trpo_nostd | 0.0113 | 0.0195 | 0.3200 | 0.5469 |
| FOODS_3_124_CA_3_validation | I1 | trpo:vpg | -0.0472 | 0.0020 | -0.8000 | 0.0859 |
| FOODS_3_124_CA_3_validation | I2 | trpo:ng_fixed | -0.0964 | 0.0020 | -0.7400 | 0.0859 |
| FOODS_3_124_CA_3_validation | I2 | trpo:trpo_mc | 0.6616 | 0.0020 | 1.0000 | 0.0859 |
| FOODS_3_124_CA_3_validation | I2 | trpo:trpo_nostd | 0.0639 | 0.1309 | 0.3400 | 1.0000 |
| FOODS_3_124_CA_3_validation | I2 | trpo:vpg | 0.0199 | 0.8457 | 0.0800 | 1.0000 |
| FOODS_3_124_CA_3_validation | I3 | trpo:ng_fixed | 0.0127 | 0.7695 | 0.1000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I3 | trpo:trpo_mc | 0.2473 | 0.0020 | 1.0000 | 0.0859 |
| FOODS_3_124_CA_3_validation | I3 | trpo:trpo_nostd | -0.0233 | 0.4316 | -0.2800 | 1.0000 |
| FOODS_3_124_CA_3_validation | I3 | trpo:vpg | 0.0311 | 0.1055 | 0.3800 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I1 | trpo:ng_fixed | -0.0576 | 0.0059 | -0.7200 | 0.1816 |
| HOBBIES_1_371_WI_1_validation | I1 | trpo:trpo_mc | 0.2860 | 0.0020 | 0.8800 | 0.0859 |
| HOBBIES_1_371_WI_1_validation | I1 | trpo:trpo_nostd | -0.0019 | 0.9219 | -0.0200 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I1 | trpo:vpg | 0.1536 | 0.0137 | 0.7400 | 0.3965 |
| HOBBIES_1_371_WI_1_validation | I2 | trpo:ng_fixed | -0.0435 | 0.5566 | -0.2000 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I2 | trpo:trpo_mc | 0.6878 | 0.0039 | 0.8200 | 0.1289 |
| HOBBIES_1_371_WI_1_validation | I2 | trpo:trpo_nostd | -0.0106 | 1.0000 | -0.0200 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I2 | trpo:vpg | -0.1263 | 0.0273 | -0.6600 | 0.7109 |
| HOBBIES_1_371_WI_1_validation | I3 | trpo:ng_fixed | -0.0384 | 0.0840 | -0.5200 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I3 | trpo:trpo_mc | 0.1256 | 0.0039 | 0.8800 | 0.1289 |
| HOBBIES_1_371_WI_1_validation | I3 | trpo:trpo_nostd | -0.0050 | 0.9219 | 0.0000 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I3 | trpo:vpg | -0.0044 | 0.4316 | -0.6000 | 1.0000 |
| TECHM | T1 | trpo:ng_fixed | -0.0097 | 0.8457 | -0.0800 | 1.0000 |
| TECHM | T1 | trpo:trpo_mc | 0.1295 | 0.2324 | 0.3400 | 1.0000 |
| TECHM | T1 | trpo:trpo_nostd | 0.0632 | 0.2324 | 0.1800 | 1.0000 |
| TECHM | T1 | trpo:vpg | 0.1031 | 0.3223 | 0.2800 | 1.0000 |
| TECHM | T2 | trpo:ng_fixed | 0.1500 | 0.0371 | 0.4800 | 0.9277 |
| TECHM | T2 | trpo:trpo_mc | 0.0985 | 0.3223 | 0.3200 | 1.0000 |
| TECHM | T2 | trpo:trpo_nostd | 0.0122 | 0.8457 | 0.0400 | 1.0000 |
| TECHM | T2 | trpo:vpg | 0.0680 | 0.4922 | 0.1800 | 1.0000 |

## Sensitivity (one at a time; best validation return, mean over seeds)

| domain | key | policy | variable | value | mean | std | count |
|---|---|---|---|---|---|---|---|
| inventory | FOODS_3_124_CA_3_validation | I2 | B | 128 | 3.0013 | 0.0189 | 5 |
| inventory | FOODS_3_124_CA_3_validation | I2 | B | 16 | 2.9832 | 0.0702 | 5 |
| inventory | FOODS_3_124_CA_3_validation | I2 | B | 32 | 3.0264 | 0.0278 | 5 |
| inventory | FOODS_3_124_CA_3_validation | I2 | B | 64 | 3.0214 | 0.0369 | 5 |
| inventory | FOODS_3_124_CA_3_validation | I2 | B | 8 | 2.7831 | 0.0752 | 5 |
| inventory | FOODS_3_124_CA_3_validation | I2 | cg_iters | 10 | 2.9962 | 0.0486 | 5 |
| inventory | FOODS_3_124_CA_3_validation | I2 | cg_iters | 2 | 3.0417 | 0.0341 | 5 |
| inventory | FOODS_3_124_CA_3_validation | I2 | cg_iters | 20 | 3.0264 | 0.0278 | 5 |
| inventory | FOODS_3_124_CA_3_validation | I2 | cg_iters | 5 | 3.0347 | 0.0413 | 5 |
| inventory | FOODS_3_124_CA_3_validation | I2 | damp | 0.0001 | 2.9982 | 0.0456 | 5 |
| inventory | FOODS_3_124_CA_3_validation | I2 | damp | 0.01 | 3.0264 | 0.0278 | 5 |
| inventory | FOODS_3_124_CA_3_validation | I2 | damp | 0.1 | 3.0348 | 0.0434 | 5 |
| inventory | FOODS_3_124_CA_3_validation | I2 | damp | 1.0 | 3.0572 | 0.0274 | 5 |
| inventory | FOODS_3_124_CA_3_validation | I2 | delta | 0.003 | 2.9792 | 0.0241 | 5 |
| inventory | FOODS_3_124_CA_3_validation | I2 | delta | 0.01 | 2.9936 | 0.0240 | 5 |
| inventory | FOODS_3_124_CA_3_validation | I2 | delta | 0.03 | 3.0251 | 0.0198 | 5 |
| inventory | FOODS_3_124_CA_3_validation | I2 | delta | 0.1 | 3.0264 | 0.0278 | 5 |
| inventory | FOODS_3_124_CA_3_validation | I2 | gamma | 0.9 | 2.9701 | 0.0265 | 5 |
| inventory | FOODS_3_124_CA_3_validation | I2 | gamma | 0.99 | 3.0264 | 0.0278 | 5 |
| inventory | FOODS_3_124_CA_3_validation | I2 | gamma | 1.0 | 3.0172 | 0.0233 | 5 |
| inventory | FOODS_3_124_CA_3_validation | I2 | horizon | 180 | 4.1453 | 0.0220 | 5 |
| inventory | FOODS_3_124_CA_3_validation | I2 | horizon | 30 | 2.3763 | 0.1325 | 5 |
| inventory | FOODS_3_124_CA_3_validation | I2 | horizon | 90 | 3.0264 | 0.0278 | 5 |
| inventory | FOODS_3_124_CA_3_validation | I2 | lam | 0.0 | 2.6919 | 0.1263 | 5 |
| inventory | FOODS_3_124_CA_3_validation | I2 | lam | 0.9 | 3.0264 | 0.0278 | 5 |
| inventory | FOODS_3_124_CA_3_validation | I2 | lam | 0.97 | 2.8393 | 0.0440 | 5 |
| inventory | FOODS_3_124_CA_3_validation | I2 | lam | 1.0 | 2.7147 | 0.1138 | 5 |
| inventory | FOODS_3_124_CA_3_validation | I2 | sigma0 | 0.25 | 3.0165 | 0.0399 | 5 |
| inventory | FOODS_3_124_CA_3_validation | I2 | sigma0 | 0.5 | 3.0264 | 0.0278 | 5 |
| inventory | FOODS_3_124_CA_3_validation | I2 | sigma0 | 1.0 | 3.0329 | 0.0417 | 5 |
| inventory | FOODS_3_124_CA_3_validation | I2 | std | False | 3.0263 | 0.0250 | 5 |
| inventory | FOODS_3_124_CA_3_validation | I2 | std | True | 3.0264 | 0.0278 | 5 |
| trading | TECHM | T1 | B | 128 | 0.9982 | 0.0203 | 5 |
| trading | TECHM | T1 | B | 16 | 0.9144 | 0.1039 | 5 |
| trading | TECHM | T1 | B | 32 | 0.9199 | 0.1521 | 5 |
| trading | TECHM | T1 | B | 64 | 1.0404 | 0.0806 | 5 |
| trading | TECHM | T1 | B | 8 | 0.8467 | 0.2391 | 5 |
| trading | TECHM | T1 | cg_iters | 10 | 0.9199 | 0.1521 | 5 |
| trading | TECHM | T1 | cg_iters | 2 | 0.9049 | 0.1469 | 5 |
| trading | TECHM | T1 | cg_iters | 20 | 0.9199 | 0.1521 | 5 |
| trading | TECHM | T1 | cg_iters | 5 | 0.8952 | 0.2332 | 5 |
| trading | TECHM | T1 | damp | 0.0001 | 0.7836 | 0.2182 | 5 |
| trading | TECHM | T1 | damp | 0.01 | 0.9199 | 0.1521 | 5 |
| trading | TECHM | T1 | damp | 0.1 | 0.9607 | 0.0919 | 5 |
| trading | TECHM | T1 | damp | 1.0 | 0.8908 | 0.2691 | 5 |
| trading | TECHM | T1 | delta | 0.003 | 0.8998 | 0.1667 | 5 |
| trading | TECHM | T1 | delta | 0.01 | 0.9199 | 0.1521 | 5 |
| trading | TECHM | T1 | delta | 0.03 | 0.9052 | 0.1801 | 5 |
| trading | TECHM | T1 | delta | 0.1 | 0.9853 | 0.0841 | 5 |
| trading | TECHM | T1 | gamma | 0.9 | 0.9447 | 0.1700 | 5 |
| trading | TECHM | T1 | gamma | 0.995 | 0.9199 | 0.1521 | 5 |
| trading | TECHM | T1 | gamma | 1.0 | 0.8009 | 0.1247 | 5 |
| trading | TECHM | T1 | horizon | 125 | 0.3559 | 0.0816 | 5 |
| trading | TECHM | T1 | horizon | 250 | 0.9199 | 0.1521 | 5 |
| trading | TECHM | T1 | horizon | 500 | 1.8993 | 0.1658 | 5 |
| trading | TECHM | T1 | lam | 0.0 | 0.6872 | 0.3289 | 5 |
| trading | TECHM | T1 | lam | 0.9 | 0.8444 | 0.1362 | 5 |
| trading | TECHM | T1 | lam | 0.97 | 0.9199 | 0.1521 | 5 |
| trading | TECHM | T1 | lam | 1.0 | 0.9663 | 0.1155 | 5 |
| trading | TECHM | T1 | std | False | 0.8184 | 0.1521 | 5 |
| trading | TECHM | T1 | std | True | 0.9199 | 0.1521 | 5 |
