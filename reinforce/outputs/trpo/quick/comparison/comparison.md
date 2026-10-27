# TRPO vs previous methods (quick mode)

Baselines (REINFORCE, random search, Nelder-Mead, SPSA, central FD) are **re-used** from `outputs/full` of this folder (same seeds, same theta0, 40,000-episode budget). TRPO counts pilot + training batches against the same budget; validation/diagnostic rollouts are excluded for all methods. Tuning effort differs: TRPO tunes delta x lambda (12 configs) and VPG lr (3); REINFORCE tuned lr x B x entropy (18).

Previous results unchanged during comparison (hash check of outputs/full): **True**

## Summary (validation return, deterministic policy, last checkpoint)

| key | policy | method | n | final | ci_lo | ci_hi | eps95_median | reached95 |
|---|---|---|---|---|---|---|---|---|
| FOODS_2_128_WI_2_validation | I1 | cm:central | 10 | 0.3643 | 0.3521 | 0.3797 | 0.0000 | 0.5000 |
| FOODS_2_128_WI_2_validation | I1 | cm:neldermead | 10 | 0.7101 | 0.7084 | 0.7114 | 2048.0000 | 0.9000 |
| FOODS_2_128_WI_2_validation | I1 | cm:random | 10 | 0.4635 | 0.4302 | 0.4931 | 1024.0000 | 0.7000 |
| FOODS_2_128_WI_2_validation | I1 | cm:spsa | 10 | 0.7158 | 0.7070 | 0.7240 | 2048.0000 | 1.0000 |
| FOODS_2_128_WI_2_validation | I1 | reinforce:loo_time | 10 | 0.7211 | 0.6962 | 0.7430 | 3472.0000 | 0.8000 |
| FOODS_2_128_WI_2_validation | I1 | reinforce:none | 10 | 0.7328 | 0.7112 | 0.7520 | 1984.0000 | 0.6000 |
| FOODS_2_128_WI_2_validation | I1 | trpo:ng_fixed | 3 | 0.5417 | 0.3151 | 0.6821 | 288.0000 | 0.6667 |
| FOODS_2_128_WI_2_validation | I1 | trpo:trpo | 3 | 0.4517 | 0.3266 | 0.5209 | 288.0000 | 0.6667 |
| FOODS_2_128_WI_2_validation | I1 | trpo:vpg | 3 | 0.5864 | 0.5556 | 0.6322 | 32.0000 | 0.3333 |
| FOODS_2_128_WI_2_validation | I2 | cm:central | 10 | 0.3360 | 0.3074 | 0.3583 | nan | 0.0000 |
| FOODS_2_128_WI_2_validation | I2 | cm:neldermead | 10 | 0.5382 | 0.4977 | 0.5722 | nan | 0.0000 |
| FOODS_2_128_WI_2_validation | I2 | cm:random | 10 | 0.2743 | 0.1850 | 0.3495 | nan | 0.0000 |
| FOODS_2_128_WI_2_validation | I2 | cm:spsa | 10 | 0.3092 | 0.2426 | 0.3795 | 1024.0000 | 0.1000 |
| FOODS_2_128_WI_2_validation | I2 | reinforce:loo_time | 10 | -5.0272 | -15.4479 | 0.3299 | 4960.0000 | 0.5000 |
| FOODS_2_128_WI_2_validation | I2 | reinforce:none | 10 | -20.2884 | -35.8511 | -4.7358 | 11408.0000 | 0.4000 |
| FOODS_2_128_WI_2_validation | I2 | trpo:ng_fixed | 3 | 0.7385 | 0.7111 | 0.7677 | 800.0000 | 0.3333 |
| FOODS_2_128_WI_2_validation | I2 | trpo:trpo | 3 | 0.7585 | 0.7403 | 0.7776 | 2080.0000 | 1.0000 |
| FOODS_2_128_WI_2_validation | I2 | trpo:vpg | 3 | 0.6897 | 0.6086 | 0.7490 | 1312.0000 | 0.3333 |
| FOODS_2_128_WI_2_validation | I3 | reinforce:loo_time | 10 | -0.6464 | -2.4845 | 0.5542 | 1440.0000 | 0.8000 |
| FOODS_2_128_WI_2_validation | I3 | reinforce:none | 10 | -0.3047 | -1.4524 | 0.5310 | 960.0000 | 0.8000 |
| FOODS_2_128_WI_2_validation | I3 | trpo:ng_fixed | 3 | 0.5942 | 0.5237 | 0.6351 | 2592.0000 | 1.0000 |
| FOODS_2_128_WI_2_validation | I3 | trpo:trpo | 3 | 0.5833 | 0.4811 | 0.6608 | 1824.0000 | 1.0000 |
| FOODS_2_128_WI_2_validation | I3 | trpo:vpg | 3 | -3.3659 | -8.6101 | 0.7597 | 288.0000 | 0.3333 |
| FOODS_3_124_CA_3_validation | I1 | cm:central | 10 | 2.4528 | 2.4381 | 2.4673 | 2560.0000 | 0.7000 |
| FOODS_3_124_CA_3_validation | I1 | cm:neldermead | 10 | 2.4167 | 2.3730 | 2.4604 | 0.0000 | 0.3000 |
| FOODS_3_124_CA_3_validation | I1 | cm:random | 10 | 2.4912 | 2.4557 | 2.5298 | 1024.0000 | 0.7000 |
| FOODS_3_124_CA_3_validation | I1 | cm:spsa | 10 | 2.6066 | 2.5937 | 2.6203 | 1024.0000 | 0.9000 |
| FOODS_3_124_CA_3_validation | I1 | reinforce:loo_time | 10 | 2.7925 | 2.7754 | 2.8070 | 5280.0000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I1 | reinforce:none | 10 | 2.8659 | 2.8544 | 2.8774 | 10560.0000 | 0.9000 |
| FOODS_3_124_CA_3_validation | I1 | trpo:ng_fixed | 3 | 2.8568 | 2.8229 | 2.8930 | 672.0000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I1 | trpo:trpo | 3 | 2.8123 | 2.7880 | 2.8284 | 672.0000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I1 | trpo:vpg | 3 | 2.8774 | 2.8605 | 2.8876 | 1184.0000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I2 | cm:central | 10 | 2.7003 | 2.6697 | 2.7306 | nan | 0.0000 |
| FOODS_3_124_CA_3_validation | I2 | cm:neldermead | 10 | 2.5586 | 2.4924 | 2.6107 | nan | 0.0000 |
| FOODS_3_124_CA_3_validation | I2 | cm:random | 10 | 2.7123 | 2.6440 | 2.7728 | 24576.0000 | 0.4000 |
| FOODS_3_124_CA_3_validation | I2 | cm:spsa | 10 | 2.5166 | 2.4424 | 2.5780 | 1024.0000 | 0.3000 |
| FOODS_3_124_CA_3_validation | I2 | reinforce:loo_time | 10 | 2.5529 | 2.4610 | 2.6395 | 4960.0000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I2 | reinforce:none | 10 | -7.4836 | -28.0775 | 2.8439 | 9920.0000 | 0.6000 |
| FOODS_3_124_CA_3_validation | I2 | trpo:ng_fixed | 3 | 2.9598 | 2.9010 | 3.0294 | 1696.0000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I2 | trpo:trpo | 3 | 2.9633 | 2.9296 | 3.0212 | 1568.0000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I2 | trpo:vpg | 3 | 2.8885 | 2.8452 | 2.9243 | 1824.0000 | 0.6667 |
| FOODS_3_124_CA_3_validation | I3 | reinforce:loo_time | 10 | 2.8782 | 2.8418 | 2.9101 | 960.0000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I3 | reinforce:none | 10 | 2.8750 | 2.8217 | 2.9263 | 1440.0000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I3 | trpo:ng_fixed | 3 | 2.4489 | 2.3632 | 2.5724 | 416.0000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I3 | trpo:trpo | 3 | 2.5847 | 2.3033 | 2.7711 | 416.0000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I3 | trpo:vpg | 3 | 2.9327 | 2.9228 | 2.9401 | 288.0000 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I1 | cm:central | 10 | 2.9776 | 2.9547 | 2.9970 | 11520.0000 | 0.4000 |
| HOBBIES_1_371_WI_1_validation | I1 | cm:neldermead | 10 | 2.0465 | 1.4086 | 2.6291 | nan | 0.0000 |
| HOBBIES_1_371_WI_1_validation | I1 | cm:random | 10 | 3.0163 | 2.9964 | 3.0356 | 3072.0000 | 0.3000 |
| HOBBIES_1_371_WI_1_validation | I1 | cm:spsa | 10 | 3.0089 | 2.9872 | 3.0296 | nan | 0.0000 |
| HOBBIES_1_371_WI_1_validation | I1 | reinforce:loo_time | 10 | 3.1526 | 3.1014 | 3.2062 | 6448.0000 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I1 | reinforce:none | 10 | 3.0553 | 3.0096 | 3.1168 | 6944.0000 | 0.8000 |
| HOBBIES_1_371_WI_1_validation | I1 | trpo:ng_fixed | 3 | 3.3233 | 3.2951 | 3.3643 | 1440.0000 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I1 | trpo:trpo | 3 | 3.2484 | 3.2272 | 3.2823 | 1696.0000 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I1 | trpo:vpg | 3 | 3.1792 | 3.0824 | 3.3070 | 928.0000 | 0.3333 |
| HOBBIES_1_371_WI_1_validation | I2 | cm:central | 10 | 3.0212 | 2.9887 | 3.0530 | nan | 0.0000 |
| HOBBIES_1_371_WI_1_validation | I2 | cm:neldermead | 10 | 2.6212 | 2.5754 | 2.6573 | nan | 0.0000 |
| HOBBIES_1_371_WI_1_validation | I2 | cm:random | 10 | 2.9979 | 2.9296 | 3.0592 | 9216.0000 | 0.1000 |
| HOBBIES_1_371_WI_1_validation | I2 | cm:spsa | 10 | 2.8846 | 2.8409 | 2.9259 | 2048.0000 | 0.1000 |
| HOBBIES_1_371_WI_1_validation | I2 | reinforce:loo_time | 10 | -19.6598 | -42.2903 | 2.7947 | 1984.0000 | 0.5000 |
| HOBBIES_1_371_WI_1_validation | I2 | reinforce:none | 10 | -19.6087 | -42.1948 | -4.3026 | 4960.0000 | 0.6000 |
| HOBBIES_1_371_WI_1_validation | I2 | trpo:ng_fixed | 3 | 3.1201 | 2.7784 | 3.2953 | 1696.0000 | 0.6667 |
| HOBBIES_1_371_WI_1_validation | I2 | trpo:trpo | 3 | 3.2374 | 3.1873 | 3.2887 | 1568.0000 | 0.6667 |
| HOBBIES_1_371_WI_1_validation | I2 | trpo:vpg | 3 | 3.2920 | 3.2369 | 3.3371 | 800.0000 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I3 | reinforce:loo_time | 10 | 2.9870 | 2.8652 | 3.0694 | 992.0000 | 0.9000 |
| HOBBIES_1_371_WI_1_validation | I3 | reinforce:none | 10 | 2.9728 | 2.8358 | 3.1042 | 1984.0000 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I3 | trpo:ng_fixed | 3 | 3.2446 | 3.1766 | 3.3086 | 928.0000 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I3 | trpo:trpo | 3 | 3.2545 | 3.1934 | 3.3356 | 928.0000 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I3 | trpo:vpg | 3 | 1.8345 | -0.5984 | 3.2953 | 800.0000 | 1.0000 |
| TECHM | T1 | cm:central | 10 | 0.7374 | 0.6803 | 0.7881 | 13440.0000 | 0.2000 |
| TECHM | T1 | cm:neldermead | 10 | 0.6857 | 0.6242 | 0.7456 | 2016.0000 | 0.3000 |
| TECHM | T1 | cm:random | 10 | 0.6216 | 0.5520 | 0.6857 | 4096.0000 | 0.5000 |
| TECHM | T1 | cm:spsa | 10 | 0.6731 | 0.6217 | 0.7165 | nan | 0.0000 |
| TECHM | T1 | reinforce:loo_time | 10 | 0.8051 | 0.7260 | 0.8871 | 4960.0000 | 0.7000 |
| TECHM | T1 | reinforce:none | 10 | 0.7818 | 0.7121 | 0.8604 | 4960.0000 | 0.8000 |
| TECHM | T1 | trpo:ng_fixed | 3 | 0.8892 | 0.7169 | 1.0002 | 2656.0000 | 0.6667 |
| TECHM | T1 | trpo:trpo | 3 | 0.6100 | 0.4107 | 0.8502 | 288.0000 | 0.3333 |
| TECHM | T1 | trpo:vpg | 3 | 0.7868 | 0.5511 | 1.0180 | 800.0000 | 0.3333 |
| TECHM | T2 | reinforce:loo_time | 10 | 0.7853 | 0.7424 | 0.8246 | 5760.0000 | 1.0000 |
| TECHM | T2 | reinforce:none | 10 | 0.7701 | 0.7262 | 0.8161 | 4800.0000 | 1.0000 |
| TECHM | T2 | trpo:ng_fixed | 3 | 0.6270 | 0.2609 | 0.9205 | 1312.0000 | 1.0000 |
| TECHM | T2 | trpo:trpo | 3 | 0.6225 | 0.4232 | 0.9843 | 992.0000 | 0.6667 |
| TECHM | T2 | trpo:vpg | 3 | 0.4981 | 0.3374 | 0.7145 | nan | 0.0000 |

## TRPO vs each method (paired by seed, Wilcoxon, Holm over all rows)

| key | policy | vs | n_pairs | diff_trpo_minus | p_wilcoxon_paired | p_mannwhitney | cliffs_delta | p_holm | verdict |
|---|---|---|---|---|---|---|---|---|---|
| FOODS_2_128_WI_2_validation | I1 | cm:central | 3 | 0.0796 | 0.5000 | 0.7000 | 0.3333 | 1.0000 | no significant difference |
| FOODS_2_128_WI_2_validation | I1 | cm:neldermead | 3 | -0.2587 | 0.2500 | 0.1000 | -1.0000 | 1.0000 | no significant difference |
| FOODS_2_128_WI_2_validation | I1 | cm:random | 3 | -0.0143 | 1.0000 | 0.7000 | 0.3333 | 1.0000 | no significant difference |
| FOODS_2_128_WI_2_validation | I1 | cm:spsa | 3 | -0.2593 | 0.2500 | 0.1000 | -1.0000 | 1.0000 | no significant difference |
| FOODS_2_128_WI_2_validation | I1 | reinforce:loo_time | 3 | -0.2634 | 0.2500 | 0.1000 | -1.0000 | 1.0000 | no significant difference |
| FOODS_2_128_WI_2_validation | I1 | reinforce:none | 3 | -0.2788 | 0.2500 | 0.1000 | -1.0000 | 1.0000 | no significant difference |
| FOODS_2_128_WI_2_validation | I1 | trpo:ng_fixed | 3 | -0.0900 | 0.5000 | 0.7000 | -0.3333 | 1.0000 | no significant difference |
| FOODS_2_128_WI_2_validation | I1 | trpo:vpg | 3 | -0.1347 | 0.2500 | 0.1000 | -1.0000 | 1.0000 | no significant difference |
| FOODS_2_128_WI_2_validation | I2 | cm:central | 3 | 0.4131 | 0.2500 | 0.1000 | 1.0000 | 1.0000 | no significant difference |
| FOODS_2_128_WI_2_validation | I2 | cm:neldermead | 3 | 0.2669 | 0.2500 | 0.1000 | 1.0000 | 1.0000 | no significant difference |
| FOODS_2_128_WI_2_validation | I2 | cm:random | 3 | 0.4659 | 0.2500 | 0.1000 | 1.0000 | 1.0000 | no significant difference |
| FOODS_2_128_WI_2_validation | I2 | cm:spsa | 3 | 0.5280 | 0.2500 | 0.1000 | 1.0000 | 1.0000 | no significant difference |
| FOODS_2_128_WI_2_validation | I2 | reinforce:loo_time | 3 | 0.8685 | 0.2500 | 0.1000 | 1.0000 | 1.0000 | no significant difference |
| FOODS_2_128_WI_2_validation | I2 | reinforce:none | 3 | 34.9315 | 0.2500 | 0.1000 | 1.0000 | 1.0000 | no significant difference |
| FOODS_2_128_WI_2_validation | I2 | trpo:ng_fixed | 3 | 0.0200 | 0.7500 | 0.4000 | 0.5556 | 1.0000 | no significant difference |
| FOODS_2_128_WI_2_validation | I2 | trpo:vpg | 3 | 0.0688 | 0.5000 | 0.2000 | 0.7778 | 1.0000 | no significant difference |
| FOODS_2_128_WI_2_validation | I3 | reinforce:loo_time | 3 | 0.9439 | 0.7500 | 0.4000 | 0.5556 | 1.0000 | no significant difference |
| FOODS_2_128_WI_2_validation | I3 | reinforce:none | 3 | 0.8501 | 1.0000 | 0.7000 | 0.3333 | 1.0000 | no significant difference |
| FOODS_2_128_WI_2_validation | I3 | trpo:ng_fixed | 3 | -0.0109 | 0.7500 | 1.0000 | -0.1111 | 1.0000 | no significant difference |
| FOODS_2_128_WI_2_validation | I3 | trpo:vpg | 3 | 3.9493 | 0.5000 | 0.7000 | 0.3333 | 1.0000 | no significant difference |
| FOODS_3_124_CA_3_validation | I1 | cm:central | 3 | 0.3849 | 0.2500 | 0.1000 | 1.0000 | 1.0000 | no significant difference |
| FOODS_3_124_CA_3_validation | I1 | cm:neldermead | 3 | 0.3663 | 0.2500 | 0.1000 | 1.0000 | 1.0000 | no significant difference |
| FOODS_3_124_CA_3_validation | I1 | cm:random | 3 | 0.3796 | 0.2500 | 0.1000 | 1.0000 | 1.0000 | no significant difference |
| FOODS_3_124_CA_3_validation | I1 | cm:spsa | 3 | 0.2068 | 0.2500 | 0.1000 | 1.0000 | 1.0000 | no significant difference |
| FOODS_3_124_CA_3_validation | I1 | reinforce:loo_time | 3 | -0.0015 | 0.7500 | 1.0000 | -0.1111 | 1.0000 | no significant difference |
| FOODS_3_124_CA_3_validation | I1 | reinforce:none | 3 | -0.0500 | 0.2500 | 0.1000 | -1.0000 | 1.0000 | no significant difference |
| FOODS_3_124_CA_3_validation | I1 | trpo:ng_fixed | 3 | -0.0446 | 0.2500 | 0.2000 | -0.7778 | 1.0000 | no significant difference |
| FOODS_3_124_CA_3_validation | I1 | trpo:vpg | 3 | -0.0651 | 0.2500 | 0.1000 | -1.0000 | 1.0000 | no significant difference |
| FOODS_3_124_CA_3_validation | I2 | cm:central | 3 | 0.2473 | 0.2500 | 0.1000 | 1.0000 | 1.0000 | no significant difference |
| FOODS_3_124_CA_3_validation | I2 | cm:neldermead | 3 | 0.4291 | 0.2500 | 0.1000 | 1.0000 | 1.0000 | no significant difference |
| FOODS_3_124_CA_3_validation | I2 | cm:random | 3 | 0.2633 | 0.2500 | 0.1000 | 1.0000 | 1.0000 | no significant difference |
| FOODS_3_124_CA_3_validation | I2 | cm:spsa | 3 | 0.3494 | 0.2500 | 0.1000 | 1.0000 | 1.0000 | no significant difference |
| FOODS_3_124_CA_3_validation | I2 | reinforce:loo_time | 3 | 0.3954 | 0.2500 | 0.1000 | 1.0000 | 1.0000 | no significant difference |
| FOODS_3_124_CA_3_validation | I2 | reinforce:none | 3 | 0.2261 | 0.2500 | 0.1000 | 1.0000 | 1.0000 | no significant difference |
| FOODS_3_124_CA_3_validation | I2 | trpo:ng_fixed | 3 | 0.0035 | 1.0000 | 1.0000 | -0.1111 | 1.0000 | no significant difference |
| FOODS_3_124_CA_3_validation | I2 | trpo:vpg | 3 | 0.0748 | 0.2500 | 0.1000 | 1.0000 | 1.0000 | no significant difference |
| FOODS_3_124_CA_3_validation | I3 | reinforce:loo_time | 3 | -0.3296 | 0.2500 | 0.1000 | -1.0000 | 1.0000 | no significant difference |
| FOODS_3_124_CA_3_validation | I3 | reinforce:none | 3 | -0.2925 | 0.2500 | 0.1000 | -1.0000 | 1.0000 | no significant difference |
| FOODS_3_124_CA_3_validation | I3 | trpo:ng_fixed | 3 | 0.1358 | 0.5000 | 0.7000 | 0.3333 | 1.0000 | no significant difference |
| FOODS_3_124_CA_3_validation | I3 | trpo:vpg | 3 | -0.3480 | 0.2500 | 0.1000 | -1.0000 | 1.0000 | no significant difference |
| HOBBIES_1_371_WI_1_validation | I1 | cm:central | 3 | 0.2840 | 0.2500 | 0.1000 | 1.0000 | 1.0000 | no significant difference |
| HOBBIES_1_371_WI_1_validation | I1 | cm:neldermead | 3 | 0.6144 | 0.2500 | 0.1000 | 1.0000 | 1.0000 | no significant difference |
| HOBBIES_1_371_WI_1_validation | I1 | cm:random | 3 | 0.2408 | 0.2500 | 0.1000 | 1.0000 | 1.0000 | no significant difference |
| HOBBIES_1_371_WI_1_validation | I1 | cm:spsa | 3 | 0.2359 | 0.2500 | 0.1000 | 1.0000 | 1.0000 | no significant difference |
| HOBBIES_1_371_WI_1_validation | I1 | reinforce:loo_time | 3 | 0.1704 | 0.2500 | 0.1000 | 1.0000 | 1.0000 | no significant difference |
| HOBBIES_1_371_WI_1_validation | I1 | reinforce:none | 3 | 0.1268 | 0.5000 | 0.7000 | 0.3333 | 1.0000 | no significant difference |
| HOBBIES_1_371_WI_1_validation | I1 | trpo:ng_fixed | 3 | -0.0749 | 0.2500 | 0.1000 | -1.0000 | 1.0000 | no significant difference |
| HOBBIES_1_371_WI_1_validation | I1 | trpo:vpg | 3 | 0.0692 | 0.5000 | 0.7000 | 0.3333 | 1.0000 | no significant difference |
| HOBBIES_1_371_WI_1_validation | I2 | cm:central | 3 | 0.2389 | 0.2500 | 0.1000 | 1.0000 | 1.0000 | no significant difference |
| HOBBIES_1_371_WI_1_validation | I2 | cm:neldermead | 3 | 0.5870 | 0.2500 | 0.1000 | 1.0000 | 1.0000 | no significant difference |
| HOBBIES_1_371_WI_1_validation | I2 | cm:random | 3 | 0.2490 | 0.2500 | 0.1000 | 1.0000 | 1.0000 | no significant difference |
| HOBBIES_1_371_WI_1_validation | I2 | cm:spsa | 3 | 0.3098 | 0.2500 | 0.1000 | 1.0000 | 1.0000 | no significant difference |
| HOBBIES_1_371_WI_1_validation | I2 | reinforce:loo_time | 3 | 25.3851 | 0.2500 | 0.1000 | 1.0000 | 1.0000 | no significant difference |
| HOBBIES_1_371_WI_1_validation | I2 | reinforce:none | 3 | 50.1956 | 0.5000 | 0.2000 | 0.7778 | 1.0000 | no significant difference |
| HOBBIES_1_371_WI_1_validation | I2 | trpo:ng_fixed | 3 | 0.1174 | 0.7500 | 1.0000 | -0.1111 | 1.0000 | no significant difference |
| HOBBIES_1_371_WI_1_validation | I2 | trpo:vpg | 3 | -0.0546 | 0.2500 | 0.2000 | -0.7778 | 1.0000 | no significant difference |
| HOBBIES_1_371_WI_1_validation | I3 | reinforce:loo_time | 3 | 0.4226 | 0.2500 | 0.1000 | 1.0000 | 1.0000 | no significant difference |
| HOBBIES_1_371_WI_1_validation | I3 | reinforce:none | 3 | 0.1394 | 0.2500 | 0.1000 | 1.0000 | 1.0000 | no significant difference |
| HOBBIES_1_371_WI_1_validation | I3 | trpo:ng_fixed | 3 | 0.0099 | 0.5000 | 1.0000 | 0.1111 | 1.0000 | no significant difference |
| HOBBIES_1_371_WI_1_validation | I3 | trpo:vpg | 3 | 1.4200 | 0.5000 | 0.4000 | 0.5556 | 1.0000 | no significant difference |
| TECHM | T1 | cm:central | 3 | -0.1058 | 0.5000 | 0.7000 | -0.3333 | 1.0000 | no significant difference |
| TECHM | T1 | cm:neldermead | 3 | -0.0660 | 0.2500 | 0.7000 | -0.3333 | 1.0000 | no significant difference |
| TECHM | T1 | cm:random | 3 | 0.0373 | 0.5000 | 1.0000 | 0.1111 | 1.0000 | no significant difference |
| TECHM | T1 | cm:spsa | 3 | -0.0060 | 1.0000 | 1.0000 | -0.1111 | 1.0000 | no significant difference |
| TECHM | T1 | reinforce:loo_time | 3 | -0.3058 | 0.2500 | 0.1000 | -1.0000 | 1.0000 | no significant difference |
| TECHM | T1 | reinforce:none | 3 | -0.2892 | 0.2500 | 0.2000 | -0.7778 | 1.0000 | no significant difference |
| TECHM | T1 | trpo:ng_fixed | 3 | -0.2792 | 0.2500 | 0.2000 | -0.7778 | 1.0000 | no significant difference |
| TECHM | T1 | trpo:vpg | 3 | -0.1768 | 0.5000 | 0.7000 | -0.3333 | 1.0000 | no significant difference |
| TECHM | T2 | reinforce:loo_time | 3 | -0.2013 | 0.5000 | 0.7000 | -0.3333 | 1.0000 | no significant difference |
| TECHM | T2 | reinforce:none | 3 | -0.1655 | 0.5000 | 0.7000 | -0.3333 | 1.0000 | no significant difference |
| TECHM | T2 | trpo:ng_fixed | 3 | -0.0045 | 1.0000 | 1.0000 | 0.1111 | 1.0000 | no significant difference |
| TECHM | T2 | trpo:vpg | 3 | 0.1244 | 0.2500 | 0.7000 | 0.3333 | 1.0000 | no significant difference |

## Verdict counts

| vs | no significant difference |
|---|---|
| cm:central | 7 |
| cm:neldermead | 7 |
| cm:random | 7 |
| cm:spsa | 7 |
| reinforce:loo_time | 11 |
| reinforce:none | 11 |
| trpo:ng_fixed | 11 |
| trpo:vpg | 11 |