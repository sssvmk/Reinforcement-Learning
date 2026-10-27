# TRPO vs previous methods (full mode)

Baselines (REINFORCE, random search, Nelder-Mead, SPSA, central FD) are **re-used** from `outputs/full` of this folder (same seeds, same theta0, 40,000-episode budget). TRPO counts pilot + training batches against the same budget; validation/diagnostic rollouts are excluded for all methods. Tuning effort differs: TRPO tunes delta x lambda (12 configs) and VPG lr (3); REINFORCE tuned lr x B x entropy (18).

Previous results unchanged during comparison (hash check of outputs/full): **True**

## Summary (validation return, deterministic policy, last checkpoint)

| key | policy | method | n | final | ci_lo | ci_hi | eps95_median | reached95 |
|---|---|---|---|---|---|---|---|---|
| FOODS_2_128_WI_2_validation | I1 | cm:central | 10 | 0.3643 | 0.3521 | 0.3797 | 1280.0000 | 0.1000 |
| FOODS_2_128_WI_2_validation | I1 | cm:neldermead | 10 | 0.7101 | 0.7084 | 0.7114 | 2048.0000 | 0.9000 |
| FOODS_2_128_WI_2_validation | I1 | cm:random | 10 | 0.4635 | 0.4302 | 0.4931 | 2048.0000 | 0.6000 |
| FOODS_2_128_WI_2_validation | I1 | cm:spsa | 10 | 0.7158 | 0.7070 | 0.7240 | 4096.0000 | 0.9000 |
| FOODS_2_128_WI_2_validation | I1 | reinforce:loo_time | 10 | 0.7211 | 0.6962 | 0.7430 | 5952.0000 | 0.5000 |
| FOODS_2_128_WI_2_validation | I1 | reinforce:none | 10 | 0.7328 | 0.7112 | 0.7520 | 2976.0000 | 0.3000 |
| FOODS_2_128_WI_2_validation | I1 | trpo:ng_fixed | 10 | 0.7634 | 0.7535 | 0.7729 | 4000.0000 | 1.0000 |
| FOODS_2_128_WI_2_validation | I1 | trpo:trpo | 10 | 0.6775 | 0.6594 | 0.6971 | 3504.0000 | 1.0000 |
| FOODS_2_128_WI_2_validation | I1 | trpo:vpg | 10 | 0.7471 | 0.7339 | 0.7593 | 16400.0000 | 0.6000 |
| FOODS_2_128_WI_2_validation | I2 | cm:central | 10 | 0.3360 | 0.3074 | 0.3583 | nan | 0.0000 |
| FOODS_2_128_WI_2_validation | I2 | cm:neldermead | 10 | 0.5382 | 0.4977 | 0.5722 | nan | 0.0000 |
| FOODS_2_128_WI_2_validation | I2 | cm:random | 10 | 0.2743 | 0.1850 | 0.3495 | nan | 0.0000 |
| FOODS_2_128_WI_2_validation | I2 | cm:spsa | 10 | 0.3092 | 0.2426 | 0.3795 | 1024.0000 | 0.1000 |
| FOODS_2_128_WI_2_validation | I2 | reinforce:loo_time | 10 | -5.0272 | -15.4479 | 0.3299 | 4960.0000 | 0.7000 |
| FOODS_2_128_WI_2_validation | I2 | reinforce:none | 10 | -20.2884 | -35.8511 | -4.7358 | 11408.0000 | 0.4000 |
| FOODS_2_128_WI_2_validation | I2 | trpo:ng_fixed | 10 | 0.7411 | 0.7163 | 0.7659 | 16400.0000 | 0.8000 |
| FOODS_2_128_WI_2_validation | I2 | trpo:trpo | 10 | 0.7544 | 0.7239 | 0.7803 | 10448.0000 | 1.0000 |
| FOODS_2_128_WI_2_validation | I2 | trpo:vpg | 10 | 0.6383 | 0.5739 | 0.6901 | 2016.0000 | 0.6000 |
| FOODS_2_128_WI_2_validation | I3 | reinforce:loo_time | 10 | -0.6464 | -2.4845 | 0.5542 | 1440.0000 | 0.8000 |
| FOODS_2_128_WI_2_validation | I3 | reinforce:none | 10 | -0.3047 | -1.4524 | 0.5310 | 960.0000 | 0.8000 |
| FOODS_2_128_WI_2_validation | I3 | trpo:ng_fixed | 10 | 0.5059 | 0.1046 | 0.7311 | 1024.0000 | 1.0000 |
| FOODS_2_128_WI_2_validation | I3 | trpo:trpo | 10 | 0.7031 | 0.6648 | 0.7426 | 1024.0000 | 1.0000 |
| FOODS_2_128_WI_2_validation | I3 | trpo:vpg | 10 | 0.7732 | 0.7472 | 0.7969 | 1024.0000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I1 | cm:central | 10 | 2.4528 | 2.4381 | 2.4673 | 2560.0000 | 0.7000 |
| FOODS_3_124_CA_3_validation | I1 | cm:neldermead | 10 | 2.4167 | 2.3730 | 2.4604 | 0.0000 | 0.3000 |
| FOODS_3_124_CA_3_validation | I1 | cm:random | 10 | 2.4912 | 2.4557 | 2.5298 | 1024.0000 | 0.8000 |
| FOODS_3_124_CA_3_validation | I1 | cm:spsa | 10 | 2.6066 | 2.5937 | 2.6203 | 1024.0000 | 0.9000 |
| FOODS_3_124_CA_3_validation | I1 | reinforce:loo_time | 10 | 2.7925 | 2.7754 | 2.8070 | 5280.0000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I1 | reinforce:none | 10 | 2.8659 | 2.8544 | 2.8774 | 8640.0000 | 0.9000 |
| FOODS_3_124_CA_3_validation | I1 | trpo:ng_fixed | 10 | 2.8421 | 2.8310 | 2.8526 | 1520.0000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I1 | trpo:trpo | 10 | 2.8145 | 2.8031 | 2.8266 | 2016.0000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I1 | trpo:vpg | 10 | 2.8618 | 2.8436 | 2.8785 | 1024.0000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I2 | cm:central | 10 | 2.7003 | 2.6697 | 2.7306 | nan | 0.0000 |
| FOODS_3_124_CA_3_validation | I2 | cm:neldermead | 10 | 2.5586 | 2.4924 | 2.6107 | nan | 0.0000 |
| FOODS_3_124_CA_3_validation | I2 | cm:random | 10 | 2.7123 | 2.6440 | 2.7728 | 30720.0000 | 0.3000 |
| FOODS_3_124_CA_3_validation | I2 | cm:spsa | 10 | 2.5166 | 2.4424 | 2.5780 | 1024.0000 | 0.3000 |
| FOODS_3_124_CA_3_validation | I2 | reinforce:loo_time | 10 | 2.5529 | 2.4610 | 2.6395 | 3968.0000 | 0.7000 |
| FOODS_3_124_CA_3_validation | I2 | reinforce:none | 10 | -7.4836 | -28.0775 | 2.8439 | 13392.0000 | 0.4000 |
| FOODS_3_124_CA_3_validation | I2 | trpo:ng_fixed | 10 | 2.9944 | 2.9671 | 3.0226 | 1024.0000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I2 | trpo:trpo | 10 | 2.8979 | 2.8608 | 2.9349 | 2016.0000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I2 | trpo:vpg | 10 | 2.8780 | 2.8157 | 2.9408 | 3504.0000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I3 | reinforce:loo_time | 10 | 2.8782 | 2.8418 | 2.9101 | 960.0000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I3 | reinforce:none | 10 | 2.8750 | 2.8217 | 2.9263 | 2400.0000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I3 | trpo:ng_fixed | 10 | 3.0354 | 3.0054 | 3.0630 | 1024.0000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I3 | trpo:trpo | 10 | 3.0481 | 3.0224 | 3.0739 | 1024.0000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I3 | trpo:vpg | 10 | 3.0170 | 3.0035 | 3.0335 | 1024.0000 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I1 | cm:central | 10 | 2.9776 | 2.9547 | 2.9970 | 10880.0000 | 0.4000 |
| HOBBIES_1_371_WI_1_validation | I1 | cm:neldermead | 10 | 2.0465 | 1.4086 | 2.6291 | nan | 0.0000 |
| HOBBIES_1_371_WI_1_validation | I1 | cm:random | 10 | 3.0163 | 2.9964 | 3.0356 | 3072.0000 | 0.3000 |
| HOBBIES_1_371_WI_1_validation | I1 | cm:spsa | 10 | 3.0089 | 2.9872 | 3.0296 | nan | 0.0000 |
| HOBBIES_1_371_WI_1_validation | I1 | reinforce:loo_time | 10 | 3.1526 | 3.1014 | 3.2062 | 5456.0000 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I1 | reinforce:none | 10 | 3.0553 | 3.0096 | 3.1168 | 5456.0000 | 0.8000 |
| HOBBIES_1_371_WI_1_validation | I1 | trpo:ng_fixed | 10 | 3.3122 | 3.2976 | 3.3245 | 3008.0000 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I1 | trpo:trpo | 10 | 3.2546 | 3.2258 | 3.2831 | 3008.0000 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I1 | trpo:vpg | 10 | 3.1010 | 3.0300 | 3.1759 | 2016.0000 | 0.9000 |
| HOBBIES_1_371_WI_1_validation | I2 | cm:central | 10 | 3.0212 | 2.9887 | 3.0530 | nan | 0.0000 |
| HOBBIES_1_371_WI_1_validation | I2 | cm:neldermead | 10 | 2.6212 | 2.5754 | 2.6573 | nan | 0.0000 |
| HOBBIES_1_371_WI_1_validation | I2 | cm:random | 10 | 2.9979 | 2.9296 | 3.0592 | 13312.0000 | 0.2000 |
| HOBBIES_1_371_WI_1_validation | I2 | cm:spsa | 10 | 2.8846 | 2.8409 | 2.9259 | 1536.0000 | 0.2000 |
| HOBBIES_1_371_WI_1_validation | I2 | reinforce:loo_time | 10 | -19.6598 | -42.2903 | 2.7947 | 2480.0000 | 0.6000 |
| HOBBIES_1_371_WI_1_validation | I2 | reinforce:none | 10 | -19.6087 | -42.1948 | -4.3026 | 4960.0000 | 0.6000 |
| HOBBIES_1_371_WI_1_validation | I2 | trpo:ng_fixed | 10 | 3.1931 | 3.1145 | 3.2531 | 2016.0000 | 0.6000 |
| HOBBIES_1_371_WI_1_validation | I2 | trpo:trpo | 10 | 3.1495 | 3.0733 | 3.2166 | 4496.0000 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I2 | trpo:vpg | 10 | 3.2758 | 3.2340 | 3.3164 | 5984.0000 | 0.9000 |
| HOBBIES_1_371_WI_1_validation | I3 | reinforce:loo_time | 10 | 2.9870 | 2.8652 | 3.0694 | 992.0000 | 0.9000 |
| HOBBIES_1_371_WI_1_validation | I3 | reinforce:none | 10 | 2.9728 | 2.8358 | 3.1042 | 1984.0000 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I3 | trpo:ng_fixed | 10 | 3.2675 | 3.2477 | 3.2853 | 1024.0000 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I3 | trpo:trpo | 10 | 3.2290 | 3.1982 | 3.2562 | 1024.0000 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I3 | trpo:vpg | 10 | 3.2335 | 3.0809 | 3.3514 | 1024.0000 | 1.0000 |
| TECHM | T1 | cm:central | 10 | 0.7374 | 0.6803 | 0.7881 | 3920.0000 | 0.8000 |
| TECHM | T1 | cm:neldermead | 10 | 0.6857 | 0.6242 | 0.7456 | 1008.0000 | 0.4000 |
| TECHM | T1 | cm:random | 10 | 0.6216 | 0.5520 | 0.6857 | 4096.0000 | 0.9000 |
| TECHM | T1 | cm:spsa | 10 | 0.6731 | 0.6217 | 0.7165 | nan | 0.0000 |
| TECHM | T1 | reinforce:loo_time | 10 | 0.8051 | 0.7260 | 0.8871 | 2976.0000 | 0.9000 |
| TECHM | T1 | reinforce:none | 10 | 0.7818 | 0.7121 | 0.8604 | 5952.0000 | 1.0000 |
| TECHM | T1 | trpo:ng_fixed | 10 | 0.7791 | 0.6010 | 0.9213 | 1024.0000 | 0.8000 |
| TECHM | T1 | trpo:trpo | 10 | 0.7694 | 0.6133 | 0.8935 | 1520.0000 | 0.8000 |
| TECHM | T1 | trpo:vpg | 10 | 0.6663 | 0.5076 | 0.8129 | 2016.0000 | 0.7000 |
| TECHM | T2 | reinforce:loo_time | 10 | 0.7853 | 0.7424 | 0.8246 | 5760.0000 | 1.0000 |
| TECHM | T2 | reinforce:none | 10 | 0.7701 | 0.7262 | 0.8161 | 4800.0000 | 1.0000 |
| TECHM | T2 | trpo:ng_fixed | 10 | 0.5304 | 0.4245 | 0.6123 | 2016.0000 | 1.0000 |
| TECHM | T2 | trpo:trpo | 10 | 0.6804 | 0.5689 | 0.7776 | 3008.0000 | 0.7000 |
| TECHM | T2 | trpo:vpg | 10 | 0.6123 | 0.4791 | 0.7408 | 8960.0000 | 0.6000 |

## TRPO vs each method (paired by seed, Wilcoxon, Holm over all rows)

| key | policy | vs | n_pairs | diff_trpo_minus | p_wilcoxon_paired | p_mannwhitney | cliffs_delta | p_holm | verdict |
|---|---|---|---|---|---|---|---|---|---|
| FOODS_2_128_WI_2_validation | I1 | cm:central | 10 | 0.3132 | 0.0020 | 0.0002 | 1.0000 | 0.1406 | no significant difference |
| FOODS_2_128_WI_2_validation | I1 | cm:neldermead | 10 | -0.0326 | 0.0195 | 0.0257 | -0.6000 | 0.5859 | no significant difference |
| FOODS_2_128_WI_2_validation | I1 | cm:random | 10 | 0.2140 | 0.0020 | 0.0002 | 1.0000 | 0.1406 | no significant difference |
| FOODS_2_128_WI_2_validation | I1 | cm:spsa | 10 | -0.0383 | 0.0039 | 0.0113 | -0.6800 | 0.1562 | no significant difference |
| FOODS_2_128_WI_2_validation | I1 | reinforce:loo_time | 10 | -0.0435 | 0.0488 | 0.0140 | -0.6600 | 1.0000 | no significant difference |
| FOODS_2_128_WI_2_validation | I1 | reinforce:none | 10 | -0.0553 | 0.0098 | 0.0058 | -0.7400 | 0.3418 | no significant difference |
| FOODS_2_128_WI_2_validation | I1 | trpo:ng_fixed | 10 | -0.0859 | 0.0020 | 0.0002 | -1.0000 | 0.1406 | no significant difference |
| FOODS_2_128_WI_2_validation | I1 | trpo:vpg | 10 | -0.0696 | 0.0020 | 0.0004 | -0.9400 | 0.1406 | no significant difference |
| FOODS_2_128_WI_2_validation | I2 | cm:central | 10 | 0.4183 | 0.0020 | 0.0002 | 1.0000 | 0.1406 | no significant difference |
| FOODS_2_128_WI_2_validation | I2 | cm:neldermead | 10 | 0.2162 | 0.0020 | 0.0002 | 1.0000 | 0.1406 | no significant difference |
| FOODS_2_128_WI_2_validation | I2 | cm:random | 10 | 0.4801 | 0.0020 | 0.0002 | 1.0000 | 0.1406 | no significant difference |
| FOODS_2_128_WI_2_validation | I2 | cm:spsa | 10 | 0.4452 | 0.0020 | 0.0002 | 1.0000 | 0.1406 | no significant difference |
| FOODS_2_128_WI_2_validation | I2 | reinforce:loo_time | 10 | 5.7816 | 0.0020 | 0.0002 | 0.9800 | 0.1406 | no significant difference |
| FOODS_2_128_WI_2_validation | I2 | reinforce:none | 10 | 21.0428 | 0.0020 | 0.0002 | 0.9800 | 0.1406 | no significant difference |
| FOODS_2_128_WI_2_validation | I2 | trpo:ng_fixed | 10 | 0.0132 | 0.2754 | 0.3847 | 0.2400 | 1.0000 | no significant difference |
| FOODS_2_128_WI_2_validation | I2 | trpo:vpg | 10 | 0.1160 | 0.0098 | 0.0028 | 0.8000 | 0.3418 | no significant difference |
| FOODS_2_128_WI_2_validation | I3 | reinforce:loo_time | 10 | 1.3495 | 0.0020 | 0.0004 | 0.9400 | 0.1406 | no significant difference |
| FOODS_2_128_WI_2_validation | I3 | reinforce:none | 10 | 1.0078 | 0.0039 | 0.0010 | 0.8800 | 0.1562 | no significant difference |
| FOODS_2_128_WI_2_validation | I3 | trpo:ng_fixed | 10 | 0.1972 | 0.6250 | 0.7913 | 0.0800 | 1.0000 | no significant difference |
| FOODS_2_128_WI_2_validation | I3 | trpo:vpg | 10 | -0.0701 | 0.0488 | 0.0376 | -0.5600 | 1.0000 | no significant difference |
| FOODS_3_124_CA_3_validation | I1 | cm:central | 10 | 0.3617 | 0.0020 | 0.0002 | 1.0000 | 0.1406 | no significant difference |
| FOODS_3_124_CA_3_validation | I1 | cm:neldermead | 10 | 0.3978 | 0.0020 | 0.0002 | 1.0000 | 0.1406 | no significant difference |
| FOODS_3_124_CA_3_validation | I1 | cm:random | 10 | 0.3233 | 0.0020 | 0.0002 | 1.0000 | 0.1406 | no significant difference |
| FOODS_3_124_CA_3_validation | I1 | cm:spsa | 10 | 0.2080 | 0.0020 | 0.0002 | 1.0000 | 0.1406 | no significant difference |
| FOODS_3_124_CA_3_validation | I1 | reinforce:loo_time | 10 | 0.0220 | 0.1309 | 0.1405 | 0.4000 | 1.0000 | no significant difference |
| FOODS_3_124_CA_3_validation | I1 | reinforce:none | 10 | -0.0514 | 0.0020 | 0.0004 | -0.9400 | 0.1406 | no significant difference |
| FOODS_3_124_CA_3_validation | I1 | trpo:ng_fixed | 10 | -0.0275 | 0.0195 | 0.0113 | -0.6800 | 0.5859 | no significant difference |
| FOODS_3_124_CA_3_validation | I1 | trpo:vpg | 10 | -0.0472 | 0.0020 | 0.0028 | -0.8000 | 0.1406 | no significant difference |
| FOODS_3_124_CA_3_validation | I2 | cm:central | 10 | 0.1976 | 0.0020 | 0.0002 | 1.0000 | 0.1406 | no significant difference |
| FOODS_3_124_CA_3_validation | I2 | cm:neldermead | 10 | 0.3393 | 0.0020 | 0.0002 | 1.0000 | 0.1406 | no significant difference |
| FOODS_3_124_CA_3_validation | I2 | cm:random | 10 | 0.1856 | 0.0020 | 0.0008 | 0.9000 | 0.1406 | no significant difference |
| FOODS_3_124_CA_3_validation | I2 | cm:spsa | 10 | 0.3814 | 0.0020 | 0.0002 | 1.0000 | 0.1406 | no significant difference |
| FOODS_3_124_CA_3_validation | I2 | reinforce:loo_time | 10 | 0.3451 | 0.0020 | 0.0002 | 1.0000 | 0.1406 | no significant difference |
| FOODS_3_124_CA_3_validation | I2 | reinforce:none | 10 | 10.3816 | 0.0645 | 0.0211 | 0.6200 | 1.0000 | no significant difference |
| FOODS_3_124_CA_3_validation | I2 | trpo:ng_fixed | 10 | -0.0964 | 0.0020 | 0.0058 | -0.7400 | 0.1406 | no significant difference |
| FOODS_3_124_CA_3_validation | I2 | trpo:vpg | 10 | 0.0199 | 0.8457 | 0.7913 | 0.0800 | 1.0000 | no significant difference |
| FOODS_3_124_CA_3_validation | I3 | reinforce:loo_time | 10 | 0.1700 | 0.0020 | 0.0002 | 1.0000 | 0.1406 | no significant difference |
| FOODS_3_124_CA_3_validation | I3 | reinforce:none | 10 | 0.1731 | 0.0020 | 0.0002 | 1.0000 | 0.1406 | no significant difference |
| FOODS_3_124_CA_3_validation | I3 | trpo:ng_fixed | 10 | 0.0127 | 0.7695 | 0.7337 | 0.1000 | 1.0000 | no significant difference |
| FOODS_3_124_CA_3_validation | I3 | trpo:vpg | 10 | 0.0311 | 0.1055 | 0.1620 | 0.3800 | 1.0000 | no significant difference |
| HOBBIES_1_371_WI_1_validation | I1 | cm:central | 10 | 0.2770 | 0.0020 | 0.0002 | 1.0000 | 0.1406 | no significant difference |
| HOBBIES_1_371_WI_1_validation | I1 | cm:neldermead | 10 | 1.2081 | 0.0020 | 0.0002 | 1.0000 | 0.1406 | no significant difference |
| HOBBIES_1_371_WI_1_validation | I1 | cm:random | 10 | 0.2383 | 0.0020 | 0.0002 | 1.0000 | 0.1406 | no significant difference |
| HOBBIES_1_371_WI_1_validation | I1 | cm:spsa | 10 | 0.2458 | 0.0020 | 0.0002 | 1.0000 | 0.1406 | no significant difference |
| HOBBIES_1_371_WI_1_validation | I1 | reinforce:loo_time | 10 | 0.1020 | 0.0195 | 0.0140 | 0.6600 | 0.5859 | no significant difference |
| HOBBIES_1_371_WI_1_validation | I1 | reinforce:none | 10 | 0.1993 | 0.0020 | 0.0013 | 0.8600 | 0.1406 | no significant difference |
| HOBBIES_1_371_WI_1_validation | I1 | trpo:ng_fixed | 10 | -0.0576 | 0.0059 | 0.0073 | -0.7200 | 0.2168 | no significant difference |
| HOBBIES_1_371_WI_1_validation | I1 | trpo:vpg | 10 | 0.1536 | 0.0137 | 0.0058 | 0.7400 | 0.4375 | no significant difference |
| HOBBIES_1_371_WI_1_validation | I2 | cm:central | 10 | 0.1283 | 0.0059 | 0.0257 | 0.6000 | 0.2168 | no significant difference |
| HOBBIES_1_371_WI_1_validation | I2 | cm:neldermead | 10 | 0.5283 | 0.0020 | 0.0002 | 1.0000 | 0.1406 | no significant difference |
| HOBBIES_1_371_WI_1_validation | I2 | cm:random | 10 | 0.1517 | 0.0273 | 0.0173 | 0.6400 | 0.7383 | no significant difference |
| HOBBIES_1_371_WI_1_validation | I2 | cm:spsa | 10 | 0.2649 | 0.0039 | 0.0006 | 0.9200 | 0.1562 | no significant difference |
| HOBBIES_1_371_WI_1_validation | I2 | reinforce:loo_time | 10 | 22.8093 | 0.0137 | 0.0090 | 0.7000 | 0.4375 | no significant difference |
| HOBBIES_1_371_WI_1_validation | I2 | reinforce:none | 10 | 22.7582 | 0.0645 | 0.0889 | 0.4600 | 1.0000 | no significant difference |
| HOBBIES_1_371_WI_1_validation | I2 | trpo:ng_fixed | 10 | -0.0435 | 0.5566 | 0.4727 | -0.2000 | 1.0000 | no significant difference |
| HOBBIES_1_371_WI_1_validation | I2 | trpo:vpg | 10 | -0.1263 | 0.0273 | 0.0140 | -0.6600 | 0.7383 | no significant difference |
| HOBBIES_1_371_WI_1_validation | I3 | reinforce:loo_time | 10 | 0.2420 | 0.0020 | 0.0002 | 0.9800 | 0.1406 | no significant difference |
| HOBBIES_1_371_WI_1_validation | I3 | reinforce:none | 10 | 0.2563 | 0.0098 | 0.0013 | 0.8600 | 0.3418 | no significant difference |
| HOBBIES_1_371_WI_1_validation | I3 | trpo:ng_fixed | 10 | -0.0384 | 0.0840 | 0.0539 | -0.5200 | 1.0000 | no significant difference |
| HOBBIES_1_371_WI_1_validation | I3 | trpo:vpg | 10 | -0.0044 | 0.4316 | 0.0257 | -0.6000 | 1.0000 | no significant difference |
| TECHM | T1 | cm:central | 10 | 0.0320 | 0.4922 | 0.3075 | 0.2800 | 1.0000 | no significant difference |
| TECHM | T1 | cm:neldermead | 10 | 0.0837 | 0.1934 | 0.1620 | 0.3800 | 1.0000 | no significant difference |
| TECHM | T1 | cm:random | 10 | 0.1478 | 0.0645 | 0.0539 | 0.5200 | 1.0000 | no significant difference |
| TECHM | T1 | cm:spsa | 10 | 0.0963 | 0.1055 | 0.0757 | 0.4800 | 1.0000 | no significant difference |
| TECHM | T1 | reinforce:loo_time | 10 | -0.0357 | 0.9219 | 0.9698 | -0.0200 | 1.0000 | no significant difference |
| TECHM | T1 | reinforce:none | 10 | -0.0124 | 0.8457 | 0.8501 | 0.0600 | 1.0000 | no significant difference |
| TECHM | T1 | trpo:ng_fixed | 10 | -0.0097 | 0.8457 | 0.7913 | -0.0800 | 1.0000 | no significant difference |
| TECHM | T1 | trpo:vpg | 10 | 0.1031 | 0.3223 | 0.3075 | 0.2800 | 1.0000 | no significant difference |
| TECHM | T2 | reinforce:loo_time | 10 | -0.1049 | 0.2324 | 0.1859 | -0.3600 | 1.0000 | no significant difference |
| TECHM | T2 | reinforce:none | 10 | -0.0898 | 0.3223 | 0.3447 | -0.2600 | 1.0000 | no significant difference |
| TECHM | T2 | trpo:ng_fixed | 10 | 0.1500 | 0.0371 | 0.0757 | 0.4800 | 0.9277 | no significant difference |
| TECHM | T2 | trpo:vpg | 10 | 0.0680 | 0.4922 | 0.5205 | 0.1800 | 1.0000 | no significant difference |

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