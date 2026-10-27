# TRPO vs previous methods (ent_quick mode)

Baselines are **re-used** from `outputs/trpo\full` of this folder (same seeds, same theta0). TRPO counts pilot + training batches against the same budget; validation/diagnostic rollouts are excluded for all methods. Tuning effort differs: TRPO tunes delta x lambda (12 configs) and VPG lr (3); REINFORCE tuned lr x B x entropy (18).

Previous results unchanged during comparison (hash check of outputs/full): **True**

## Summary (validation return, deterministic policy, last checkpoint)

| key | policy | method | n | final | ci_lo | ci_hi | eps95_median | reached95 |
|---|---|---|---|---|---|---|---|---|
| FOODS_2_128_WI_2_validation | I1 | trpo:trpo | 10 | 0.6775 | 0.6594 | 0.6971 | 528.0000 | 1.0000 |
| FOODS_2_128_WI_2_validation | I1 | trpo:trpo_ent | 3 | 0.0769 | -0.1231 | 0.2216 | 32.0000 | 1.0000 |
| FOODS_2_128_WI_2_validation | I2 | trpo:trpo | 10 | 0.7544 | 0.7239 | 0.7803 | 10944.0000 | 1.0000 |
| FOODS_2_128_WI_2_validation | I2 | trpo:trpo_ent | 3 | 0.5414 | 0.4513 | 0.6175 | 2080.0000 | 1.0000 |
| FOODS_2_128_WI_2_validation | I3 | trpo:trpo | 10 | 0.7031 | 0.6648 | 0.7426 | 1024.0000 | 1.0000 |
| FOODS_2_128_WI_2_validation | I3 | trpo:trpo_ent | 3 | 0.6234 | 0.4901 | 0.7911 | 2208.0000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I1 | trpo:trpo | 10 | 2.8145 | 2.8031 | 2.8266 | 1520.0000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I1 | trpo:trpo_ent | 3 | 2.7857 | 2.7536 | 2.8017 | 672.0000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I2 | trpo:trpo | 10 | 2.8979 | 2.8608 | 2.9349 | 1024.0000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I2 | trpo:trpo_ent | 3 | 2.2517 | 2.2292 | 2.2667 | 672.0000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I3 | trpo:trpo | 10 | 3.0481 | 3.0224 | 3.0739 | 1024.0000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I3 | trpo:trpo_ent | 3 | 2.6837 | 2.2368 | 2.9542 | 416.0000 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I1 | trpo:trpo | 10 | 3.2546 | 3.2258 | 3.2831 | 3008.0000 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I1 | trpo:trpo_ent | 3 | 3.1913 | 3.1317 | 3.2214 | 1056.0000 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I2 | trpo:trpo | 10 | 3.1495 | 3.0733 | 3.2166 | 2016.0000 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I2 | trpo:trpo_ent | 3 | 1.7557 | 1.4620 | 2.2635 | 672.0000 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I3 | trpo:trpo | 10 | 3.2290 | 3.1982 | 3.2562 | 1024.0000 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I3 | trpo:trpo_ent | 3 | 3.2341 | 3.1620 | 3.3309 | 800.0000 | 1.0000 |
| TECHM | T1 | trpo:trpo | 10 | 0.7694 | 0.6133 | 0.8935 | 1024.0000 | 0.8000 |
| TECHM | T1 | trpo:trpo_ent | 3 | 0.7869 | 0.5949 | 0.8843 | 1568.0000 | 1.0000 |
| TECHM | T2 | trpo:trpo | 10 | 0.6804 | 0.5689 | 0.7776 | 1024.0000 | 0.9000 |
| TECHM | T2 | trpo:trpo_ent | 3 | 0.6852 | 0.5992 | 0.8388 | 544.0000 | 1.0000 |

## TRPO vs each method (paired by seed, Wilcoxon, Holm over all rows)

| key | policy | vs | n_pairs | diff_trpo_minus | p_wilcoxon_paired | p_mannwhitney | cliffs_delta | p_holm | verdict |
|---|---|---|---|---|---|---|---|---|---|
| FOODS_2_128_WI_2_validation | I1 | trpo:trpo | 3 | -0.6080 | 0.2500 | 0.1000 | -1.0000 | 1.0000 | no significant difference |
| FOODS_2_128_WI_2_validation | I2 | trpo:trpo | 3 | -0.2345 | 0.2500 | 0.1000 | -1.0000 | 1.0000 | no significant difference |
| FOODS_2_128_WI_2_validation | I3 | trpo:trpo | 3 | -0.1381 | 0.2500 | 0.4000 | -0.5556 | 1.0000 | no significant difference |
| FOODS_3_124_CA_3_validation | I1 | trpo:trpo | 3 | -0.0217 | 0.5000 | 0.4000 | -0.5556 | 1.0000 | no significant difference |
| FOODS_3_124_CA_3_validation | I2 | trpo:trpo | 3 | -0.6431 | 0.2500 | 0.1000 | -1.0000 | 1.0000 | no significant difference |
| FOODS_3_124_CA_3_validation | I3 | trpo:trpo | 3 | -0.3426 | 0.2500 | 0.1000 | -1.0000 | 1.0000 | no significant difference |
| HOBBIES_1_371_WI_1_validation | I1 | trpo:trpo | 3 | -0.1016 | 0.2500 | 0.1000 | -1.0000 | 1.0000 | no significant difference |
| HOBBIES_1_371_WI_1_validation | I2 | trpo:trpo | 3 | -1.3794 | 0.2500 | 0.1000 | -1.0000 | 1.0000 | no significant difference |
| HOBBIES_1_371_WI_1_validation | I3 | trpo:trpo | 3 | -0.0272 | 0.7500 | 0.7000 | -0.3333 | 1.0000 | no significant difference |
| TECHM | T1 | trpo:trpo | 3 | 0.2165 | 0.5000 | 0.7000 | 0.3333 | 1.0000 | no significant difference |
| TECHM | T2 | trpo:trpo | 3 | 0.0681 | 0.5000 | 0.7000 | 0.3333 | 1.0000 | no significant difference |

## Verdict counts

| vs | no significant difference |
|---|---|
| trpo:trpo | 11 |