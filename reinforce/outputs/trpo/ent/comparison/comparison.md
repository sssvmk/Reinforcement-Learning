# TRPO vs previous methods (ent mode)

Baselines are **re-used** from `outputs/trpo\full` of this folder (same seeds, same theta0). TRPO counts pilot + training batches against the same budget; validation/diagnostic rollouts are excluded for all methods. Tuning effort differs: TRPO tunes delta x lambda (12 configs) and VPG lr (3); REINFORCE tuned lr x B x entropy (18).

Previous results unchanged during comparison (hash check of outputs/full): **True**

## Summary (validation return, deterministic policy, last checkpoint)

| key | policy | method | n | final | ci_lo | ci_hi | eps95_median | reached95 |
|---|---|---|---|---|---|---|---|---|
| FOODS_2_128_WI_2_validation | I1 | trpo:trpo | 10 | 0.6775 | 0.6594 | 0.6971 | 528.0000 | 1.0000 |
| FOODS_2_128_WI_2_validation | I1 | trpo:trpo_ent | 10 | 0.6502 | 0.6278 | 0.6739 | 528.0000 | 1.0000 |
| FOODS_2_128_WI_2_validation | I2 | trpo:trpo | 10 | 0.7544 | 0.7239 | 0.7803 | 10448.0000 | 1.0000 |
| FOODS_2_128_WI_2_validation | I2 | trpo:trpo_ent | 10 | 0.7538 | 0.7284 | 0.7745 | 8960.0000 | 1.0000 |
| FOODS_2_128_WI_2_validation | I3 | trpo:trpo | 10 | 0.7031 | 0.6648 | 0.7426 | 1024.0000 | 1.0000 |
| FOODS_2_128_WI_2_validation | I3 | trpo:trpo_ent | 10 | 0.7527 | 0.7331 | 0.7736 | 1520.0000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I1 | trpo:trpo | 10 | 2.8145 | 2.8031 | 2.8266 | 1520.0000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I1 | trpo:trpo_ent | 10 | 2.8092 | 2.7961 | 2.8233 | 1024.0000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I2 | trpo:trpo | 10 | 2.8979 | 2.8608 | 2.9349 | 1024.0000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I2 | trpo:trpo_ent | 10 | 2.7929 | 2.7631 | 2.8353 | 1024.0000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I3 | trpo:trpo | 10 | 3.0481 | 3.0224 | 3.0739 | 1024.0000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I3 | trpo:trpo_ent | 10 | 3.0599 | 3.0512 | 3.0676 | 1024.0000 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I1 | trpo:trpo | 10 | 3.2546 | 3.2258 | 3.2831 | 3008.0000 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I1 | trpo:trpo_ent | 10 | 3.2242 | 3.1763 | 3.2690 | 3008.0000 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I2 | trpo:trpo | 10 | 3.1495 | 3.0733 | 3.2166 | 2016.0000 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I2 | trpo:trpo_ent | 10 | 3.1056 | 3.0141 | 3.1752 | 2016.0000 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I3 | trpo:trpo | 10 | 3.2290 | 3.1982 | 3.2562 | 1024.0000 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I3 | trpo:trpo_ent | 10 | 3.2416 | 3.2187 | 3.2652 | 1024.0000 | 1.0000 |
| TECHM | T1 | trpo:trpo | 10 | 0.7694 | 0.6133 | 0.8935 | 1520.0000 | 0.8000 |
| TECHM | T1 | trpo:trpo_ent | 10 | 0.8291 | 0.7438 | 0.9216 | 3008.0000 | 0.9000 |
| TECHM | T2 | trpo:trpo | 10 | 0.6804 | 0.5689 | 0.7776 | 1024.0000 | 0.9000 |
| TECHM | T2 | trpo:trpo_ent | 10 | 0.6503 | 0.5094 | 0.7804 | 1520.0000 | 1.0000 |

## TRPO vs each method (paired by seed, Wilcoxon, Holm over all rows)

| key | policy | vs | n_pairs | diff_trpo_minus | p_wilcoxon_paired | p_mannwhitney | cliffs_delta | p_holm | verdict |
|---|---|---|---|---|---|---|---|---|---|
| FOODS_2_128_WI_2_validation | I1 | trpo:trpo | 10 | -0.0273 | 0.0020 | 0.0757 | -0.4800 | 0.0215 | TRPO worse |
| FOODS_2_128_WI_2_validation | I2 | trpo:trpo | 10 | -0.0005 | 0.9219 | 0.6232 | -0.1400 | 1.0000 | no significant difference |
| FOODS_2_128_WI_2_validation | I3 | trpo:trpo | 10 | 0.0495 | 0.1055 | 0.1041 | 0.4400 | 0.6328 | no significant difference |
| FOODS_3_124_CA_3_validation | I1 | trpo:trpo | 10 | -0.0054 | 0.0840 | 0.4727 | -0.2000 | 0.5879 | no significant difference |
| FOODS_3_124_CA_3_validation | I2 | trpo:trpo | 10 | -0.1051 | 0.0020 | 0.0017 | -0.8400 | 0.0215 | TRPO worse |
| FOODS_3_124_CA_3_validation | I3 | trpo:trpo | 10 | 0.0118 | 0.3750 | 0.8501 | 0.0600 | 1.0000 | no significant difference |
| HOBBIES_1_371_WI_1_validation | I1 | trpo:trpo | 10 | -0.0304 | 0.0273 | 0.5708 | -0.1600 | 0.2461 | no significant difference |
| HOBBIES_1_371_WI_1_validation | I2 | trpo:trpo | 10 | -0.0439 | 0.0645 | 0.3075 | -0.2800 | 0.5156 | no significant difference |
| HOBBIES_1_371_WI_1_validation | I3 | trpo:trpo | 10 | 0.0125 | 0.3750 | 0.6776 | 0.1200 | 1.0000 | no significant difference |
| TECHM | T1 | trpo:trpo | 10 | 0.0597 | 0.4316 | 0.8501 | 0.0600 | 1.0000 | no significant difference |
| TECHM | T2 | trpo:trpo | 10 | -0.0300 | 0.6953 | 0.9698 | -0.0200 | 1.0000 | no significant difference |

## Verdict counts

| vs | TRPO worse | no significant difference |
|---|---|---|
| trpo:trpo | 2 | 9 |