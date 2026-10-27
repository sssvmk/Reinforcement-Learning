# TRPO algorithm evaluation (ent mode)

Return units: validation return / reward normaliser (higher is better). `final` = last checkpoint, deterministic policy.

## Per problem / policy / variant

| key | policy | method | n | final | ci_lo | ci_hi | eps95_median | reached95 | worst_drop_median | seed_sd | converge | stall | collapse | diverge |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FOODS_2_128_WI_2_validation | I1 | trpo:trpo_ent | 10 | 0.6502 | 0.6278 | 0.6739 | 528.0000 | 1.0000 | -0.0569 | 0.0391 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_2_128_WI_2_validation | I2 | trpo:trpo_ent | 10 | 0.7538 | 0.7284 | 0.7745 | 8960.0000 | 1.0000 | -0.0720 | 0.0386 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_2_128_WI_2_validation | I3 | trpo:trpo_ent | 10 | 0.7527 | 0.7331 | 0.7736 | 1520.0000 | 1.0000 | -0.1630 | 0.0339 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_3_124_CA_3_validation | I1 | trpo:trpo_ent | 10 | 2.8092 | 2.7961 | 2.8233 | 1024.0000 | 1.0000 | -0.0566 | 0.0234 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_3_124_CA_3_validation | I2 | trpo:trpo_ent | 10 | 2.7929 | 2.7631 | 2.8353 | 1024.0000 | 1.0000 | -0.2075 | 0.0641 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_3_124_CA_3_validation | I3 | trpo:trpo_ent | 10 | 3.0599 | 3.0512 | 3.0676 | 1024.0000 | 1.0000 | -0.0353 | 0.0141 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| HOBBIES_1_371_WI_1_validation | I1 | trpo:trpo_ent | 10 | 3.2242 | 3.1763 | 3.2690 | 3008.0000 | 1.0000 | -0.0746 | 0.0801 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| HOBBIES_1_371_WI_1_validation | I2 | trpo:trpo_ent | 10 | 3.1056 | 3.0141 | 3.1752 | 1520.0000 | 1.0000 | -0.2933 | 0.1378 | 0.9000 | 0.1000 | 0.0000 | 0.0000 |
| HOBBIES_1_371_WI_1_validation | I3 | trpo:trpo_ent | 10 | 3.2416 | 3.2187 | 3.2652 | 1024.0000 | 1.0000 | -0.1218 | 0.0393 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| TECHM | T1 | trpo:trpo_ent | 10 | 0.8291 | 0.7438 | 0.9216 | 3008.0000 | 0.9000 | -0.0630 | 0.1523 | 0.9000 | 0.1000 | 0.0000 | 0.0000 |
| TECHM | T2 | trpo:trpo_ent | 10 | 0.6503 | 0.5094 | 0.7804 | 1520.0000 | 1.0000 | -0.0737 | 0.2348 | 0.9000 | 0.0000 | 0.1000 | 0.0000 |

## Update quality (trust-region check)

| key | policy | method | frac_improved | mean_KL | max_KL | backtracks | t_update_ms | t_collect_ms |
|---|---|---|---|---|---|---|---|---|
| FOODS_2_128_WI_2_validation | I1 | trpo:trpo_ent | 1.0000 | 0.0053 | 0.0100 | 0.6072 | 2.6074 | 7.5159 |
| FOODS_2_128_WI_2_validation | I2 | trpo:trpo_ent | 1.0000 | 0.0053 | 0.0100 | 0.6097 | 2.2492 | 6.5388 |
| FOODS_2_128_WI_2_validation | I3 | trpo:trpo_ent | 1.0000 | 0.0164 | 0.0300 | 0.5860 | 39.6368 | 20.9022 |
| FOODS_3_124_CA_3_validation | I1 | trpo:trpo_ent | 1.0000 | 0.0017 | 0.0030 | 0.5819 | 2.1974 | 6.2590 |
| FOODS_3_124_CA_3_validation | I2 | trpo:trpo_ent | 1.0000 | 0.0506 | 0.1000 | 0.6024 | 1.9006 | 5.7635 |
| FOODS_3_124_CA_3_validation | I3 | trpo:trpo_ent | 1.0000 | 0.0019 | 0.0030 | 0.4576 | 28.5788 | 13.8471 |
| HOBBIES_1_371_WI_1_validation | I1 | trpo:trpo_ent | 1.0000 | 0.0017 | 0.0030 | 0.5713 | 3.1456 | 9.2848 |
| HOBBIES_1_371_WI_1_validation | I2 | trpo:trpo_ent | 1.0000 | 0.0159 | 0.0300 | 0.5909 | 3.9634 | 13.6649 |
| HOBBIES_1_371_WI_1_validation | I3 | trpo:trpo_ent | 1.0000 | 0.0062 | 0.0100 | 0.4793 | 34.3202 | 18.4343 |
| TECHM | T1 | trpo:trpo_ent | 1.0000 | 0.0055 | 0.0100 | 0.5717 | 9.1466 | 14.1970 |
| TECHM | T2 | trpo:trpo_ent | 1.0000 | 0.0151 | 0.0300 | 0.6486 | 32.7827 | 24.9759 |

## TRPO update diagnostics

| key | policy | corr_surrogate_vs_realised | cos_CG_vs_exact | cg_residual | vf_r2 | frac_accepted |
|---|---|---|---|---|---|---|
| FOODS_2_128_WI_2_validation | I1 | 0.0761 | 1.0000 | 0.0001 | 0.3834 | 1.0000 |
| FOODS_2_128_WI_2_validation | I2 | -0.0322 | 1.0000 | 0.0000 | 0.4713 | 1.0000 |
| FOODS_2_128_WI_2_validation | I3 | -0.0147 | 0.9979 | 0.0079 | 0.4900 | 1.0000 |
| FOODS_3_124_CA_3_validation | I1 | 0.1263 | 1.0000 | 0.0002 | 0.8151 | 1.0000 |
| FOODS_3_124_CA_3_validation | I2 | -0.0835 | 1.0000 | 0.0001 | 0.8095 | 1.0000 |
| FOODS_3_124_CA_3_validation | I3 | 0.2659 | 0.9983 | 0.0113 | 0.8431 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I1 | 0.0656 | 1.0000 | 0.0001 | 0.3053 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I2 | 0.0110 | 1.0000 | 0.0001 | 0.4024 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I3 | -0.0513 | 0.9967 | 0.0173 | 0.4850 | 1.0000 |
| TECHM | T1 | 0.0133 | 0.9877 | 0.0021 | 0.0472 | 1.0000 |
| TECHM | T2 | 0.1413 | 1.0000 | 0.0002 | 0.0059 | 1.0000 |

## TRPO vs ablations (paired by seed, Wilcoxon, Holm-adjusted over all rows)

| key | policy | vs | diff_trpo_minus | p_wilcoxon | cliffs_delta | p_holm |
|---|---|---|---|---|---|---|
| FOODS_2_128_WI_2_validation | I1 | trpo:trpo_ent | nan | 1.0000 | nan | 1.0000 |
| FOODS_2_128_WI_2_validation | I2 | trpo:trpo_ent | nan | 1.0000 | nan | 1.0000 |
| FOODS_2_128_WI_2_validation | I3 | trpo:trpo_ent | nan | 1.0000 | nan | 1.0000 |
| FOODS_3_124_CA_3_validation | I1 | trpo:trpo_ent | nan | 1.0000 | nan | 1.0000 |
| FOODS_3_124_CA_3_validation | I2 | trpo:trpo_ent | nan | 1.0000 | nan | 1.0000 |
| FOODS_3_124_CA_3_validation | I3 | trpo:trpo_ent | nan | 1.0000 | nan | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I1 | trpo:trpo_ent | nan | 1.0000 | nan | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I2 | trpo:trpo_ent | nan | 1.0000 | nan | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I3 | trpo:trpo_ent | nan | 1.0000 | nan | 1.0000 |
| TECHM | T1 | trpo:trpo_ent | nan | 1.0000 | nan | 1.0000 |
| TECHM | T2 | trpo:trpo_ent | nan | 1.0000 | nan | 1.0000 |
