# TRPO algorithm evaluation (ent_quick mode)

Return units: validation return / reward normaliser (higher is better). `final` = last checkpoint, deterministic policy.

## Per problem / policy / variant

| key | policy | method | n | final | ci_lo | ci_hi | eps95_median | reached95 | worst_drop_median | seed_sd | converge | stall | collapse | diverge |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FOODS_2_128_WI_2_validation | I1 | trpo:trpo_ent | 3 | 0.0769 | -0.1231 | 0.2216 | 32.0000 | 1.0000 | -0.1876 | 0.1788 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| FOODS_2_128_WI_2_validation | I2 | trpo:trpo_ent | 3 | 0.5414 | 0.4513 | 0.6175 | 32.0000 | 1.0000 | -0.1545 | 0.0840 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| FOODS_2_128_WI_2_validation | I3 | trpo:trpo_ent | 3 | 0.6234 | 0.4901 | 0.7911 | 2208.0000 | 1.0000 | -0.4304 | 0.1534 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_3_124_CA_3_validation | I1 | trpo:trpo_ent | 3 | 2.7857 | 2.7536 | 2.8017 | 544.0000 | 1.0000 | -0.1726 | 0.0278 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_3_124_CA_3_validation | I2 | trpo:trpo_ent | 3 | 2.2517 | 2.2292 | 2.2667 | 32.0000 | 1.0000 | -0.3034 | 0.0198 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| FOODS_3_124_CA_3_validation | I3 | trpo:trpo_ent | 3 | 2.6837 | 2.2368 | 2.9542 | 416.0000 | 1.0000 | -0.0710 | 0.3899 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| HOBBIES_1_371_WI_1_validation | I1 | trpo:trpo_ent | 3 | 3.1913 | 3.1317 | 3.2214 | 1056.0000 | 1.0000 | -0.1169 | 0.0516 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| HOBBIES_1_371_WI_1_validation | I2 | trpo:trpo_ent | 3 | 1.7557 | 1.4620 | 2.2635 | 32.0000 | 1.0000 | -0.2286 | 0.4416 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| HOBBIES_1_371_WI_1_validation | I3 | trpo:trpo_ent | 3 | 3.2341 | 3.1620 | 3.3309 | 672.0000 | 1.0000 | -0.0809 | 0.0871 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| TECHM | T1 | trpo:trpo_ent | 3 | 0.7869 | 0.5949 | 0.8843 | 1568.0000 | 1.0000 | -0.0594 | 0.1662 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| TECHM | T2 | trpo:trpo_ent | 3 | 0.6852 | 0.5992 | 0.8388 | 544.0000 | 1.0000 | -0.0343 | 0.1333 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |

## Update quality (trust-region check)

| key | policy | method | frac_improved | mean_KL | max_KL | backtracks | t_update_ms | t_collect_ms |
|---|---|---|---|---|---|---|---|---|
| FOODS_2_128_WI_2_validation | I1 | trpo:trpo_ent | 1.0000 | 0.0184 | 0.0300 | 0.4606 | 2.1548 | 6.5639 |
| FOODS_2_128_WI_2_validation | I2 | trpo:trpo_ent | 1.0000 | 0.0174 | 0.0300 | 0.5161 | 2.1708 | 6.7117 |
| FOODS_2_128_WI_2_validation | I3 | trpo:trpo_ent | 1.0000 | 0.0192 | 0.0300 | 0.4444 | 28.6891 | 14.1925 |
| FOODS_3_124_CA_3_validation | I1 | trpo:trpo_ent | 1.0000 | 0.0172 | 0.0300 | 0.5305 | 2.2056 | 6.5816 |
| FOODS_3_124_CA_3_validation | I2 | trpo:trpo_ent | 1.0000 | 0.0197 | 0.0299 | 0.3728 | 1.8579 | 6.5796 |
| FOODS_3_124_CA_3_validation | I3 | trpo:trpo_ent | 1.0000 | 0.0183 | 0.0300 | 0.5000 | 26.2037 | 14.1958 |
| HOBBIES_1_371_WI_1_validation | I1 | trpo:trpo_ent | 1.0000 | 0.0067 | 0.0100 | 0.3996 | 2.2291 | 6.6211 |
| HOBBIES_1_371_WI_1_validation | I2 | trpo:trpo_ent | 1.0000 | 0.0199 | 0.0300 | 0.3674 | 2.0392 | 6.7926 |
| HOBBIES_1_371_WI_1_validation | I3 | trpo:trpo_ent | 1.0000 | 0.0065 | 0.0100 | 0.4444 | 30.3434 | 16.5162 |
| TECHM | T1 | trpo:trpo_ent | 1.0000 | 0.0165 | 0.0300 | 0.5609 | 9.4874 | 14.5294 |
| TECHM | T2 | trpo:trpo_ent | 1.0000 | 0.0053 | 0.0100 | 0.6165 | 43.6413 | 34.8227 |

## TRPO update diagnostics

| key | policy | corr_surrogate_vs_realised | cos_CG_vs_exact | cg_residual | vf_r2 | frac_accepted |
|---|---|---|---|---|---|---|
| FOODS_2_128_WI_2_validation | I1 | 0.1156 | 1.0000 | 0.0002 | 0.3047 | 1.0000 |
| FOODS_2_128_WI_2_validation | I2 | 0.0145 | 1.0000 | 0.0001 | 0.4503 | 1.0000 |
| FOODS_2_128_WI_2_validation | I3 | -0.0211 | 0.9995 | 0.0069 | 0.4178 | 1.0000 |
| FOODS_3_124_CA_3_validation | I1 | 0.3043 | 1.0000 | 0.0002 | 0.7986 | 1.0000 |
| FOODS_3_124_CA_3_validation | I2 | -0.0876 | 1.0000 | 0.0002 | 0.7132 | 1.0000 |
| FOODS_3_124_CA_3_validation | I3 | 0.0553 | 0.9994 | 0.0102 | 0.7550 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I1 | 0.1904 | 1.0000 | 0.0001 | 0.3108 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I2 | 0.1719 | 0.9999 | 0.0001 | 0.3206 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I3 | 0.0276 | 0.9996 | 0.0069 | 0.3712 | 1.0000 |
| TECHM | T1 | 0.0100 | 0.9971 | 0.0005 | 0.0265 | 1.0000 |
| TECHM | T2 | 0.2653 | 1.0000 | 0.0002 | 0.0059 | 1.0000 |

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
