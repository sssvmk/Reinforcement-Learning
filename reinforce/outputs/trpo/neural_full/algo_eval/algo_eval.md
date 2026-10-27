# TRPO algorithm evaluation (neural_full mode)

Return units: validation return / reward normaliser (higher is better). `final` = last checkpoint, deterministic policy.

## Per problem / policy / variant

| key | policy | method | n | final | ci_lo | ci_hi | eps95_median | reached95 | worst_drop_median | seed_sd | converge | stall | collapse | diverge |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FOODS_2_128_WI_2_validation | I4 | trpo:trpo | 10 | 0.6535 | 0.6118 | 0.6938 | 4992.0000 | 0.9000 | -0.0957 | 0.0698 | 0.6000 | 0.3000 | 0.1000 | 0.0000 |
| FOODS_2_128_WI_2_validation | I4 | trpo:vpg | 10 | 0.7663 | 0.7370 | 0.7939 | 2016.0000 | 1.0000 | -0.1162 | 0.0505 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_3_124_CA_3_validation | I4 | trpo:trpo | 10 | 2.6418 | 2.5565 | 2.7142 | 3504.0000 | 1.0000 | -0.2601 | 0.1376 | 0.9000 | 0.1000 | 0.0000 | 0.0000 |
| FOODS_3_124_CA_3_validation | I4 | trpo:vpg | 10 | 2.8850 | 2.7416 | 2.9826 | 1024.0000 | 1.0000 | -0.4878 | 0.2212 | 0.9000 | 0.1000 | 0.0000 | 0.0000 |
| HOBBIES_1_371_WI_1_validation | I4 | trpo:trpo | 10 | 3.1635 | 3.0947 | 3.2216 | 3504.0000 | 1.0000 | -0.3770 | 0.1077 | 0.9000 | 0.1000 | 0.0000 | 0.0000 |
| HOBBIES_1_371_WI_1_validation | I4 | trpo:vpg | 10 | 3.1520 | 3.0376 | 3.2511 | 3008.0000 | 1.0000 | -0.5198 | 0.1875 | 0.9000 | 0.0000 | 0.1000 | 0.0000 |
| TECHM | T3 | trpo:trpo | 10 | 0.4466 | 0.3337 | 0.5686 | 5488.0000 | 0.8000 | -0.1011 | 0.2099 | 0.3000 | 0.5000 | 0.2000 | 0.0000 |
| TECHM | T3 | trpo:vpg | 10 | 0.7612 | 0.6551 | 0.8606 | 1024.0000 | 1.0000 | -0.0425 | 0.1797 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |

## Update quality (trust-region check)

| key | policy | method | frac_improved | mean_KL | max_KL | backtracks | t_update_ms | t_collect_ms |
|---|---|---|---|---|---|---|---|---|
| FOODS_2_128_WI_2_validation | I4 | trpo:trpo | 1.0000 | 0.0148 | 0.0300 | 0.6742 | 53.9606 | 175.3368 |
| FOODS_2_128_WI_2_validation | I4 | trpo:vpg | 0.7579 | 0.0157 | 2.4068 | 0.0000 | 28.1993 | 159.9394 |
| FOODS_3_124_CA_3_validation | I4 | trpo:trpo | 1.0000 | 0.0550 | 0.1000 | 0.5560 | 51.9656 | 157.4222 |
| FOODS_3_124_CA_3_validation | I4 | trpo:vpg | 0.7401 | 0.1236 | 8.2124 | 0.0000 | 27.0138 | 159.4610 |
| HOBBIES_1_371_WI_1_validation | I4 | trpo:trpo | 1.0000 | 0.0529 | 0.1000 | 0.6020 | 53.6833 | 167.1769 |
| HOBBIES_1_371_WI_1_validation | I4 | trpo:vpg | 0.7841 | 1.1367 | 1801.6467 | 0.0000 | 26.6312 | 153.0471 |
| TECHM | T3 | trpo:trpo | 1.0000 | 0.0053 | 0.0100 | 0.6154 | 96.4864 | 639.8002 |
| TECHM | T3 | trpo:vpg | 0.8298 | 0.0002 | 0.0217 | 0.0000 | 40.3485 | 558.6557 |

## TRPO update diagnostics

| key | policy | corr_surrogate_vs_realised | cos_CG_vs_exact | cg_residual | vf_r2 | frac_accepted |
|---|---|---|---|---|---|---|
| FOODS_2_128_WI_2_validation | I4 | -0.0177 | nan | 0.0269 | 0.4969 | 1.0000 |
| FOODS_3_124_CA_3_validation | I4 | -0.0022 | nan | 0.0347 | 0.7844 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I4 | -0.0154 | nan | 0.0436 | 0.4298 | 1.0000 |
| TECHM | T3 | -0.0203 | nan | 0.0968 | 0.2145 | 1.0000 |

## TRPO vs ablations (paired by seed, Wilcoxon, Holm-adjusted over all rows)

| key | policy | vs | diff_trpo_minus | p_wilcoxon | cliffs_delta | p_holm |
|---|---|---|---|---|---|---|
| FOODS_2_128_WI_2_validation | I4 | trpo:vpg | -0.1128 | 0.0020 | -0.8600 | 0.0078 |
| FOODS_3_124_CA_3_validation | I4 | trpo:vpg | -0.2432 | 0.0645 | -0.8000 | 0.1289 |
| HOBBIES_1_371_WI_1_validation | I4 | trpo:vpg | 0.0115 | 1.0000 | -0.1200 | 1.0000 |
| TECHM | T3 | trpo:vpg | -0.3146 | 0.0098 | -0.7600 | 0.0293 |
