# TRPO algorithm evaluation (neural_quick mode)

Return units: validation return / reward normaliser (higher is better). `final` = last checkpoint, deterministic policy.

## Per problem / policy / variant

| key | policy | method | n | final | ci_lo | ci_hi | eps95_median | reached95 | worst_drop_median | seed_sd | converge | stall | collapse | diverge |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FOODS_2_128_WI_2_validation | I4 | trpo:trpo | 2 | 0.6189 | 0.6058 | 0.6320 | 32.0000 | 0.5000 | -0.0925 | 0.0185 | 0.0000 | 0.5000 | 0.5000 | 0.0000 |
| FOODS_2_128_WI_2_validation | I4 | trpo:vpg | 2 | 0.7665 | 0.7490 | 0.7840 | 352.0000 | 1.0000 | -0.2255 | 0.0247 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_3_124_CA_3_validation | I4 | trpo:trpo | 2 | 2.8047 | 2.6989 | 2.9104 | 544.0000 | 0.5000 | -0.0553 | 0.1495 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| FOODS_3_124_CA_3_validation | I4 | trpo:vpg | 2 | 2.7774 | 2.7160 | 2.8388 | 768.0000 | 1.0000 | -0.4342 | 0.0868 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| HOBBIES_1_371_WI_1_validation | I4 | trpo:trpo | 2 | 2.9631 | 2.7093 | 3.2168 | 992.0000 | 0.5000 | -0.0991 | 0.3588 | 0.5000 | 0.5000 | 0.0000 | 0.0000 |
| HOBBIES_1_371_WI_1_validation | I4 | trpo:vpg | 2 | 2.9318 | 2.9111 | 2.9525 | 256.0000 | 1.0000 | -0.4494 | 0.0293 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| TECHM | T3 | trpo:trpo | 2 | 0.3676 | 0.0339 | 0.7014 | 32.0000 | 0.5000 | -0.0770 | 0.4720 | 0.5000 | 0.0000 | 0.5000 | 0.0000 |
| TECHM | T3 | trpo:vpg | 2 | 0.5286 | 0.2459 | 0.8112 | 32.0000 | 0.5000 | -0.0345 | 0.3997 | 0.5000 | 0.0000 | 0.5000 | 0.0000 |

## Update quality (trust-region check)

| key | policy | method | frac_improved | mean_KL | max_KL | backtracks | t_update_ms | t_collect_ms |
|---|---|---|---|---|---|---|---|---|
| FOODS_2_128_WI_2_validation | I4 | trpo:trpo | 1.0000 | 0.0113 | 0.0291 | 1.0000 | 54.2894 | 138.7505 |
| FOODS_2_128_WI_2_validation | I4 | trpo:vpg | 0.7500 | 0.0682 | 0.9739 | 0.0000 | 26.6999 | 137.2675 |
| FOODS_3_124_CA_3_validation | I4 | trpo:trpo | 1.0000 | 0.0118 | 0.0298 | 0.9511 | 51.2501 | 137.6960 |
| FOODS_3_124_CA_3_validation | I4 | trpo:vpg | 0.7446 | 0.0801 | 1.5857 | 0.0000 | 26.3140 | 140.6676 |
| HOBBIES_1_371_WI_1_validation | I4 | trpo:trpo | 1.0000 | 0.0120 | 0.0299 | 0.9185 | 44.8623 | 117.9919 |
| HOBBIES_1_371_WI_1_validation | I4 | trpo:vpg | 0.7500 | 0.0637 | 1.4053 | 0.0000 | 25.6757 | 126.4451 |
| TECHM | T3 | trpo:trpo | 1.0000 | 0.0130 | 0.0298 | 0.8533 | 104.4423 | 631.4921 |
| TECHM | T3 | trpo:vpg | 0.8533 | 0.0005 | 0.0073 | 0.0000 | 48.4172 | 620.2030 |

## TRPO update diagnostics

| key | policy | corr_surrogate_vs_realised | cos_CG_vs_exact | cg_residual | vf_r2 | frac_accepted |
|---|---|---|---|---|---|---|
| FOODS_2_128_WI_2_validation | I4 | -0.1242 | nan | 0.0213 | 0.4567 | 1.0000 |
| FOODS_3_124_CA_3_validation | I4 | -0.0591 | nan | 0.0205 | 0.8132 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I4 | 0.0965 | nan | 0.0271 | 0.4623 | 1.0000 |
| TECHM | T3 | 0.0128 | nan | 0.0802 | 0.0389 | 1.0000 |

## TRPO vs ablations (paired by seed, Wilcoxon, Holm-adjusted over all rows)

| key | policy | vs | diff_trpo_minus | p_wilcoxon | cliffs_delta | p_holm |
|---|---|---|---|---|---|---|
| FOODS_2_128_WI_2_validation | I4 | trpo:vpg | -0.1476 | 0.5000 | -1.0000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I4 | trpo:vpg | 0.0273 | 1.0000 | 0.0000 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I4 | trpo:vpg | 0.0313 | 1.0000 | 0.0000 | 1.0000 |
| TECHM | T3 | trpo:vpg | -0.1610 | 0.5000 | -0.5000 | 1.0000 |
