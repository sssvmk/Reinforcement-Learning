# TRPO vs previous methods (neural_full mode)

Baselines are **re-used** from `outputs/neural_full` of this folder (same seeds, same theta0). TRPO counts pilot + training batches against the same budget; validation/diagnostic rollouts are excluded for all methods. Tuning effort differs: TRPO tunes delta x lambda (12 configs) and VPG lr (3); REINFORCE tuned lr x B x entropy (18).

Previous results unchanged during comparison (hash check of outputs/full): **True**

## Summary (validation return, deterministic policy, last checkpoint)

| key | policy | method | n | final | ci_lo | ci_hi | eps95_median | reached95 |
|---|---|---|---|---|---|---|---|---|
| FOODS_2_128_WI_2_validation | I4 | cm:random | 10 | 0.4257 | 0.3498 | 0.4813 | nan | 0.0000 |
| FOODS_2_128_WI_2_validation | I4 | cm:spsa | 10 | 0.3735 | 0.2146 | 0.4649 | 12288.0000 | 0.1000 |
| FOODS_2_128_WI_2_validation | I4 | reinforce:loo_time | 10 | -5.0550 | -15.4926 | 0.4240 | 6944.0000 | 0.6000 |
| FOODS_2_128_WI_2_validation | I4 | reinforce:none | 10 | -5.7351 | -16.1421 | 0.0391 | 8928.0000 | 0.4000 |
| FOODS_2_128_WI_2_validation | I4 | trpo:trpo | 10 | 0.6535 | 0.6118 | 0.6938 | 21856.0000 | 0.5000 |
| FOODS_2_128_WI_2_validation | I4 | trpo:vpg | 10 | 0.7663 | 0.7370 | 0.7939 | 3504.0000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I4 | cm:random | 10 | 2.6025 | 2.5225 | 2.6819 | 25600.0000 | 0.3000 |
| FOODS_3_124_CA_3_validation | I4 | cm:spsa | 10 | 2.2086 | 1.6975 | 2.5978 | 6144.0000 | 0.1000 |
| FOODS_3_124_CA_3_validation | I4 | reinforce:loo_time | 10 | 2.8859 | 2.8230 | 2.9471 | 3968.0000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I4 | reinforce:none | 10 | -7.9700 | -28.5775 | 2.6359 | 3968.0000 | 0.8000 |
| FOODS_3_124_CA_3_validation | I4 | trpo:trpo | 10 | 2.6418 | 2.5565 | 2.7142 | 8464.0000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I4 | trpo:vpg | 10 | 2.8850 | 2.7416 | 2.9826 | 2016.0000 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I4 | cm:random | 10 | 2.8571 | 2.7771 | 2.9346 | nan | 0.0000 |
| HOBBIES_1_371_WI_1_validation | I4 | cm:spsa | 10 | 2.8253 | 2.6956 | 2.9471 | 2560.0000 | 0.2000 |
| HOBBIES_1_371_WI_1_validation | I4 | reinforce:loo_time | 10 | 3.2390 | 3.1985 | 3.2736 | 9920.0000 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I4 | reinforce:none | 10 | 2.9303 | 2.5096 | 3.1701 | 9920.0000 | 0.8000 |
| HOBBIES_1_371_WI_1_validation | I4 | trpo:trpo | 10 | 3.1635 | 3.0947 | 3.2216 | 4992.0000 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I4 | trpo:vpg | 10 | 3.1520 | 3.0376 | 3.2511 | 3008.0000 | 1.0000 |
| TECHM | T3 | cm:random | 10 | 0.7579 | 0.6956 | 0.8277 | 4096.0000 | 1.0000 |
| TECHM | T3 | cm:spsa | 10 | 0.7424 | 0.7076 | 0.7755 | 25600.0000 | 0.3000 |
| TECHM | T3 | reinforce:loo_time | 10 | 0.8291 | 0.7247 | 0.9313 | 4464.0000 | 1.0000 |
| TECHM | T3 | reinforce:none | 10 | 0.8048 | 0.7211 | 0.8880 | 4960.0000 | 1.0000 |
| TECHM | T3 | trpo:trpo | 10 | 0.4466 | 0.3337 | 0.5686 | 20368.0000 | 0.2000 |
| TECHM | T3 | trpo:vpg | 10 | 0.7612 | 0.6551 | 0.8606 | 2512.0000 | 1.0000 |

## TRPO vs each method (paired by seed, Wilcoxon, Holm over all rows)

| key | policy | vs | n_pairs | diff_trpo_minus | p_wilcoxon_paired | p_mannwhitney | cliffs_delta | p_holm | verdict |
|---|---|---|---|---|---|---|---|---|---|
| FOODS_2_128_WI_2_validation | I4 | cm:random | 10 | 0.2278 | 0.0020 | 0.0002 | 1.0000 | 0.0391 | TRPO better |
| FOODS_2_128_WI_2_validation | I4 | cm:spsa | 10 | 0.2800 | 0.0020 | 0.0002 | 1.0000 | 0.0391 | TRPO better |
| FOODS_2_128_WI_2_validation | I4 | reinforce:loo_time | 10 | 5.7084 | 0.0488 | 0.0757 | 0.4800 | 0.4395 | no significant difference |
| FOODS_2_128_WI_2_validation | I4 | reinforce:none | 10 | 6.3885 | 0.0488 | 0.1405 | 0.4000 | 0.4395 | no significant difference |
| FOODS_2_128_WI_2_validation | I4 | trpo:vpg | 10 | -0.1128 | 0.0020 | 0.0013 | -0.8600 | 0.0391 | TRPO worse |
| FOODS_3_124_CA_3_validation | I4 | cm:random | 10 | 0.0393 | 0.3750 | 0.5205 | 0.1800 | 1.0000 | no significant difference |
| FOODS_3_124_CA_3_validation | I4 | cm:spsa | 10 | 0.4332 | 0.1602 | 0.0757 | 0.4800 | 0.8008 | no significant difference |
| FOODS_3_124_CA_3_validation | I4 | reinforce:loo_time | 10 | -0.2441 | 0.0020 | 0.0008 | -0.9000 | 0.0391 | TRPO worse |
| FOODS_3_124_CA_3_validation | I4 | reinforce:none | 10 | 10.6118 | 0.3750 | 0.6776 | 0.1200 | 1.0000 | no significant difference |
| FOODS_3_124_CA_3_validation | I4 | trpo:vpg | 10 | -0.2432 | 0.0645 | 0.0028 | -0.8000 | 0.4512 | no significant difference |
| HOBBIES_1_371_WI_1_validation | I4 | cm:random | 10 | 0.3064 | 0.0020 | 0.0006 | 0.9200 | 0.0391 | TRPO better |
| HOBBIES_1_371_WI_1_validation | I4 | cm:spsa | 10 | 0.3383 | 0.0039 | 0.0008 | 0.9000 | 0.0547 | no significant difference |
| HOBBIES_1_371_WI_1_validation | I4 | reinforce:loo_time | 10 | -0.0755 | 0.0645 | 0.0757 | -0.4800 | 0.4512 | no significant difference |
| HOBBIES_1_371_WI_1_validation | I4 | reinforce:none | 10 | 0.2332 | 0.3750 | 0.2413 | 0.3200 | 1.0000 | no significant difference |
| HOBBIES_1_371_WI_1_validation | I4 | trpo:vpg | 10 | 0.0115 | 1.0000 | 0.6776 | -0.1200 | 1.0000 | no significant difference |
| TECHM | T3 | cm:random | 10 | -0.3113 | 0.0039 | 0.0073 | -0.7200 | 0.0547 | no significant difference |
| TECHM | T3 | cm:spsa | 10 | -0.2958 | 0.0098 | 0.0113 | -0.6800 | 0.1074 | no significant difference |
| TECHM | T3 | reinforce:loo_time | 10 | -0.3825 | 0.0059 | 0.0022 | -0.8200 | 0.0703 | no significant difference |
| TECHM | T3 | reinforce:none | 10 | -0.3582 | 0.0020 | 0.0028 | -0.8000 | 0.0391 | TRPO worse |
| TECHM | T3 | trpo:vpg | 10 | -0.3146 | 0.0098 | 0.0046 | -0.7600 | 0.1074 | no significant difference |

## Verdict counts

| vs | TRPO better | TRPO worse | no significant difference |
|---|---|---|---|
| cm:random | 2 | 0 | 2 |
| cm:spsa | 1 | 0 | 3 |
| reinforce:loo_time | 0 | 1 | 3 |
| reinforce:none | 0 | 1 | 3 |
| trpo:vpg | 0 | 1 | 3 |