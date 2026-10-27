# TRPO vs previous methods (neural_quick mode)

Baselines are **re-used** from `outputs/neural_full` of this folder (same seeds, same theta0). TRPO counts pilot + training batches against the same budget; validation/diagnostic rollouts are excluded for all methods. Tuning effort differs: TRPO tunes delta x lambda (12 configs) and VPG lr (3); REINFORCE tuned lr x B x entropy (18).

Previous results unchanged during comparison (hash check of outputs/full): **True**

## Summary (validation return, deterministic policy, last checkpoint)

| key | policy | method | n | final | ci_lo | ci_hi | eps95_median | reached95 |
|---|---|---|---|---|---|---|---|---|
| FOODS_2_128_WI_2_validation | I4 | cm:random | 10 | 0.4257 | 0.3498 | 0.4813 | nan | 0.0000 |
| FOODS_2_128_WI_2_validation | I4 | cm:spsa | 10 | 0.3735 | 0.2146 | 0.4649 | 12288.0000 | 0.1000 |
| FOODS_2_128_WI_2_validation | I4 | reinforce:loo_time | 10 | -5.0550 | -15.4926 | 0.4240 | 6944.0000 | 0.6000 |
| FOODS_2_128_WI_2_validation | I4 | reinforce:none | 10 | -5.7351 | -16.1421 | 0.0391 | 8928.0000 | 0.4000 |
| FOODS_2_128_WI_2_validation | I4 | trpo:trpo | 2 | 0.6189 | 0.6058 | 0.6320 | nan | 0.0000 |
| FOODS_2_128_WI_2_validation | I4 | trpo:vpg | 2 | 0.7665 | 0.7490 | 0.7840 | 544.0000 | 0.5000 |
| FOODS_3_124_CA_3_validation | I4 | cm:random | 10 | 2.6025 | 2.5225 | 2.6819 | 25600.0000 | 0.3000 |
| FOODS_3_124_CA_3_validation | I4 | cm:spsa | 10 | 2.2086 | 1.6975 | 2.5978 | 6144.0000 | 0.1000 |
| FOODS_3_124_CA_3_validation | I4 | reinforce:loo_time | 10 | 2.8859 | 2.8230 | 2.9471 | 3968.0000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I4 | reinforce:none | 10 | -7.9700 | -28.5775 | 2.6359 | 3968.0000 | 0.8000 |
| FOODS_3_124_CA_3_validation | I4 | trpo:trpo | 2 | 2.8047 | 2.6989 | 2.9104 | 1056.0000 | 0.5000 |
| FOODS_3_124_CA_3_validation | I4 | trpo:vpg | 2 | 2.7774 | 2.7160 | 2.8388 | 736.0000 | 0.5000 |
| HOBBIES_1_371_WI_1_validation | I4 | cm:random | 10 | 2.8571 | 2.7771 | 2.9346 | nan | 0.0000 |
| HOBBIES_1_371_WI_1_validation | I4 | cm:spsa | 10 | 2.8253 | 2.6956 | 2.9471 | 2560.0000 | 0.2000 |
| HOBBIES_1_371_WI_1_validation | I4 | reinforce:loo_time | 10 | 3.2390 | 3.1985 | 3.2736 | 9920.0000 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I4 | reinforce:none | 10 | 2.9303 | 2.5096 | 3.1701 | 9920.0000 | 0.8000 |
| HOBBIES_1_371_WI_1_validation | I4 | trpo:trpo | 2 | 2.9631 | 2.7093 | 3.2168 | 2720.0000 | 0.5000 |
| HOBBIES_1_371_WI_1_validation | I4 | trpo:vpg | 2 | 2.9318 | 2.9111 | 2.9525 | nan | 0.0000 |
| TECHM | T3 | cm:random | 10 | 0.7579 | 0.6956 | 0.8277 | 4096.0000 | 1.0000 |
| TECHM | T3 | cm:spsa | 10 | 0.7424 | 0.7076 | 0.7755 | 25600.0000 | 0.3000 |
| TECHM | T3 | reinforce:loo_time | 10 | 0.8291 | 0.7247 | 0.9313 | 4464.0000 | 1.0000 |
| TECHM | T3 | reinforce:none | 10 | 0.8048 | 0.7211 | 0.8880 | 4960.0000 | 1.0000 |
| TECHM | T3 | trpo:trpo | 2 | 0.3676 | 0.0339 | 0.7014 | 352.0000 | 0.5000 |
| TECHM | T3 | trpo:vpg | 2 | 0.5286 | 0.2459 | 0.8112 | 2336.0000 | 0.5000 |

## TRPO vs each method (paired by seed, Wilcoxon, Holm over all rows)

|  |
|

## Verdict counts

