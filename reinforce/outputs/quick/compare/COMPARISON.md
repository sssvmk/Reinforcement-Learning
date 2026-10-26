# REINFORCE vs FDPG

Hash check: 305 FDPG files, asserted unchanged after the run.

## Reproduction check (FDPG saved thetas re-run here, Q_max=None, FDPG windows; tolerance 2%)

| domain | key | policy | estimator | metric | fdpg_reported | reproduced_here | rel_diff | status |
|---|---|---|---|---|---|---|---|---|
| inventory | FOODS_3_124_CA_3_validation | I1 | spsa | mean test-window profit | 758.5631 | 758.5631 | 0.0000 | REPRODUCED |
| inventory | FOODS_3_124_CA_3_validation | I2 | spsa | mean test-window profit | 808.7173 | 808.7173 | 0.0000 | REPRODUCED |
| inventory | FOODS_3_124_CA_3_validation | I3 | spsa | mean test-window profit | 761.0738 | 761.0738 | -0.0000 | REPRODUCED |
| inventory | FOODS_2_128_WI_2_validation | I1 | forward | mean test-window profit | 170.4979 | 170.4979 | -0.0000 | REPRODUCED |
| inventory | FOODS_2_128_WI_2_validation | I2 | regression | mean test-window profit | 171.6441 | 171.6441 | -0.0000 | REPRODUCED |
| inventory | FOODS_2_128_WI_2_validation | I3 | central | mean test-window profit | 166.9415 | 166.9415 | -0.0000 | REPRODUCED |
| inventory | HOBBIES_1_371_WI_1_validation | I1 | forward | mean test-window profit | 94.8806 | 94.8806 | 0.0000 | REPRODUCED |
| inventory | HOBBIES_1_371_WI_1_validation | I2 | forward | mean test-window profit | 102.8691 | 102.8691 | 0.0000 | REPRODUCED |
| inventory | HOBBIES_1_371_WI_1_validation | I3 | spsa | mean test-window profit | 100.7105 | 100.7105 | 0.0000 | REPRODUCED |
| trading | TECHM | T1 | forward | test cumulative return | 0.7802 | 0.7802 | 0.0000 | REPRODUCED |
| trading | TECHM | T2 | regression | test cumulative return | 0.9885 | 0.9885 | 0.0000 | REPRODUCED |

## Evidence B: re-evaluated in this environment (same windows, Q_max, normalisation); REINFORCE vs FDPG-best-by-validation

| domain | key | policy | method | fdpg_estimator | val_mean_R | val_mean_F | test_mean_R | test_mean_F | p_val | p_test | cliffs_test | same_family | p_test_holm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| inventory | FOODS_3_124_CA_3_validation | I1 | reinforce:loo_time | spsa | 2.8962 | 2.8844 | 762.0167 | 758.5631 | 0.4685 | 0.0070 | 1.0000 | True | 0.0769 |
| inventory | FOODS_3_124_CA_3_validation | I1 | reinforce:none | spsa | 2.8484 | 2.8844 | 748.7443 | 758.5631 | 0.4685 | 0.4685 | -0.3333 | True | 1.0000 |
| inventory | FOODS_3_124_CA_3_validation | I2 | reinforce:loo_time | spsa | 2.9080 | 3.1065 | 776.6889 | 808.7173 | 0.0070 | 0.0140 | -0.9333 | False | 0.0839 |
| inventory | FOODS_3_124_CA_3_validation | I2 | reinforce:none | spsa | 2.9751 | 3.1065 | 780.6488 | 808.7173 | 0.0070 | 0.0140 | -0.9333 | False | 0.0839 |
| inventory | FOODS_3_124_CA_3_validation | I3 | reinforce:loo_time | central_nocrn | 3.0142 | 2.9111 | 796.8574 | 766.9939 | 0.0070 | 0.0070 | 1.0000 | False | 0.0769 |
| inventory | FOODS_3_124_CA_3_validation | I3 | reinforce:none | central_nocrn | -0.1810 | 2.9111 | 290.4129 | 766.9939 | 0.4685 | 0.4685 | 0.3333 | False | 1.0000 |
| inventory | FOODS_2_128_WI_2_validation | I1 | reinforce:loo_time | forward | 0.7346 | 0.8330 | 164.6307 | 170.4979 | 0.0070 | 0.0070 | -1.0000 | True | 0.0769 |
| inventory | FOODS_2_128_WI_2_validation | I1 | reinforce:none | forward | 0.7333 | 0.8330 | 164.0702 | 170.4979 | 0.0070 | 0.0070 | -1.0000 | True | 0.0769 |
| inventory | FOODS_2_128_WI_2_validation | I2 | reinforce:loo_time | regression | 0.6934 | 0.8319 | 160.1945 | 171.6441 | 0.0070 | 0.0140 | -0.9333 | False | 0.0839 |
| inventory | FOODS_2_128_WI_2_validation | I2 | reinforce:none | regression | 0.6783 | 0.8319 | 158.3031 | 171.6441 | 0.0070 | 0.0070 | -1.0000 | False | 0.0769 |
| inventory | FOODS_2_128_WI_2_validation | I3 | reinforce:loo_time | central | -1.0672 | 0.7790 | 16.2927 | 166.9415 | 0.0070 | 0.0070 | -1.0000 | False | 0.0769 |
| inventory | FOODS_2_128_WI_2_validation | I3 | reinforce:none | central | -1.9664 | 0.7790 | -55.9874 | 166.9415 | 0.0070 | 0.0070 | -1.0000 | False | 0.0769 |
| inventory | HOBBIES_1_371_WI_1_validation | I1 | reinforce:loo_time | forward | 3.2420 | 3.0422 | 104.9919 | 94.8806 | 0.0070 | 0.0070 | 1.0000 | True | 0.0769 |
| inventory | HOBBIES_1_371_WI_1_validation | I1 | reinforce:none | forward | 3.2820 | 3.0422 | 97.2821 | 94.8806 | 0.0070 | 0.0070 | 1.0000 | True | 0.0769 |
| inventory | HOBBIES_1_371_WI_1_validation | I2 | reinforce:loo_time | spsa_nocrn | 3.1628 | 3.3830 | 100.1456 | 101.9020 | 0.0070 | 0.8112 | -0.1333 | False | 1.0000 |
| inventory | HOBBIES_1_371_WI_1_validation | I2 | reinforce:none | spsa_nocrn | 3.2592 | 3.3830 | 96.2719 | 101.9020 | 0.0769 | 0.3706 | -0.4000 | False | 1.0000 |
| inventory | HOBBIES_1_371_WI_1_validation | I3 | reinforce:loo_time | spsa_nocrn | 3.0144 | 3.2839 | 94.1478 | 98.4598 | 0.0280 | 0.0490 | -0.8000 | False | 0.1958 |
| inventory | HOBBIES_1_371_WI_1_validation | I3 | reinforce:none | spsa_nocrn | 1.6824 | 3.2839 | 89.2130 | 98.4598 | 0.0070 | 0.2168 | -0.5333 | False | 1.0000 |
| trading | TECHM | T1 | reinforce:loo_time | spsa_nocrn | 0.7397 | 0.8536 | 0.5557 | 0.6891 | 0.6923 | 1.0000 | 0.0000 | True | 1.0000 |
| trading | TECHM | T1 | reinforce:none | spsa_nocrn | 0.7187 | 0.8536 | 0.7380 | 0.6891 | 0.3706 | 1.0000 | 0.0000 | True | 1.0000 |
| trading | TECHM | T2 | reinforce:loo_time | regression | 0.8446 | 0.6356 | 0.7310 | 0.9885 | 0.0070 | 0.4685 | -0.3333 | False | 1.0000 |
| trading | TECHM | T2 | reinforce:none | regression | 0.9120 | 0.6356 | 0.7342 | 0.9885 | 0.0070 | 0.0070 | -1.0000 | False | 0.0769 |

## Evidence A: FDPG's reported results vs ours (indicative)

| domain | key | policy | fdpg_method | fdpg_final_val | fdpg_rollouts_to_90 | reinforce_final_val | reinforce_best_mean_val | reinforce_rollouts_to_90 | caveat |
|---|---|---|---|---|---|---|---|---|---|
| inventory | FOODS_3_124_CA_3_validation | I1 | spsa | 2.8898 | 2065.5857 | 2.8507 | 2.8882 | 1280.0000 | normalisation differs (Q_max, norm rule): indicative only |
| inventory | FOODS_3_124_CA_3_validation | I2 | spsa | 3.1123 | 3236.2775 | 2.8085 | 2.8951 | 2048.0000 | normalisation differs (Q_max, norm rule): indicative only |
| inventory | FOODS_3_124_CA_3_validation | I3 | spsa | 2.8899 | 3236.2775 | 2.7922 | 2.8542 | 256.0000 | normalisation differs (Q_max, norm rule): indicative only |
| inventory | FOODS_2_128_WI_2_validation | I1 | forward | 0.8310 | 5546.8805 | 0.6272 | 0.7033 | 640.0000 | normalisation differs (Q_max, norm rule): indicative only |
| inventory | FOODS_2_128_WI_2_validation | I2 | regression | 0.8298 | 10400.4556 | -17.1535 | 0.6597 | 0.0000 | normalisation differs (Q_max, norm rule): indicative only |
| inventory | FOODS_2_128_WI_2_validation | I3 | central | 0.7771 | 5546.8805 | -3.1381 | -1.0896 | 512.0000 | normalisation differs (Q_max, norm rule): indicative only |
| inventory | HOBBIES_1_371_WI_1_validation | I1 | forward | 3.0481 | 1577.7627 | 3.2201 | 3.2201 | 768.0000 | normalisation differs (Q_max, norm rule): indicative only |
| inventory | HOBBIES_1_371_WI_1_validation | I2 | forward | 3.3616 | 769.1971 | 2.9392 | 3.0410 | 1664.0000 | normalisation differs (Q_max, norm rule): indicative only |
| inventory | HOBBIES_1_371_WI_1_validation | I3 | spsa | 3.1917 | 2065.5857 | -6.4446 | 2.6063 | 128.0000 | normalisation differs (Q_max, norm rule): indicative only |
| trading | TECHM | T1 | forward | 0.7681 | 4634.9792 | 0.6545 | 0.6641 | 1792.0000 | normalisation differs (Q_max, norm rule): indicative only |
| trading | TECHM | T2 | regression | 0.6369 | 3236.2775 | 0.7506 | 0.7517 | 4864.0000 | normalisation differs (Q_max, norm rule): indicative only |

Notes: seeds are not paired across projects (Mann-Whitney); R-I2/R-I3/R-T2 are not the same policy families as F-I2/F-I3/F-T2 (`same_family` False); FDPG used no Q_max or discount; tuning effort differs.
