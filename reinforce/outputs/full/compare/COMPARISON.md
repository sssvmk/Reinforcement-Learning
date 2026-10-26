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
| inventory | FOODS_3_124_CA_3_validation | I1 | reinforce:loo_time | spsa | 2.8724 | 2.8844 | 757.3654 | 758.5631 | 0.0002 | 0.4727 | -0.2000 | True | 0.4826 |
| inventory | FOODS_3_124_CA_3_validation | I1 | reinforce:none | spsa | 2.8741 | 2.8844 | 754.5057 | 758.5631 | 0.1405 | 0.0028 | -0.8000 | True | 0.0198 |
| inventory | FOODS_3_124_CA_3_validation | I2 | reinforce:loo_time | spsa | 2.9840 | 3.1065 | 780.6954 | 808.7173 | 0.0002 | 0.0008 | -0.9000 | False | 0.0069 |
| inventory | FOODS_3_124_CA_3_validation | I2 | reinforce:none | spsa | 2.9004 | 3.1065 | 767.8543 | 808.7173 | 0.0002 | 0.0003 | -0.9600 | False | 0.0033 |
| inventory | FOODS_3_124_CA_3_validation | I3 | reinforce:loo_time | central_nocrn | 3.0502 | 2.9111 | 797.8463 | 766.9939 | 0.0010 | 0.0010 | 0.8800 | False | 0.0081 |
| inventory | FOODS_3_124_CA_3_validation | I3 | reinforce:none | central_nocrn | 3.0403 | 2.9111 | 795.8263 | 766.9939 | 0.0003 | 0.0006 | 0.9200 | False | 0.0047 |
| inventory | FOODS_2_128_WI_2_validation | I1 | reinforce:loo_time | forward | 0.7460 | 0.8330 | 165.8783 | 170.4979 | 0.0002 | 0.0002 | -1.0000 | True | 0.0020 |
| inventory | FOODS_2_128_WI_2_validation | I1 | reinforce:none | forward | 0.7401 | 0.8330 | 166.9839 | 170.4979 | 0.0002 | 0.0028 | -0.8000 | True | 0.0198 |
| inventory | FOODS_2_128_WI_2_validation | I2 | reinforce:loo_time | regression | 0.7237 | 0.8319 | 164.4056 | 171.6441 | 0.0002 | 0.0091 | -0.7000 | False | 0.0364 |
| inventory | FOODS_2_128_WI_2_validation | I2 | reinforce:none | regression | 0.7457 | 0.8319 | 163.9962 | 171.6441 | 0.0002 | 0.0036 | -0.7800 | False | 0.0198 |
| inventory | FOODS_2_128_WI_2_validation | I3 | reinforce:loo_time | central | -0.0767 | 0.7790 | 82.8436 | 166.9415 | 0.3447 | 0.0017 | -0.8400 | False | 0.0119 |
| inventory | FOODS_2_128_WI_2_validation | I3 | reinforce:none | central | -0.0519 | 0.7790 | 86.9760 | 166.9415 | 0.3075 | 0.7337 | 0.1000 | False | 1.0000 |
| inventory | HOBBIES_1_371_WI_1_validation | I1 | reinforce:loo_time | forward | 3.3168 | 3.0422 | 100.0184 | 94.8806 | 0.0002 | 0.0036 | 0.7800 | True | 0.0181 |
| inventory | HOBBIES_1_371_WI_1_validation | I1 | reinforce:none | forward | 3.3095 | 3.0422 | 101.5030 | 94.8806 | 0.0002 | 0.0003 | 0.9600 | True | 0.0033 |
| inventory | HOBBIES_1_371_WI_1_validation | I2 | reinforce:loo_time | spsa_nocrn | 3.2135 | 3.3830 | 96.0810 | 101.9020 | 0.0028 | 0.2413 | -0.3200 | False | 0.4826 |
| inventory | HOBBIES_1_371_WI_1_validation | I2 | reinforce:none | spsa_nocrn | 3.2422 | 3.3830 | 98.9337 | 101.9020 | 0.0091 | 0.2123 | -0.3400 | False | 0.8492 |
| inventory | HOBBIES_1_371_WI_1_validation | I3 | reinforce:loo_time | spsa_nocrn | 3.2361 | 3.2839 | 104.4616 | 98.4598 | 0.9097 | 0.0028 | 0.8000 | False | 0.0170 |
| inventory | HOBBIES_1_371_WI_1_validation | I3 | reinforce:none | spsa_nocrn | 3.1930 | 3.2839 | 98.1303 | 98.4598 | 0.6776 | 0.8501 | 0.0600 | False | 1.0000 |
| trading | TECHM | T1 | reinforce:loo_time | spsa_nocrn | 0.9318 | 0.8536 | 0.8131 | 0.6891 | 0.1405 | 0.1041 | 0.4400 | True | 0.3123 |
| trading | TECHM | T1 | reinforce:none | spsa_nocrn | 0.9382 | 0.8536 | 0.7856 | 0.6891 | 0.0757 | 0.4727 | 0.2000 | True | 1.0000 |
| trading | TECHM | T2 | reinforce:loo_time | regression | 0.9555 | 0.6356 | 0.7559 | 0.9885 | 0.0002 | 0.0002 | -1.0000 | False | 0.0020 |
| trading | TECHM | T2 | reinforce:none | regression | 0.9647 | 0.6356 | 0.6158 | 0.9885 | 0.0002 | 0.0002 | -1.0000 | False | 0.0020 |

## Evidence A: FDPG's reported results vs ours (indicative)

| domain | key | policy | fdpg_method | fdpg_final_val | fdpg_rollouts_to_90 | reinforce_final_val | reinforce_best_mean_val | reinforce_rollouts_to_90 | caveat |
|---|---|---|---|---|---|---|---|---|---|
| inventory | FOODS_3_124_CA_3_validation | I1 | spsa | 2.8898 | 2065.5857 | 2.7925 | 2.8509 | 5760.0000 | normalisation differs (Q_max, norm rule): indicative only |
| inventory | FOODS_3_124_CA_3_validation | I2 | spsa | 3.1123 | 3236.2775 | 2.5529 | 2.8870 | 4960.0000 | normalisation differs (Q_max, norm rule): indicative only |
| inventory | FOODS_3_124_CA_3_validation | I3 | spsa | 2.8899 | 3236.2775 | 2.8782 | 3.0226 | 960.0000 | normalisation differs (Q_max, norm rule): indicative only |
| inventory | FOODS_2_128_WI_2_validation | I1 | forward | 0.8310 | 5546.8805 | 0.7211 | 0.7358 | 4960.0000 | normalisation differs (Q_max, norm rule): indicative only |
| inventory | FOODS_2_128_WI_2_validation | I2 | regression | 0.8298 | 10400.4556 | -5.0272 | 0.5111 | 0.0000 | normalisation differs (Q_max, norm rule): indicative only |
| inventory | FOODS_2_128_WI_2_validation | I3 | central | 0.7771 | 5546.8805 | -0.6464 | -0.1511 | 1920.0000 | normalisation differs (Q_max, norm rule): indicative only |
| inventory | HOBBIES_1_371_WI_1_validation | I1 | forward | 3.0481 | 1577.7627 | 3.1526 | 3.2301 | 992.0000 | normalisation differs (Q_max, norm rule): indicative only |
| inventory | HOBBIES_1_371_WI_1_validation | I2 | forward | 3.3616 | 769.1971 | -19.6598 | 2.6102 | 0.0000 | normalisation differs (Q_max, norm rule): indicative only |
| inventory | HOBBIES_1_371_WI_1_validation | I3 | spsa | 3.1917 | 2065.5857 | 2.9870 | 2.9888 | 6944.0000 | normalisation differs (Q_max, norm rule): indicative only |
| trading | TECHM | T1 | forward | 0.7681 | 4634.9792 | 0.8051 | 0.8244 | 8928.0000 | normalisation differs (Q_max, norm rule): indicative only |
| trading | TECHM | T2 | regression | 0.6369 | 3236.2775 | 0.7853 | 0.7988 | 29760.0000 | normalisation differs (Q_max, norm rule): indicative only |

Notes: seeds are not paired across projects (Mann-Whitney); R-I2/R-I3/R-T2 are not the same policy families as F-I2/F-I3/F-T2 (`same_family` False); FDPG used no Q_max or discount; tuning effort differs.
