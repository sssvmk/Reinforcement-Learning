# Algorithm-performance summary (full)
analysis_plan.yaml sha256[:16] = faf34b6d2602a63c

## inventory__FOODS_2_128_WI_2_validation__I1
tuned hyperparameters (validation only): eps=0.4, a=0.5; budget=40000 rollouts; seeds=10

### Final validation return, sample efficiency, runtime
| method | final_val_mean | final_val_sd | final_val_ci_lo | final_val_ci_hi | rollouts_to_90pct_mean | rollouts_to_95pct_mean | reached_95pct_share | val@25%budget | val@100%budget | runtime_per_iter_ms |
|---|---|---|---|---|---|---|---|---|---|---|
| central | 0.831 | 0.000 | 0.831 | 0.831 | 5043.200 | 7244.800 | 1.000 | 0.820 | 0.777 | 13.205 |
| forward | 0.831 | 0.000 | 0.831 | 0.831 | 5299.200 | 7680.000 | 1.000 | 0.818 | 0.781 | 13.220 |
| gaussian | 0.831 | 0.000 | 0.831 | 0.831 | 10803.200 | 15462.400 | 1.000 | 0.780 | 0.814 | 22.448 |
| regression | 0.831 | 0.000 | 0.831 | 0.831 | 7488.000 | 10752.000 | 1.000 | 0.806 | 0.800 | 17.616 |
| spsa | 0.821 | 0.012 | 0.813 | 0.828 | 20275.200 | 39872.000 | 0.700 | 0.718 | 0.819 | 39.457 |
| central_nocrn | 0.831 | 0.000 | 0.831 | 0.831 | 5273.600 | 7347.200 | 1.000 | 0.819 | 0.772 | 13.602 |
| grid | 0.552 | 0.019 | 0.543 | 0.566 | 20480.000 | 20480.000 | 1.000 | 0.372 | 0.552 | 4.933 |
| neldermead | 0.585 | 0.269 | 0.412 | 0.725 | 64147.200 | 64147.200 | 0.200 | 0.585 | 0.585 | 4.594 |
| random | 0.553 | 0.019 | 0.544 | 0.566 | 64032.000 | 72006.400 | 0.100 | 0.560 | 0.553 | 4.634 |
| spsa_nocrn | 0.820 | 0.012 | 0.813 | 0.827 | 20992.000 | 40384.000 | 0.700 | 0.717 | 0.819 | 41.435 |

### Stability (share of seeds)
| method | converged | unconverged | bound | stall | diverge |
|---|---|---|---|---|---|
| central | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| forward | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| gaussian | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| regression | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| spsa | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| central_nocrn | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| grid | 0.000 | 0.000 | 0.500 | 0.200 | 0.300 |
| neldermead | 0.500 | 0.000 | 0.000 | 0.500 | 0.000 |
| random | 0.500 | 0.000 | 0.000 | 0.200 | 0.300 |
| spsa_nocrn | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |

### Gradient quality vs reference (rows: point x estimator)
| point | estimator | cosine | rel_l2 | sign_agree | bias_l2 | variance |
|---|---|---|---|---|---|---|
| start | forward | 1.000 | 0.088 | 1.000 | 0.041 | 0.007 |
| start | central | 1.000 | 0.082 | 1.000 | 0.002 | 0.007 |
| start | spsa | 1.000 | 0.042 | 1.000 | 0.015 | 0.002 |
| start | regression | 1.000 | 0.086 | 1.000 | 0.003 | 0.009 |
| start | gaussian | 1.000 | 0.502 | 1.000 | 0.158 | 0.295 |
| start | dense_grid_derivative | nan | nan | nan | 0.009 | 0.000 |
| mid | forward | 1.000 | 0.118 | 1.000 | 0.020 | 0.002 |
| mid | central | 1.000 | 0.113 | 1.000 | 0.001 | 0.002 |
| mid | spsa | 1.000 | 0.060 | 1.000 | 0.002 | 0.001 |
| mid | regression | 1.000 | 0.116 | 1.000 | 0.004 | 0.002 |
| mid | gaussian | 1.000 | 0.532 | 1.000 | 0.051 | 0.041 |
| mid | dense_grid_derivative | nan | nan | nan | 0.000 | 0.000 |
| final | forward | 1.000 | 0.114 | 1.000 | 0.035 | 0.002 |
| final | central | 1.000 | 0.093 | 1.000 | 0.000 | 0.002 |
| final | spsa | 1.000 | 0.053 | 1.000 | 0.004 | 0.001 |
| final | regression | 1.000 | 0.106 | 1.000 | 0.002 | 0.003 |
| final | gaussian | 1.000 | 0.514 | 1.000 | 0.072 | 0.065 |
| final | dense_grid_derivative | nan | nan | nan | 0.000 | 0.000 |

### Common random numbers
| estimator | crn | variance | cosine | bias_l2 |
|---|---|---|---|---|
| forward | True | 0.011 | 1.000 | 0.045 |
| forward | False | 0.156 | 1.000 | 0.011 |
| central | True | 0.011 | 1.000 | 0.004 |
| central | False | 0.046 | 1.000 | 0.017 |
| spsa | True | 0.002 | 1.000 | 0.000 |
| spsa | False | 0.010 | 1.000 | 0.013 |
| regression | True | 0.013 | 1.000 | 0.010 |
| regression | False | 0.295 | 0.867 | 0.074 |
| gaussian | True | 1.002 | 1.000 | 0.073 |
| gaussian | False | 0.648 | 1.000 | 0.071 |

| estimator | crn | final_val_mean | final_val_sd | n |
|---|---|---|---|---|
| central | True | 0.831 | 0.000 | 10 |
| central | False | 0.831 | 0.000 | 10 |
| spsa | True | 0.821 | 0.012 | 10 |
| spsa | False | 0.820 | 0.011 | 10 |

### Optimality gap vs exhaustive grid on validation paths
| method | gap_mean | gap_sd | grid_best_val |
|---|---|---|---|
| central | 0.000 | 0.000 | 0.831 |
| central_nocrn | 0.000 | 0.000 | 0.831 |
| forward | 0.000 | 0.000 | 0.831 |
| gaussian | 0.000 | 0.000 | 0.831 |
| grid | 0.279 | 0.019 | 0.831 |
| neldermead | 0.246 | 0.269 | 0.831 |
| random | 0.278 | 0.019 | 0.831 |
| regression | 0.000 | 0.000 | 0.831 |
| spsa | 0.010 | 0.012 | 0.831 |
| spsa_nocrn | 0.011 | 0.012 | 0.831 |

### Paired Wilcoxon tests (Holm; 52 comparisons in total, 24 significant at 0.05)
| outcome | A | B | mean_diff_A_minus_B | p_holm | rank_biserial | cohens_dz |
|---|---|---|---|---|---|---|
| final_validation_return | forward | random | 0.278 | 0.041 | 1.000 | 14.668 |
| final_validation_return | forward | neldermead | 0.246 | 0.041 | 1.000 | 0.916 |
| final_validation_return | central | random | 0.278 | 0.041 | 1.000 | 14.666 |
| final_validation_return | central | neldermead | 0.246 | 0.041 | 1.000 | 0.916 |
| final_validation_return | spsa | random | 0.268 | 0.041 | 1.000 | 14.745 |
| final_validation_return | spsa | neldermead | 0.237 | 0.041 | 1.000 | 0.915 |
| final_validation_return | regression | random | 0.278 | 0.041 | 1.000 | 14.669 |
| final_validation_return | regression | neldermead | 0.246 | 0.041 | 1.000 | 0.916 |
| final_validation_return | gaussian | random | 0.277 | 0.041 | 1.000 | 14.700 |
| final_validation_return | gaussian | neldermead | 0.246 | 0.041 | 1.000 | 0.916 |
| rollouts_to_95pct | forward | central | 435.200 | 0.047 | 1.000 | 1.605 |
| rollouts_to_95pct | forward | spsa | -32192.000 | 0.041 | -1.000 | -1.277 |
| rollouts_to_95pct | forward | regression | -3072.000 | 0.041 | -1.000 | -2.270 |
| rollouts_to_95pct | forward | gaussian | -7782.400 | 0.041 | -1.000 | -1.993 |
| rollouts_to_95pct | forward | random | -64326.400 | 0.047 | -0.964 | -2.698 |
| rollouts_to_95pct | central | spsa | -32627.200 | 0.041 | -1.000 | -1.285 |
| rollouts_to_95pct | central | regression | -3507.200 | 0.041 | -1.000 | -2.201 |
| rollouts_to_95pct | central | gaussian | -8217.600 | 0.041 | -1.000 | -1.990 |
| rollouts_to_95pct | central | random | -64761.600 | 0.047 | -0.964 | -2.702 |
| rollouts_to_95pct | spsa | regression | 29120.000 | 0.041 | 1.000 | 1.216 |
| rollouts_to_95pct | spsa | gaussian | 24409.600 | 0.041 | 1.000 | 1.093 |
| rollouts_to_95pct | regression | gaussian | -4710.400 | 0.041 | -1.000 | -1.662 |
| rollouts_to_95pct | regression | random | -61254.400 | 0.047 | -0.964 | -2.607 |
| rollouts_to_95pct | gaussian | random | -56544.000 | 0.047 | -0.964 | -2.519 |

### Sensitivity (see CSV; plots saved)
| param | value | estimator | bias_l2 | variance | cosine | final_val_mean | final_val_sd |
|---|---|---|---|---|---|---|---|
| eps | 0.010 | forward | 0.024 | 0.006 | 1.000 |  |  |
| eps | 0.030 | forward | 0.023 | 0.006 | 1.000 |  |  |
| eps | 0.100 | forward | 0.015 | 0.006 | 1.000 |  |  |
| eps | 0.300 | forward | 0.012 | 0.005 | 1.000 |  |  |
| eps | 1.000 | forward | 0.082 | 0.005 | 1.000 |  |  |
| eps | 0.010 | central | 0.024 | 0.006 | 1.000 |  |  |
| eps | 0.030 | central | 0.016 | 0.006 | 1.000 |  |  |
| eps | 0.100 | central | 0.024 | 0.006 | 1.000 |  |  |
| eps | 0.300 | central | 0.021 | 0.006 | 1.000 |  |  |
| eps | 1.000 | central | 0.016 | 0.006 | 1.000 |  |  |
| eps | 0.010 | spsa | 0.008 | 0.001 | 1.000 |  |  |
| eps | 0.030 | spsa | 0.001 | 0.001 | 1.000 |  |  |
| eps | 0.100 | spsa | 0.009 | 0.001 | 1.000 |  |  |
| eps | 0.300 | spsa | 0.005 | 0.001 | 1.000 |  |  |
| eps | 1.000 | spsa | 0.002 | 0.001 | 1.000 |  |  |
| N_paths | 8.000 | central | 0.043 | 0.049 | 1.000 |  |  |
| N_paths | 16.000 | central | 0.063 | 0.022 | 1.000 |  |  |
| N_paths | 32.000 | central | 0.035 | 0.012 | 1.000 |  |  |
| N_paths | 64.000 | central | 0.013 | 0.008 | 1.000 |  |  |
| N_paths | 128.000 | central | 0.004 | 0.003 | 1.000 |  |  |
| N_paths | 8.000 | spsa | 0.042 | 0.011 | 1.000 |  |  |
| N_paths | 16.000 | spsa | 0.002 | 0.008 | 1.000 |  |  |
| N_paths | 32.000 | spsa | 0.003 | 0.003 | 1.000 |  |  |
| N_paths | 64.000 | spsa | 0.002 | 0.002 | 1.000 |  |  |
| N_paths | 128.000 | spsa | 0.000 | 0.001 | 1.000 |  |  |
| N_paths | 8.000 | regression | 0.045 | 0.044 | 1.000 |  |  |
| N_paths | 16.000 | regression | 0.056 | 0.026 | 1.000 |  |  |
| N_paths | 32.000 | regression | 0.035 | 0.012 | 1.000 |  |  |
| N_paths | 64.000 | regression | 0.022 | 0.008 | 1.000 |  |  |
| N_paths | 128.000 | regression | 0.004 | 0.004 | 1.000 |  |  |
| spsa_m | 1.000 | spsa | 0.007 | 0.008 | 1.000 |  |  |
| spsa_m | 2.000 | spsa | 0.005 | 0.005 | 1.000 |  |  |
| spsa_m | 4.000 | spsa | 0.011 | 0.002 | 1.000 |  |  |
| spsa_m | 8.000 | spsa | 0.009 | 0.001 | 1.000 |  |  |
| spsa_m | 16.000 | spsa | 0.005 | 0.001 | 1.000 |  |  |
| regression_K | 1.000 | regression | 0.022 | 0.010 | 1.000 |  |  |
| regression_K | 2.000 | regression | 0.023 | 0.012 | 1.000 |  |  |
| regression_K | 4.000 | regression | 0.026 | 0.011 | 1.000 |  |  |
| regression_K | 8.000 | regression | 0.015 | 0.010 | 1.000 |  |  |
| step_size_a_mult | 0.250 | central |  |  |  | 0.604 | 0.109 |
| step_size_a_mult | 1.000 | central |  |  |  | 0.831 | 0.000 |
| step_size_a_mult | 4.000 | central |  |  |  | 0.830 | 0.001 |
| param_scale_mult | 0.250 | central |  |  |  | 0.289 | 0.224 |
| param_scale_mult | 1.000 | central |  |  |  | 0.831 | 0.000 |
| param_scale_mult | 4.000 | central |  |  |  | 0.817 | 0.006 |

## inventory__FOODS_2_128_WI_2_validation__I2
tuned hyperparameters (validation only): eps=0.4, a=0.5; budget=40000 rollouts; seeds=10

### Final validation return, sample efficiency, runtime
| method | final_val_mean | final_val_sd | final_val_ci_lo | final_val_ci_hi | rollouts_to_90pct_mean | rollouts_to_95pct_mean | reached_95pct_share | val@25%budget | val@100%budget | runtime_per_iter_ms |
|---|---|---|---|---|---|---|---|---|---|---|
| central | 0.828 | 0.011 | 0.822 | 0.835 | 6656.000 | 16806.400 | 0.900 | 0.793 | 0.789 | 26.122 |
| forward | 0.826 | 0.014 | 0.818 | 0.833 | 5683.200 | 22681.600 | 0.800 | 0.806 | 0.778 | 21.409 |
| gaussian | 0.823 | 0.018 | 0.811 | 0.833 | 14809.600 | 32038.400 | 0.700 | 0.788 | 0.780 | 25.480 |
| regression | 0.830 | 0.010 | 0.823 | 0.835 | 8192.000 | 11072.000 | 1.000 | 0.782 | 0.799 | 31.910 |
| spsa | 0.829 | 0.009 | 0.823 | 0.834 | 12697.600 | 24281.600 | 0.900 | 0.748 | 0.815 | 47.436 |
| central_nocrn | 0.825 | 0.014 | 0.817 | 0.833 | 6604.800 | 23321.600 | 0.800 | 0.799 | 0.780 | 26.635 |
| grid | 0.524 | 0.035 | 0.504 | 0.542 | 80000.000 | 80000.000 | 0.000 | 0.641 | 0.524 | 5.681 |
| neldermead | 0.518 | 0.195 | 0.397 | 0.620 | 64236.800 | 72102.400 | 0.100 | 0.518 | 0.518 | 5.633 |
| random | 0.527 | 0.040 | 0.502 | 0.550 | 40064.000 | 56025.600 | 0.300 | 0.517 | 0.527 | 5.977 |
| spsa_nocrn | 0.826 | 0.012 | 0.818 | 0.832 | 14233.600 | 34022.400 | 0.800 | 0.740 | 0.813 | 46.946 |

### Stability (share of seeds)
| method | converged | unconverged | bound | stall | diverge |
|---|---|---|---|---|---|
| central | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| forward | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| gaussian | 0.900 | 0.100 | 0.000 | 0.000 | 0.000 |
| regression | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| spsa | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| central_nocrn | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| grid | 0.000 | 0.500 | 0.000 | 0.100 | 0.400 |
| neldermead | 0.400 | 0.000 | 0.000 | 0.600 | 0.000 |
| random | 0.300 | 0.200 | 0.000 | 0.100 | 0.400 |
| spsa_nocrn | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |

### Gradient quality vs reference (rows: point x estimator)
| point | estimator | cosine | rel_l2 | sign_agree | bias_l2 | variance |
|---|---|---|---|---|---|---|
| start | forward | 0.999 | 0.094 | 1.000 | 0.037 | 0.008 |
| start | central | 0.998 | 0.107 | 1.000 | 0.065 | 0.008 |
| start | spsa | 0.927 | 0.381 | 0.867 | 0.045 | 0.236 |
| start | regression | 0.996 | 0.122 | 1.000 | 0.086 | 0.010 |
| start | gaussian | 0.813 | 0.883 | 0.800 | 0.152 | 1.222 |
| mid | forward | 0.991 | 0.216 | 1.000 | 0.100 | 0.003 |
| mid | central | 0.985 | 0.206 | 1.000 | 0.086 | 0.004 |
| mid | spsa | 0.912 | 0.402 | 0.867 | 0.084 | 0.050 |
| mid | regression | 0.971 | 0.276 | 0.983 | 0.108 | 0.010 |
| mid | gaussian | 0.789 | 0.843 | 0.800 | 0.119 | 0.261 |
| final | forward | 0.905 | 0.472 | 1.000 | 0.276 | 0.003 |
| final | central | 0.931 | 0.396 | 1.000 | 0.229 | 0.004 |
| final | spsa | 0.894 | 0.529 | 0.867 | 0.248 | 0.053 |
| final | regression | 0.940 | 0.376 | 1.000 | 0.213 | 0.008 |
| final | gaussian | 0.759 | 0.761 | 0.800 | 0.192 | 0.280 |

### Common random numbers
| estimator | crn | variance | cosine | bias_l2 |
|---|---|---|---|---|
| forward | True | 0.012 | 0.999 | 0.048 |
| forward | False | 0.234 | 0.937 | 0.133 |
| central | True | 0.012 | 0.998 | 0.067 |
| central | False | 0.063 | 0.984 | 0.070 |
| spsa | True | 0.236 | 0.893 | 0.169 |
| spsa | False | 0.225 | 0.912 | 0.065 |
| regression | True | 0.017 | 0.996 | 0.077 |
| regression | False | 0.197 | 0.948 | 0.231 |
| gaussian | True | 1.824 | 0.787 | 0.175 |
| gaussian | False | 1.311 | 0.743 | 0.068 |

| estimator | crn | final_val_mean | final_val_sd | n |
|---|---|---|---|---|
| central | True | 0.828 | 0.011 | 10 |
| central | False | 0.825 | 0.013 | 10 |
| spsa | True | 0.829 | 0.009 | 10 |
| spsa | False | 0.826 | 0.011 | 10 |

### Optimality gap vs exhaustive grid on validation paths
| method | gap_mean | gap_sd | grid_best_val |
|---|---|---|---|
| central | 0.008 | 0.011 | 0.837 |
| central_nocrn | 0.011 | 0.014 | 0.837 |
| forward | 0.011 | 0.014 | 0.837 |
| gaussian | 0.014 | 0.018 | 0.837 |
| grid | 0.313 | 0.035 | 0.837 |
| neldermead | 0.319 | 0.195 | 0.837 |
| random | 0.310 | 0.040 | 0.837 |
| regression | 0.007 | 0.010 | 0.837 |
| spsa | 0.008 | 0.009 | 0.837 |
| spsa_nocrn | 0.011 | 0.012 | 0.837 |

### Paired Wilcoxon tests (Holm; 52 comparisons in total, 20 significant at 0.05)
| outcome | A | B | mean_diff_A_minus_B | p_holm | rank_biserial | cohens_dz |
|---|---|---|---|---|---|---|
| final_validation_return | forward | random | 0.299 | 0.041 | 1.000 | 6.946 |
| final_validation_return | forward | neldermead | 0.308 | 0.041 | 1.000 | 1.545 |
| final_validation_return | central | random | 0.302 | 0.041 | 1.000 | 6.914 |
| final_validation_return | central | neldermead | 0.310 | 0.041 | 1.000 | 1.561 |
| final_validation_return | spsa | random | 0.302 | 0.041 | 1.000 | 7.096 |
| final_validation_return | spsa | neldermead | 0.311 | 0.041 | 1.000 | 1.587 |
| final_validation_return | regression | random | 0.303 | 0.041 | 1.000 | 7.027 |
| final_validation_return | regression | neldermead | 0.312 | 0.041 | 1.000 | 1.564 |
| final_validation_return | gaussian | random | 0.296 | 0.041 | 1.000 | 6.373 |
| final_validation_return | gaussian | neldermead | 0.305 | 0.041 | 1.000 | 1.535 |
| rollouts_to_95pct | spsa | regression | 13209.600 | 0.041 | 1.000 | 0.594 |
| gradient_cosine_start | forward | central | 0.001 | 0.000 | 1.000 | 4.184 |
| gradient_cosine_start | forward | spsa | 0.072 | 0.000 | 0.953 | 0.613 |
| gradient_cosine_start | forward | regression | 0.003 | 0.000 | 0.828 | 0.742 |
| gradient_cosine_start | forward | gaussian | 0.186 | 0.000 | 0.978 | 0.765 |
| gradient_cosine_start | central | spsa | 0.071 | 0.000 | 0.884 | 0.603 |
| gradient_cosine_start | central | gaussian | 0.185 | 0.000 | 0.957 | 0.761 |
| gradient_cosine_start | spsa | regression | -0.069 | 0.001 | -0.712 | -0.582 |
| gradient_cosine_start | spsa | gaussian | 0.114 | 0.019 | 0.535 | 0.425 |
| gradient_cosine_start | regression | gaussian | 0.183 | 0.000 | 0.888 | 0.751 |

### Sensitivity (see CSV; plots saved)
| param | value | estimator | bias_l2 | variance | cosine | final_val_mean | final_val_sd |
|---|---|---|---|---|---|---|---|
| eps | 0.010 | forward | 0.027 | 0.006 | 1.000 |  |  |
| eps | 0.030 | forward | 0.152 | 0.011 | 0.999 |  |  |
| eps | 0.100 | forward | 0.077 | 0.008 | 0.998 |  |  |
| eps | 0.300 | forward | 0.033 | 0.006 | 1.000 |  |  |
| eps | 1.000 | forward | 0.023 | 0.006 | 1.000 |  |  |
| eps | 0.010 | central | 0.027 | 0.006 | 1.000 |  |  |
| eps | 0.030 | central | 0.089 | 0.008 | 1.000 |  |  |
| eps | 0.100 | central | 0.087 | 0.007 | 0.998 |  |  |
| eps | 0.300 | central | 0.065 | 0.006 | 0.999 |  |  |
| eps | 1.000 | central | 0.092 | 0.006 | 0.997 |  |  |
| eps | 0.010 | spsa | 0.063 | 0.285 | 0.906 |  |  |
| eps | 0.030 | spsa | 0.110 | 0.319 | 0.904 |  |  |
| eps | 0.100 | spsa | 0.085 | 0.308 | 0.903 |  |  |
| eps | 0.300 | spsa | 0.070 | 0.300 | 0.902 |  |  |
| eps | 1.000 | spsa | 0.071 | 0.292 | 0.899 |  |  |
| N_paths | 8.000 | central | 0.069 | 0.055 | 0.998 |  |  |
| N_paths | 16.000 | central | 0.077 | 0.025 | 0.998 |  |  |
| N_paths | 32.000 | central | 0.063 | 0.013 | 0.998 |  |  |
| N_paths | 64.000 | central | 0.064 | 0.009 | 0.998 |  |  |
| N_paths | 128.000 | central | 0.066 | 0.003 | 0.998 |  |  |
| N_paths | 8.000 | spsa | 0.039 | 0.333 | 0.896 |  |  |
| N_paths | 16.000 | spsa | 0.075 | 0.155 | 0.957 |  |  |
| N_paths | 32.000 | spsa | 0.196 | 0.341 | 0.829 |  |  |
| N_paths | 64.000 | spsa | 0.139 | 0.190 | 0.913 |  |  |
| N_paths | 128.000 | spsa | 0.233 | 0.241 | 0.862 |  |  |
| N_paths | 8.000 | regression | 0.073 | 0.069 | 0.995 |  |  |
| N_paths | 16.000 | regression | 0.084 | 0.033 | 0.996 |  |  |
| N_paths | 32.000 | regression | 0.054 | 0.014 | 0.998 |  |  |
| N_paths | 64.000 | regression | 0.071 | 0.013 | 0.996 |  |  |
| N_paths | 128.000 | regression | 0.075 | 0.005 | 0.997 |  |  |
| spsa_m | 1.000 | spsa | 0.488 | 0.723 | 0.796 |  |  |
| spsa_m | 2.000 | spsa | 0.194 | 0.530 | 0.865 |  |  |
| spsa_m | 4.000 | spsa | 0.054 | 0.232 | 0.914 |  |  |
| spsa_m | 8.000 | spsa | 0.054 | 0.111 | 0.958 |  |  |
| spsa_m | 16.000 | spsa | 0.054 | 0.063 | 0.977 |  |  |
| regression_K | 2.000 | regression | 0.142 | 0.323 | 0.904 |  |  |
| regression_K | 4.000 | regression | 0.069 | 0.014 | 0.997 |  |  |
| regression_K | 8.000 | regression | 0.074 | 0.009 | 0.997 |  |  |
| regression_K | 16.000 | regression | 0.072 | 0.008 | 0.997 |  |  |
| step_size_a_mult | 0.250 | central |  |  |  | 0.567 | 0.145 |
| step_size_a_mult | 1.000 | central |  |  |  | 0.805 | 0.003 |
| step_size_a_mult | 4.000 | central |  |  |  | 0.808 | 0.013 |
| param_scale_mult | 0.250 | central |  |  |  | 0.333 | 0.201 |
| param_scale_mult | 1.000 | central |  |  |  | 0.805 | 0.003 |
| param_scale_mult | 4.000 | central |  |  |  | 0.793 | 0.017 |

## inventory__FOODS_2_128_WI_2_validation__I3
tuned hyperparameters (validation only): eps=0.05, a=2.0; budget=40000 rollouts; seeds=10

### Final validation return, sample efficiency, runtime
| method | final_val_mean | final_val_sd | final_val_ci_lo | final_val_ci_hi | rollouts_to_90pct_mean | rollouts_to_95pct_mean | reached_95pct_share | val@25%budget | val@100%budget | runtime_per_iter_ms |
|---|---|---|---|---|---|---|---|---|---|---|
| central | 0.777 | 0.014 | 0.769 | 0.785 | 5888.000 | 7680.000 | 1.000 | 0.766 | 0.678 | 135.956 |
| forward | 0.777 | 0.014 | 0.769 | 0.786 | 3238.400 | 11520.000 | 0.900 | 0.748 | 0.606 | 79.809 |
| gaussian | 0.750 | 0.040 | 0.723 | 0.771 | 28096.000 | 42816.000 | 0.500 | 0.736 | 0.649 | 137.779 |
| regression | 0.776 | 0.014 | 0.769 | 0.785 | 6720.000 | 14720.000 | 0.900 | 0.765 | 0.688 | 142.918 |
| spsa | 0.737 | 0.037 | 0.716 | 0.758 | 40819.200 | 41331.200 | 0.500 | 0.694 | 0.541 | 58.389 |
| central_nocrn | 0.647 | 0.085 | 0.598 | 0.696 | 72512.000 | 72512.000 | 0.100 | 0.558 | 0.518 | 137.395 |
| neldermead | 0.450 | 0.259 | 0.290 | 0.593 | 80000.000 | 80000.000 | 0.000 | 0.450 | 0.450 | 6.870 |
| random | 0.373 | 0.062 | 0.341 | 0.413 | 80000.000 | 80000.000 | 0.000 | 0.347 | 0.373 | 7.085 |
| spsa_nocrn | 0.683 | 0.030 | 0.664 | 0.699 | 80000.000 | 80000.000 | 0.000 | 0.556 | 0.479 | 59.733 |

### Stability (share of seeds)
| method | converged | unconverged | bound | stall | diverge |
|---|---|---|---|---|---|
| central | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| forward | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| gaussian | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| regression | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| spsa | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| central_nocrn | 0.200 | 0.200 | 0.200 | 0.400 | 0.000 |
| neldermead | 0.100 | 0.000 | 0.000 | 0.900 | 0.000 |
| random | 0.100 | 0.200 | 0.000 | 0.200 | 0.500 |
| spsa_nocrn | 0.000 | 0.500 | 0.200 | 0.300 | 0.000 |

### Gradient quality vs reference (rows: point x estimator)
| point | estimator | cosine | rel_l2 | sign_agree | bias_l2 | variance |
|---|---|---|---|---|---|---|
| start | forward | 0.998 | 0.105 | 1.000 | 0.048 | 0.011 |
| start | central | 0.999 | 0.098 | 1.000 | 0.030 | 0.011 |
| start | spsa | 0.573 | 1.458 | 0.627 | 0.265 | 2.356 |
| start | regression | 0.998 | 0.112 | 0.997 | 0.046 | 0.012 |
| start | gaussian | 0.719 | 1.063 | 0.727 | 0.224 | 1.239 |
| mid | forward | 0.996 | 0.173 | 0.963 | 0.023 | 0.003 |
| mid | central | 0.996 | 0.157 | 0.930 | 0.016 | 0.003 |
| mid | spsa | 0.563 | 1.481 | 0.573 | 0.080 | 0.220 |
| mid | regression | 0.994 | 0.177 | 0.940 | 0.019 | 0.003 |
| mid | gaussian | 0.710 | 1.114 | 0.677 | 0.069 | 0.126 |
| final | forward | 0.994 | 0.151 | 0.997 | 0.043 | 0.004 |
| final | central | 0.996 | 0.132 | 1.000 | 0.029 | 0.004 |
| final | spsa | 0.558 | 1.407 | 0.617 | 0.127 | 0.464 |
| final | regression | 0.995 | 0.142 | 0.993 | 0.031 | 0.004 |
| final | gaussian | 0.714 | 1.049 | 0.693 | 0.092 | 0.262 |

### Common random numbers
| estimator | crn | variance | cosine | bias_l2 |
|---|---|---|---|---|
| forward | True | 0.015 | 0.998 | 0.037 |
| forward | False | 96.210 | 0.094 | 1.070 |
| central | True | 0.015 | 0.999 | 0.032 |
| central | False | 22.924 | 0.198 | 0.500 |
| spsa | True | 2.335 | 0.610 | 0.284 |
| spsa | False | 7.259 | 0.285 | 0.552 |
| regression | True | 0.016 | 0.997 | 0.048 |
| regression | False | 7.767 | 0.320 | 0.360 |
| gaussian | True | 0.970 | 0.729 | 0.180 |
| gaussian | False | 3.347 | 0.459 | 0.268 |

| estimator | crn | final_val_mean | final_val_sd | n |
|---|---|---|---|---|
| central | True | 0.777 | 0.013 | 10 |
| central | False | 0.647 | 0.080 | 10 |
| spsa | True | 0.737 | 0.035 | 10 |
| spsa | False | 0.683 | 0.028 | 10 |

### Paired Wilcoxon tests (Holm; 52 comparisons in total, 28 significant at 0.05)
| outcome | A | B | mean_diff_A_minus_B | p_holm | rank_biserial | cohens_dz |
|---|---|---|---|---|---|---|
| final_validation_return | forward | spsa | 0.040 | 0.041 | 1.000 | 1.481 |
| final_validation_return | forward | gaussian | 0.027 | 0.041 | 0.964 | 0.870 |
| final_validation_return | forward | random | 0.404 | 0.041 | 1.000 | 6.088 |
| final_validation_return | forward | neldermead | 0.327 | 0.041 | 1.000 | 1.324 |
| final_validation_return | central | spsa | 0.040 | 0.041 | 1.000 | 1.475 |
| final_validation_return | central | gaussian | 0.027 | 0.041 | 0.964 | 0.870 |
| final_validation_return | central | random | 0.404 | 0.041 | 1.000 | 6.102 |
| final_validation_return | central | neldermead | 0.327 | 0.041 | 1.000 | 1.321 |
| final_validation_return | spsa | regression | -0.039 | 0.041 | -1.000 | -1.440 |
| final_validation_return | spsa | random | 0.364 | 0.041 | 1.000 | 4.099 |
| final_validation_return | spsa | neldermead | 0.287 | 0.041 | 1.000 | 1.243 |
| final_validation_return | regression | gaussian | 0.027 | 0.041 | 0.964 | 0.834 |
| final_validation_return | regression | random | 0.403 | 0.041 | 1.000 | 6.123 |
| final_validation_return | regression | neldermead | 0.326 | 0.041 | 1.000 | 1.323 |
| final_validation_return | gaussian | random | 0.377 | 0.041 | 1.000 | 5.854 |
| final_validation_return | gaussian | neldermead | 0.299 | 0.041 | 1.000 | 1.254 |
| rollouts_to_95pct | central | random | -72320.000 | 0.041 | -1.000 | -21.188 |
| rollouts_to_95pct | central | neldermead | -72320.000 | 0.041 | -1.000 | -21.188 |
| gradient_cosine_start | forward | central | -0.000 | 0.008 | -0.587 | -0.543 |
| gradient_cosine_start | forward | spsa | 0.425 | 0.000 | 1.000 | 3.921 |
| gradient_cosine_start | forward | regression | 0.001 | 0.008 | 0.566 | 0.563 |
| gradient_cosine_start | forward | gaussian | 0.279 | 0.000 | 1.000 | 2.756 |
| gradient_cosine_start | central | spsa | 0.426 | 0.000 | 1.000 | 3.927 |
| gradient_cosine_start | central | regression | 0.001 | 0.000 | 0.974 | 1.383 |
| gradient_cosine_start | central | gaussian | 0.280 | 0.000 | 1.000 | 2.751 |
| gradient_cosine_start | spsa | regression | -0.425 | 0.000 | -1.000 | -3.922 |
| gradient_cosine_start | spsa | gaussian | -0.146 | 0.000 | -0.837 | -1.008 |
| gradient_cosine_start | regression | gaussian | 0.278 | 0.000 | 1.000 | 2.738 |

### Sensitivity (see CSV; plots saved)
| param | value | estimator | bias_l2 | variance | cosine | final_val_mean | final_val_sd |
|---|---|---|---|---|---|---|---|
| eps | 0.010 | forward | 0.082 | 0.010 | 0.997 |  |  |
| eps | 0.030 | forward | 0.061 | 0.009 | 0.998 |  |  |
| eps | 0.100 | forward | 0.088 | 0.009 | 0.997 |  |  |
| eps | 0.300 | forward | 0.065 | 0.008 | 0.998 |  |  |
| eps | 1.000 | forward | 0.132 | 0.009 | 0.991 |  |  |
| eps | 0.010 | central | 0.066 | 0.009 | 0.998 |  |  |
| eps | 0.030 | central | 0.035 | 0.009 | 0.999 |  |  |
| eps | 0.100 | central | 0.062 | 0.008 | 0.998 |  |  |
| eps | 0.300 | central | 0.060 | 0.008 | 0.998 |  |  |
| eps | 1.000 | central | 0.101 | 0.007 | 0.995 |  |  |
| eps | 0.010 | spsa | 0.294 | 2.133 | 0.552 |  |  |
| eps | 0.030 | spsa | 0.275 | 2.111 | 0.551 |  |  |
| eps | 0.100 | spsa | 0.284 | 2.092 | 0.551 |  |  |
| eps | 0.300 | spsa | 0.275 | 1.871 | 0.550 |  |  |
| eps | 1.000 | spsa | 0.347 | 1.243 | 0.550 |  |  |
| N_paths | 8.000 | central | 0.054 | 0.067 | 0.993 |  |  |
| N_paths | 16.000 | central | 0.069 | 0.033 | 0.996 |  |  |
| N_paths | 32.000 | central | 0.043 | 0.017 | 0.998 |  |  |
| N_paths | 64.000 | central | 0.034 | 0.010 | 0.999 |  |  |
| N_paths | 128.000 | central | 0.030 | 0.004 | 0.999 |  |  |
| N_paths | 8.000 | spsa | 0.368 | 2.499 | 0.572 |  |  |
| N_paths | 16.000 | spsa | 0.235 | 2.275 | 0.550 |  |  |
| N_paths | 32.000 | spsa | 0.295 | 2.422 | 0.564 |  |  |
| N_paths | 64.000 | spsa | 0.445 | 1.805 | 0.560 |  |  |
| N_paths | 128.000 | spsa | 0.227 | 2.176 | 0.600 |  |  |
| N_paths | 8.000 | regression | 0.068 | 0.068 | 0.992 |  |  |
| N_paths | 16.000 | regression | 0.077 | 0.040 | 0.995 |  |  |
| N_paths | 32.000 | regression | 0.055 | 0.019 | 0.997 |  |  |
| N_paths | 64.000 | regression | 0.051 | 0.013 | 0.997 |  |  |
| N_paths | 128.000 | regression | 0.044 | 0.005 | 0.998 |  |  |
| spsa_m | 1.000 | spsa | 0.685 | 10.859 | 0.318 |  |  |
| spsa_m | 2.000 | spsa | 0.294 | 5.247 | 0.432 |  |  |
| spsa_m | 4.000 | spsa | 0.269 | 2.282 | 0.566 |  |  |
| spsa_m | 8.000 | spsa | 0.148 | 1.252 | 0.688 |  |  |
| spsa_m | 16.000 | spsa | 0.161 | 0.610 | 0.793 |  |  |
| regression_K | 10.000 | regression | 0.412 | 4.730 | 0.924 |  |  |
| regression_K | 20.000 | regression | 0.048 | 0.014 | 0.998 |  |  |
| regression_K | 40.000 | regression | 0.048 | 0.013 | 0.998 |  |  |
| regression_K | 80.000 | regression | 0.047 | 0.012 | 0.998 |  |  |
| step_size_a_mult | 0.250 | central |  |  |  | 0.603 | 0.089 |
| step_size_a_mult | 1.000 | central |  |  |  | 0.770 | 0.009 |
| step_size_a_mult | 4.000 | central |  |  |  | 0.757 | 0.013 |
| param_scale_mult | 0.250 | central |  |  |  | 0.284 | 0.223 |
| param_scale_mult | 1.000 | central |  |  |  | 0.770 | 0.009 |
| param_scale_mult | 4.000 | central |  |  |  | 0.539 | 0.046 |

## inventory__FOODS_3_124_CA_3_validation__I1
tuned hyperparameters (validation only): eps=0.4, a=0.5; budget=40000 rollouts; seeds=10

### Final validation return, sample efficiency, runtime
| method | final_val_mean | final_val_sd | final_val_ci_lo | final_val_ci_hi | rollouts_to_90pct_mean | rollouts_to_95pct_mean | reached_95pct_share | val@25%budget | val@100%budget | runtime_per_iter_ms |
|---|---|---|---|---|---|---|---|---|---|---|
| central | 2.890 | 0.000 | 2.890 | 2.890 | 409.600 | 486.400 | 1.000 | 2.691 | 2.567 | 7.428 |
| forward | 2.890 | 0.000 | 2.890 | 2.890 | 460.800 | 563.200 | 1.000 | 2.717 | 2.593 | 7.687 |
| gaussian | 2.890 | 0.000 | 2.889 | 2.890 | 1126.400 | 1331.200 | 1.000 | 2.771 | 2.617 | 22.498 |
| regression | 2.890 | 0.000 | 2.890 | 2.890 | 614.400 | 729.600 | 1.000 | 2.736 | 2.593 | 16.015 |
| spsa | 2.890 | 0.000 | 2.890 | 2.890 | 1638.400 | 1945.600 | 1.000 | 2.843 | 2.692 | 23.255 |
| central_nocrn | 2.890 | 0.000 | 2.890 | 2.890 | 460.800 | 486.400 | 1.000 | 2.700 | 2.568 | 13.885 |
| grid | 2.510 | 0.045 | 2.481 | 2.533 | 9184.000 | 9184.000 | 1.000 | 2.303 | 2.510 | 4.722 |
| neldermead | 2.589 | 0.250 | 2.445 | 2.740 | 8268.800 | 16230.400 | 0.800 | 2.589 | 2.589 | 4.773 |
| random | 2.510 | 0.044 | 2.481 | 2.533 | 48006.400 | 48006.400 | 0.400 | 2.508 | 2.510 | 4.742 |
| spsa_nocrn | 2.890 | 0.000 | 2.890 | 2.890 | 1638.400 | 2048.000 | 1.000 | 2.845 | 2.689 | 41.464 |

### Stability (share of seeds)
| method | converged | unconverged | bound | stall | diverge |
|---|---|---|---|---|---|
| central | 0.700 | 0.000 | 0.000 | 0.300 | 0.000 |
| forward | 0.700 | 0.000 | 0.000 | 0.300 | 0.000 |
| gaussian | 0.700 | 0.000 | 0.000 | 0.300 | 0.000 |
| regression | 0.700 | 0.000 | 0.000 | 0.300 | 0.000 |
| spsa | 0.700 | 0.000 | 0.000 | 0.300 | 0.000 |
| central_nocrn | 0.500 | 0.200 | 0.000 | 0.300 | 0.000 |
| grid | 0.000 | 0.300 | 0.000 | 0.000 | 0.700 |
| neldermead | 0.400 | 0.000 | 0.000 | 0.300 | 0.300 |
| random | 0.300 | 0.000 | 0.000 | 0.000 | 0.700 |
| spsa_nocrn | 0.700 | 0.000 | 0.000 | 0.300 | 0.000 |

### Gradient quality vs reference (rows: point x estimator)
| point | estimator | cosine | rel_l2 | sign_agree | bias_l2 | variance |
|---|---|---|---|---|---|---|
| start | forward | 1.000 | 0.122 | 1.000 | 0.219 | 0.013 |
| start | central | 1.000 | 0.051 | 1.000 | 0.057 | 0.012 |
| start | spsa | 1.000 | 0.041 | 1.000 | 0.068 | 0.002 |
| start | regression | 1.000 | 0.137 | 1.000 | 0.076 | 0.081 |
| start | gaussian | 1.000 | 0.577 | 1.000 | 0.189 | 1.644 |
| start | dense_grid_derivative | nan | nan | nan | 0.007 | 0.000 |
| mid | forward | 0.867 | 0.511 | 0.933 | 0.043 | 0.003 |
| mid | central | 1.000 | 0.450 | 1.000 | 0.013 | 0.004 |
| mid | spsa | 1.000 | 0.236 | 1.000 | 0.022 | 0.001 |
| mid | regression | 1.000 | 0.597 | 1.000 | 0.013 | 0.007 |
| mid | gaussian | 1.000 | 0.802 | 1.000 | 0.005 | 0.013 |
| mid | dense_grid_derivative | nan | nan | nan | 0.000 | 0.000 |
| final | forward | 1.000 | 0.209 | 1.000 | 0.185 | 0.013 |
| final | central | 1.000 | 0.107 | 1.000 | 0.034 | 0.012 |
| final | spsa | 1.000 | 0.040 | 1.000 | 0.014 | 0.002 |
| final | regression | 1.000 | 0.172 | 1.000 | 0.025 | 0.032 |
| final | gaussian | 1.000 | 0.574 | 1.000 | 0.143 | 0.370 |
| final | dense_grid_derivative | nan | nan | nan | 0.000 | 0.000 |

### Common random numbers
| estimator | crn | variance | cosine | bias_l2 |
|---|---|---|---|---|
| forward | True | 0.008 | 1.000 | 0.246 |
| forward | False | 0.254 | 1.000 | 0.243 |
| central | True | 0.008 | 1.000 | 0.031 |
| central | False | 0.057 | 1.000 | 0.047 |
| spsa | True | 0.002 | 1.000 | 0.049 |
| spsa | False | 0.017 | 1.000 | 0.062 |
| regression | True | 0.092 | 1.000 | 0.113 |
| regression | False | 0.616 | 1.000 | 0.337 |
| gaussian | True | 4.718 | 1.000 | 0.255 |
| gaussian | False | 3.554 | 1.000 | 0.274 |

| estimator | crn | final_val_mean | final_val_sd | n |
|---|---|---|---|---|
| central | True | 2.890 | 0.000 | 10 |
| central | False | 2.890 | 0.000 | 10 |
| spsa | True | 2.890 | 0.000 | 10 |
| spsa | False | 2.890 | 0.000 | 10 |

### Optimality gap vs exhaustive grid on validation paths
| method | gap_mean | gap_sd | grid_best_val |
|---|---|---|---|
| central | 0.000 | 0.000 | 2.890 |
| central_nocrn | 0.000 | 0.000 | 2.890 |
| forward | 0.000 | 0.000 | 2.890 |
| gaussian | 0.001 | 0.000 | 2.890 |
| grid | 0.380 | 0.045 | 2.890 |
| neldermead | 0.301 | 0.250 | 2.890 |
| random | 0.380 | 0.044 | 2.890 |
| regression | 0.000 | 0.000 | 2.890 |
| spsa | 0.000 | 0.000 | 2.890 |
| spsa_nocrn | 0.000 | 0.000 | 2.890 |

### Paired Wilcoxon tests (Holm; 52 comparisons in total, 8 significant at 0.05)
| outcome | A | B | mean_diff_A_minus_B | p_holm | rank_biserial | cohens_dz |
|---|---|---|---|---|---|---|
| final_validation_return | forward | random | 0.379 | 0.041 | 1.000 | 8.560 |
| final_validation_return | central | random | 0.379 | 0.041 | 1.000 | 8.576 |
| final_validation_return | spsa | random | 0.379 | 0.041 | 1.000 | 8.581 |
| final_validation_return | spsa | neldermead | 0.300 | 0.041 | 1.000 | 1.203 |
| final_validation_return | regression | random | 0.379 | 0.041 | 1.000 | 8.600 |
| final_validation_return | regression | neldermead | 0.300 | 0.041 | 1.000 | 1.203 |
| final_validation_return | gaussian | random | 0.379 | 0.041 | 1.000 | 8.617 |
| final_validation_return | gaussian | neldermead | 0.300 | 0.041 | 1.000 | 1.202 |

### Sensitivity (see CSV; plots saved)
| param | value | estimator | bias_l2 | variance | cosine | final_val_mean | final_val_sd |
|---|---|---|---|---|---|---|---|
| eps | 0.010 | forward | 0.020 | 0.006 | 1.000 |  |  |
| eps | 0.030 | forward | 0.074 | 0.007 | 1.000 |  |  |
| eps | 0.100 | forward | 0.093 | 0.007 | 1.000 |  |  |
| eps | 0.300 | forward | 0.212 | 0.006 | 1.000 |  |  |
| eps | 1.000 | forward | 0.491 | 0.006 | 1.000 |  |  |
| eps | 0.010 | central | 0.020 | 0.006 | 1.000 |  |  |
| eps | 0.030 | central | 0.027 | 0.007 | 1.000 |  |  |
| eps | 0.100 | central | 0.039 | 0.007 | 1.000 |  |  |
| eps | 0.300 | central | 0.000 | 0.006 | 1.000 |  |  |
| eps | 1.000 | central | 0.098 | 0.005 | 1.000 |  |  |
| eps | 0.010 | spsa | 0.042 | 0.002 | 1.000 |  |  |
| eps | 0.030 | spsa | 0.006 | 0.002 | 1.000 |  |  |
| eps | 0.100 | spsa | 0.019 | 0.002 | 1.000 |  |  |
| eps | 0.300 | spsa | 0.020 | 0.002 | 1.000 |  |  |
| eps | 1.000 | spsa | 0.114 | 0.001 | 1.000 |  |  |
| N_paths | 8.000 | central | 0.082 | 0.085 | 1.000 |  |  |
| N_paths | 16.000 | central | 0.066 | 0.034 | 1.000 |  |  |
| N_paths | 32.000 | central | 0.059 | 0.012 | 1.000 |  |  |
| N_paths | 64.000 | central | 0.054 | 0.008 | 1.000 |  |  |
| N_paths | 128.000 | central | 0.046 | 0.003 | 1.000 |  |  |
| N_paths | 8.000 | spsa | 0.062 | 0.015 | 1.000 |  |  |
| N_paths | 16.000 | spsa | 0.052 | 0.008 | 1.000 |  |  |
| N_paths | 32.000 | spsa | 0.047 | 0.003 | 1.000 |  |  |
| N_paths | 64.000 | spsa | 0.035 | 0.001 | 1.000 |  |  |
| N_paths | 128.000 | spsa | 0.048 | 0.001 | 1.000 |  |  |
| N_paths | 8.000 | regression | 0.048 | 0.200 | 1.000 |  |  |
| N_paths | 16.000 | regression | 0.097 | 0.099 | 1.000 |  |  |
| N_paths | 32.000 | regression | 0.039 | 0.077 | 1.000 |  |  |
| N_paths | 64.000 | regression | 0.011 | 0.075 | 1.000 |  |  |
| N_paths | 128.000 | regression | 0.092 | 0.099 | 1.000 |  |  |
| spsa_m | 1.000 | spsa | 0.088 | 0.008 | 1.000 |  |  |
| spsa_m | 2.000 | spsa | 0.079 | 0.005 | 1.000 |  |  |
| spsa_m | 4.000 | spsa | 0.072 | 0.002 | 1.000 |  |  |
| spsa_m | 8.000 | spsa | 0.065 | 0.001 | 1.000 |  |  |
| spsa_m | 16.000 | spsa | 0.060 | 0.000 | 1.000 |  |  |
| regression_K | 1.000 | regression | 0.007 | 0.077 | 1.000 |  |  |
| regression_K | 2.000 | regression | 0.023 | 0.113 | 1.000 |  |  |
| regression_K | 4.000 | regression | 0.013 | 0.087 | 1.000 |  |  |
| regression_K | 8.000 | regression | 0.087 | 0.063 | 1.000 |  |  |
| step_size_a_mult | 0.250 | central |  |  |  | 2.890 | 0.000 |
| step_size_a_mult | 1.000 | central |  |  |  | 2.890 | 0.000 |
| step_size_a_mult | 4.000 | central |  |  |  | 2.881 | 0.009 |
| param_scale_mult | 0.250 | central |  |  |  | 2.528 | 0.281 |
| param_scale_mult | 1.000 | central |  |  |  | 2.890 | 0.000 |
| param_scale_mult | 4.000 | central |  |  |  | 2.684 | 0.083 |

## inventory__FOODS_3_124_CA_3_validation__I2
tuned hyperparameters (validation only): eps=0.4, a=0.5; budget=40000 rollouts; seeds=10

### Final validation return, sample efficiency, runtime
| method | final_val_mean | final_val_sd | final_val_ci_lo | final_val_ci_hi | rollouts_to_90pct_mean | rollouts_to_95pct_mean | reached_95pct_share | val@25%budget | val@100%budget | runtime_per_iter_ms |
|---|---|---|---|---|---|---|---|---|---|---|
| central | 3.106 | 0.023 | 3.092 | 3.119 | 1075.200 | 17228.800 | 0.800 | 3.006 | 2.870 | 25.680 |
| forward | 3.111 | 0.031 | 3.094 | 3.129 | 806.400 | 17152.000 | 0.800 | 2.994 | 2.876 | 22.673 |
| gaussian | 3.112 | 0.022 | 3.099 | 3.124 | 1280.000 | 9689.600 | 0.900 | 3.010 | 2.875 | 26.462 |
| regression | 3.107 | 0.028 | 3.091 | 3.123 | 1280.000 | 17408.000 | 0.800 | 3.019 | 2.891 | 31.281 |
| spsa | 3.112 | 0.023 | 3.098 | 3.126 | 2048.000 | 10764.800 | 0.900 | 3.063 | 2.936 | 46.513 |
| central_nocrn | 3.108 | 0.028 | 3.091 | 3.123 | 1126.400 | 17484.800 | 0.800 | 3.002 | 2.855 | 25.525 |
| grid | 2.773 | 0.059 | 2.737 | 2.805 | 10496.000 | 59936.000 | 0.300 | 2.789 | 2.773 | 5.595 |
| neldermead | 2.521 | 0.631 | 2.089 | 2.849 | 56038.400 | 64051.200 | 0.200 | 2.521 | 2.521 | 5.462 |
| random | 2.772 | 0.075 | 2.725 | 2.813 | 48038.400 | 56044.800 | 0.300 | 2.763 | 2.772 | 5.608 |
| spsa_nocrn | 3.107 | 0.033 | 3.087 | 3.126 | 10150.400 | 18764.800 | 0.800 | 3.065 | 2.933 | 46.734 |

### Stability (share of seeds)
| method | converged | unconverged | bound | stall | diverge |
|---|---|---|---|---|---|
| central | 0.900 | 0.000 | 0.000 | 0.100 | 0.000 |
| forward | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| gaussian | 0.900 | 0.000 | 0.000 | 0.100 | 0.000 |
| regression | 0.900 | 0.000 | 0.000 | 0.100 | 0.000 |
| spsa | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| central_nocrn | 0.800 | 0.200 | 0.000 | 0.000 | 0.000 |
| grid | 0.000 | 0.300 | 0.100 | 0.200 | 0.400 |
| neldermead | 0.400 | 0.000 | 0.000 | 0.400 | 0.200 |
| random | 0.100 | 0.300 | 0.000 | 0.100 | 0.500 |
| spsa_nocrn | 0.900 | 0.000 | 0.000 | 0.100 | 0.000 |

### Gradient quality vs reference (rows: point x estimator)
| point | estimator | cosine | rel_l2 | sign_agree | bias_l2 | variance |
|---|---|---|---|---|---|---|
| start | forward | 0.998 | 0.118 | 1.000 | 0.258 | 0.016 |
| start | central | 0.997 | 0.092 | 1.000 | 0.185 | 0.011 |
| start | spsa | 0.925 | 0.371 | 0.867 | 0.119 | 1.267 |
| start | regression | 0.992 | 0.192 | 1.000 | 0.296 | 0.134 |
| start | gaussian | 0.814 | 0.898 | 0.800 | 0.327 | 7.414 |
| mid | forward | 0.968 | 0.361 | 0.983 | 0.067 | 0.005 |
| mid | central | 0.987 | 0.286 | 1.000 | 0.031 | 0.006 |
| mid | spsa | 0.956 | 0.398 | 0.983 | 0.043 | 0.011 |
| mid | regression | 0.897 | 0.566 | 0.933 | 0.016 | 0.026 |
| mid | gaussian | 0.768 | 0.989 | 0.783 | 0.024 | 0.113 |
| final | forward | 0.996 | 0.170 | 1.000 | 0.057 | 0.012 |
| final | central | 0.998 | 0.147 | 1.000 | 0.050 | 0.012 |
| final | spsa | 0.936 | 0.412 | 0.867 | 0.063 | 0.115 |
| final | regression | 0.985 | 0.327 | 0.983 | 0.116 | 0.047 |
| final | gaussian | 0.809 | 1.043 | 0.800 | 0.167 | 0.887 |

### Common random numbers
| estimator | crn | variance | cosine | bias_l2 |
|---|---|---|---|---|
| forward | True | 0.008 | 0.998 | 0.277 |
| forward | False | 0.547 | 0.962 | 0.327 |
| central | True | 0.007 | 0.997 | 0.181 |
| central | False | 0.117 | 0.991 | 0.185 |
| spsa | True | 1.225 | 0.905 | 0.334 |
| spsa | False | 1.167 | 0.916 | 0.161 |
| regression | True | 0.202 | 0.989 | 0.190 |
| regression | False | 0.561 | 0.957 | 0.296 |
| gaussian | True | 10.225 | 0.771 | 0.396 |
| gaussian | False | 6.426 | 0.775 | 0.219 |

| estimator | crn | final_val_mean | final_val_sd | n |
|---|---|---|---|---|
| central | True | 3.106 | 0.022 | 10 |
| central | False | 3.108 | 0.027 | 10 |
| spsa | True | 3.112 | 0.022 | 10 |
| spsa | False | 3.107 | 0.031 | 10 |

### Optimality gap vs exhaustive grid on validation paths
| method | gap_mean | gap_sd | grid_best_val |
|---|---|---|---|
| central | 0.050 | 0.023 | 3.155 |
| central_nocrn | 0.047 | 0.028 | 3.155 |
| forward | 0.045 | 0.031 | 3.155 |
| gaussian | 0.044 | 0.022 | 3.155 |
| grid | 0.383 | 0.059 | 3.155 |
| neldermead | 0.635 | 0.631 | 3.155 |
| random | 0.384 | 0.075 | 3.155 |
| regression | 0.049 | 0.028 | 3.155 |
| spsa | 0.043 | 0.023 | 3.155 |
| spsa_nocrn | 0.049 | 0.033 | 3.155 |

### Paired Wilcoxon tests (Holm; 52 comparisons in total, 20 significant at 0.05)
| outcome | A | B | mean_diff_A_minus_B | p_holm | rank_biserial | cohens_dz |
|---|---|---|---|---|---|---|
| final_validation_return | forward | random | 0.339 | 0.041 | 1.000 | 4.299 |
| final_validation_return | forward | neldermead | 0.590 | 0.041 | 1.000 | 0.945 |
| final_validation_return | central | random | 0.334 | 0.041 | 1.000 | 4.483 |
| final_validation_return | central | neldermead | 0.585 | 0.041 | 1.000 | 0.930 |
| final_validation_return | spsa | random | 0.341 | 0.041 | 1.000 | 4.143 |
| final_validation_return | spsa | neldermead | 0.592 | 0.041 | 1.000 | 0.946 |
| final_validation_return | regression | random | 0.335 | 0.041 | 1.000 | 4.407 |
| final_validation_return | regression | neldermead | 0.586 | 0.041 | 1.000 | 0.936 |
| final_validation_return | gaussian | random | 0.340 | 0.041 | 1.000 | 4.444 |
| final_validation_return | gaussian | neldermead | 0.591 | 0.041 | 1.000 | 0.932 |
| gradient_cosine_start | forward | central | 0.001 | 0.000 | 1.000 | 2.949 |
| gradient_cosine_start | forward | spsa | 0.074 | 0.000 | 0.815 | 0.636 |
| gradient_cosine_start | forward | regression | 0.007 | 0.000 | 0.884 | 0.660 |
| gradient_cosine_start | forward | gaussian | 0.185 | 0.000 | 0.987 | 0.713 |
| gradient_cosine_start | central | spsa | 0.072 | 0.021 | 0.553 | 0.623 |
| gradient_cosine_start | central | regression | 0.005 | 0.019 | 0.578 | 0.524 |
| gradient_cosine_start | central | gaussian | 0.183 | 0.000 | 0.970 | 0.708 |
| gradient_cosine_start | spsa | regression | -0.067 | 0.021 | -0.535 | -0.573 |
| gradient_cosine_start | spsa | gaussian | 0.111 | 0.021 | 0.514 | 0.395 |
| gradient_cosine_start | regression | gaussian | 0.178 | 0.000 | 0.888 | 0.680 |

### Sensitivity (see CSV; plots saved)
| param | value | estimator | bias_l2 | variance | cosine | final_val_mean | final_val_sd |
|---|---|---|---|---|---|---|---|
| eps | 0.010 | forward | 0.113 | 0.005 | 0.999 |  |  |
| eps | 0.030 | forward | 0.116 | 0.005 | 0.999 |  |  |
| eps | 0.100 | forward | 0.100 | 0.008 | 1.000 |  |  |
| eps | 0.300 | forward | 0.359 | 0.009 | 0.993 |  |  |
| eps | 1.000 | forward | 0.616 | 0.007 | 0.999 |  |  |
| eps | 0.010 | central | 0.067 | 0.026 | 0.999 |  |  |
| eps | 0.030 | central | 0.056 | 0.008 | 1.000 |  |  |
| eps | 0.100 | central | 0.219 | 0.007 | 0.995 |  |  |
| eps | 0.300 | central | 0.238 | 0.006 | 0.994 |  |  |
| eps | 1.000 | central | 0.237 | 0.004 | 0.995 |  |  |
| eps | 0.010 | spsa | 0.157 | 1.657 | 0.905 |  |  |
| eps | 0.030 | spsa | 0.129 | 1.522 | 0.907 |  |  |
| eps | 0.100 | spsa | 0.090 | 1.531 | 0.902 |  |  |
| eps | 0.300 | spsa | 0.118 | 1.511 | 0.900 |  |  |
| eps | 1.000 | spsa | 0.123 | 1.519 | 0.900 |  |  |
| N_paths | 8.000 | central | 0.188 | 0.083 | 0.997 |  |  |
| N_paths | 16.000 | central | 0.178 | 0.034 | 0.997 |  |  |
| N_paths | 32.000 | central | 0.183 | 0.013 | 0.997 |  |  |
| N_paths | 64.000 | central | 0.184 | 0.008 | 0.997 |  |  |
| N_paths | 128.000 | central | 0.183 | 0.004 | 0.997 |  |  |
| N_paths | 8.000 | spsa | 0.143 | 1.824 | 0.897 |  |  |
| N_paths | 16.000 | spsa | 0.204 | 0.684 | 0.957 |  |  |
| N_paths | 32.000 | spsa | 0.421 | 1.791 | 0.843 |  |  |
| N_paths | 64.000 | spsa | 0.274 | 0.953 | 0.923 |  |  |
| N_paths | 128.000 | spsa | 0.535 | 1.313 | 0.868 |  |  |
| N_paths | 8.000 | regression | 0.221 | 0.336 | 0.983 |  |  |
| N_paths | 16.000 | regression | 0.222 | 0.163 | 0.990 |  |  |
| N_paths | 32.000 | regression | 0.228 | 0.165 | 0.990 |  |  |
| N_paths | 64.000 | regression | 0.268 | 0.170 | 0.985 |  |  |
| N_paths | 128.000 | regression | 0.255 | 0.120 | 0.990 |  |  |
| spsa_m | 1.000 | spsa | 1.163 | 4.068 | 0.791 |  |  |
| spsa_m | 2.000 | spsa | 0.480 | 2.927 | 0.865 |  |  |
| spsa_m | 4.000 | spsa | 0.132 | 1.369 | 0.912 |  |  |
| spsa_m | 8.000 | spsa | 0.118 | 0.657 | 0.958 |  |  |
| spsa_m | 16.000 | spsa | 0.114 | 0.348 | 0.977 |  |  |
| regression_K | 2.000 | regression | 0.570 | 6.813 | 0.805 |  |  |
| regression_K | 4.000 | regression | 0.146 | 0.177 | 0.992 |  |  |
| regression_K | 8.000 | regression | 0.206 | 0.106 | 0.994 |  |  |
| regression_K | 16.000 | regression | 0.193 | 0.062 | 0.995 |  |  |
| step_size_a_mult | 0.250 | central |  |  |  | 3.119 | 0.030 |
| step_size_a_mult | 1.000 | central |  |  |  | 3.117 | 0.029 |
| step_size_a_mult | 4.000 | central |  |  |  | 3.114 | 0.028 |
| param_scale_mult | 0.250 | central |  |  |  | 2.537 | 0.377 |
| param_scale_mult | 1.000 | central |  |  |  | 3.117 | 0.029 |
| param_scale_mult | 4.000 | central |  |  |  | 2.911 | 0.071 |

## inventory__FOODS_3_124_CA_3_validation__I3
tuned hyperparameters (validation only): eps=0.05, a=0.5; budget=40000 rollouts; seeds=10

### Final validation return, sample efficiency, runtime
| method | final_val_mean | final_val_sd | final_val_ci_lo | final_val_ci_hi | rollouts_to_90pct_mean | rollouts_to_95pct_mean | reached_95pct_share | val@25%budget | val@100%budget | runtime_per_iter_ms |
|---|---|---|---|---|---|---|---|---|---|---|
| central | 2.864 | 0.023 | 2.850 | 2.876 | 20608.000 | 65536.000 | 0.200 | 2.845 | 2.751 | 134.719 |
| forward | 2.865 | 0.018 | 2.854 | 2.875 | 18534.400 | 72422.400 | 0.100 | 2.815 | 2.698 | 80.908 |
| gaussian | 2.872 | 0.032 | 2.854 | 2.890 | 21376.000 | 51584.000 | 0.400 | 2.827 | 2.768 | 135.045 |
| regression | 2.863 | 0.019 | 2.851 | 2.873 | 20838.400 | 72806.400 | 0.100 | 2.843 | 2.755 | 142.920 |
| spsa | 2.890 | 0.032 | 2.869 | 2.908 | 10048.000 | 26355.200 | 0.700 | 2.821 | 2.696 | 59.480 |
| central_nocrn | 2.917 | 0.027 | 2.901 | 2.932 | 9984.000 | 18496.000 | 0.900 | 2.602 | 2.795 | 140.163 |
| neldermead | 2.257 | 0.847 | 1.675 | 2.704 | 72000.000 | 80000.000 | 0.000 | 2.257 | 2.257 | 6.840 |
| random | 2.451 | 0.132 | 2.381 | 2.533 | 72000.000 | 80000.000 | 0.000 | 2.448 | 2.451 | 6.860 |
| spsa_nocrn | 2.892 | 0.040 | 2.866 | 2.914 | 9843.200 | 34150.400 | 0.600 | 2.841 | 2.699 | 58.068 |

### Stability (share of seeds)
| method | converged | unconverged | bound | stall | diverge |
|---|---|---|---|---|---|
| central | 0.700 | 0.000 | 0.000 | 0.300 | 0.000 |
| forward | 0.700 | 0.000 | 0.000 | 0.300 | 0.000 |
| gaussian | 0.700 | 0.000 | 0.000 | 0.300 | 0.000 |
| regression | 0.700 | 0.000 | 0.000 | 0.300 | 0.000 |
| spsa | 0.800 | 0.000 | 0.000 | 0.200 | 0.000 |
| central_nocrn | 0.700 | 0.200 | 0.000 | 0.100 | 0.000 |
| neldermead | 0.300 | 0.000 | 0.000 | 0.700 | 0.000 |
| random | 0.300 | 0.000 | 0.000 | 0.000 | 0.700 |
| spsa_nocrn | 0.300 | 0.600 | 0.000 | 0.100 | 0.000 |

### Gradient quality vs reference (rows: point x estimator)
| point | estimator | cosine | rel_l2 | sign_agree | bias_l2 | variance |
|---|---|---|---|---|---|---|
| start | forward | 0.998 | 0.089 | 0.997 | 0.117 | 0.021 |
| start | central | 0.999 | 0.073 | 1.000 | 0.073 | 0.020 |
| start | spsa | 0.585 | 1.548 | 0.653 | 0.515 | 10.376 |
| start | regression | 0.995 | 0.122 | 0.973 | 0.116 | 0.053 |
| start | gaussian | 0.721 | 1.108 | 0.723 | 0.405 | 5.319 |
| mid | forward | 0.962 | 0.326 | 0.870 | 0.150 | 0.014 |
| mid | central | 0.981 | 0.265 | 0.873 | 0.108 | 0.014 |
| mid | spsa | 0.552 | 1.507 | 0.623 | 0.235 | 0.787 |
| mid | regression | 0.973 | 0.290 | 0.843 | 0.116 | 0.017 |
| mid | gaussian | 0.713 | 1.026 | 0.697 | 0.161 | 0.378 |
| final | forward | 0.992 | 0.175 | 0.997 | 0.123 | 0.017 |
| final | central | 0.997 | 0.140 | 0.993 | 0.071 | 0.017 |
| final | spsa | 0.572 | 1.539 | 0.630 | 0.328 | 2.392 |
| final | regression | 0.990 | 0.188 | 0.937 | 0.067 | 0.033 |
| final | gaussian | 0.722 | 1.061 | 0.713 | 0.225 | 1.173 |

### Common random numbers
| estimator | crn | variance | cosine | bias_l2 |
|---|---|---|---|---|
| forward | True | 0.015 | 0.998 | 0.121 |
| forward | False | 181.838 | 0.131 | 2.202 |
| central | True | 0.014 | 0.999 | 0.067 |
| central | False | 35.354 | 0.289 | 1.683 |
| spsa | True | 9.277 | 0.605 | 0.591 |
| spsa | False | 18.062 | 0.481 | 0.589 |
| regression | True | 0.039 | 0.996 | 0.086 |
| regression | False | 14.437 | 0.487 | 0.627 |
| gaussian | True | 4.043 | 0.729 | 0.420 |
| gaussian | False | 6.394 | 0.614 | 0.404 |

| estimator | crn | final_val_mean | final_val_sd | n |
|---|---|---|---|---|
| central | True | 2.864 | 0.022 | 10 |
| central | False | 2.917 | 0.025 | 10 |
| spsa | True | 2.890 | 0.031 | 10 |
| spsa | False | 2.892 | 0.038 | 10 |

### Paired Wilcoxon tests (Holm; 52 comparisons in total, 15 significant at 0.05)
| outcome | A | B | mean_diff_A_minus_B | p_holm | rank_biserial | cohens_dz |
|---|---|---|---|---|---|---|
| final_validation_return | forward | random | 0.414 | 0.041 | 1.000 | 3.157 |
| final_validation_return | central | random | 0.413 | 0.041 | 1.000 | 3.036 |
| final_validation_return | spsa | random | 0.439 | 0.041 | 1.000 | 3.376 |
| final_validation_return | regression | random | 0.412 | 0.041 | 1.000 | 3.141 |
| final_validation_return | gaussian | random | 0.420 | 0.041 | 1.000 | 3.099 |
| gradient_cosine_start | forward | central | -0.001 | 0.000 | -1.000 | -2.401 |
| gradient_cosine_start | forward | spsa | 0.413 | 0.000 | 1.000 | 3.394 |
| gradient_cosine_start | forward | regression | 0.003 | 0.000 | 0.854 | 0.723 |
| gradient_cosine_start | forward | gaussian | 0.277 | 0.000 | 1.000 | 2.999 |
| gradient_cosine_start | central | spsa | 0.414 | 0.000 | 1.000 | 3.405 |
| gradient_cosine_start | central | regression | 0.004 | 0.000 | 0.996 | 1.049 |
| gradient_cosine_start | central | gaussian | 0.278 | 0.000 | 1.000 | 3.006 |
| gradient_cosine_start | spsa | regression | -0.410 | 0.000 | -1.000 | -3.365 |
| gradient_cosine_start | spsa | gaussian | -0.136 | 0.000 | -0.789 | -0.917 |
| gradient_cosine_start | regression | gaussian | 0.274 | 0.000 | 1.000 | 2.933 |

### Sensitivity (see CSV; plots saved)
| param | value | estimator | bias_l2 | variance | cosine | final_val_mean | final_val_sd |
|---|---|---|---|---|---|---|---|
| eps | 0.010 | forward | 0.144 | 0.015 | 0.997 |  |  |
| eps | 0.030 | forward | 0.127 | 0.014 | 0.998 |  |  |
| eps | 0.100 | forward | 0.126 | 0.012 | 0.998 |  |  |
| eps | 0.300 | forward | 0.247 | 0.010 | 0.995 |  |  |
| eps | 1.000 | forward | 0.598 | 0.009 | 0.966 |  |  |
| eps | 0.010 | central | 0.080 | 0.013 | 0.999 |  |  |
| eps | 0.030 | central | 0.049 | 0.013 | 0.999 |  |  |
| eps | 0.100 | central | 0.089 | 0.012 | 0.999 |  |  |
| eps | 0.300 | central | 0.163 | 0.010 | 0.997 |  |  |
| eps | 1.000 | central | 0.456 | 0.008 | 0.982 |  |  |
| eps | 0.010 | spsa | 0.557 | 7.888 | 0.549 |  |  |
| eps | 0.030 | spsa | 0.560 | 7.957 | 0.549 |  |  |
| eps | 0.100 | spsa | 0.586 | 8.477 | 0.547 |  |  |
| eps | 0.300 | spsa | 0.659 | 8.851 | 0.539 |  |  |
| eps | 1.000 | spsa | 0.741 | 5.607 | 0.533 |  |  |
| N_paths | 8.000 | central | 0.090 | 0.153 | 0.995 |  |  |
| N_paths | 16.000 | central | 0.089 | 0.060 | 0.997 |  |  |
| N_paths | 32.000 | central | 0.075 | 0.023 | 0.998 |  |  |
| N_paths | 64.000 | central | 0.068 | 0.014 | 0.999 |  |  |
| N_paths | 128.000 | central | 0.067 | 0.006 | 0.999 |  |  |
| N_paths | 8.000 | spsa | 0.759 | 8.934 | 0.590 |  |  |
| N_paths | 16.000 | spsa | 0.502 | 9.427 | 0.550 |  |  |
| N_paths | 32.000 | spsa | 0.515 | 9.346 | 0.558 |  |  |
| N_paths | 64.000 | spsa | 0.860 | 7.205 | 0.564 |  |  |
| N_paths | 128.000 | spsa | 0.419 | 8.336 | 0.609 |  |  |
| N_paths | 8.000 | regression | 0.141 | 0.154 | 0.993 |  |  |
| N_paths | 16.000 | regression | 0.120 | 0.080 | 0.995 |  |  |
| N_paths | 32.000 | regression | 0.118 | 0.048 | 0.995 |  |  |
| N_paths | 64.000 | regression | 0.113 | 0.038 | 0.995 |  |  |
| N_paths | 128.000 | regression | 0.113 | 0.032 | 0.996 |  |  |
| spsa_m | 1.000 | spsa | 1.442 | 47.119 | 0.327 |  |  |
| spsa_m | 2.000 | spsa | 0.662 | 21.002 | 0.423 |  |  |
| spsa_m | 4.000 | spsa | 0.615 | 8.752 | 0.548 |  |  |
| spsa_m | 8.000 | spsa | 0.384 | 5.146 | 0.677 |  |  |
| spsa_m | 16.000 | spsa | 0.311 | 2.395 | 0.790 |  |  |
| regression_K | 10.000 | regression | 0.626 | 19.096 | 0.776 |  |  |
| regression_K | 20.000 | regression | 0.111 | 0.042 | 0.996 |  |  |
| regression_K | 40.000 | regression | 0.111 | 0.022 | 0.998 |  |  |
| regression_K | 80.000 | regression | 0.111 | 0.018 | 0.998 |  |  |
| step_size_a_mult | 0.250 | central |  |  |  | 2.497 | 0.272 |
| step_size_a_mult | 1.000 | central |  |  |  | 2.864 | 0.013 |
| step_size_a_mult | 4.000 | central |  |  |  | 2.846 | 0.017 |
| param_scale_mult | 0.250 | central |  |  |  | 1.597 | 0.782 |
| param_scale_mult | 1.000 | central |  |  |  | 2.864 | 0.013 |
| param_scale_mult | 4.000 | central |  |  |  | 2.658 | 0.069 |

## inventory__HOBBIES_1_371_WI_1_validation__I1
tuned hyperparameters (validation only): eps=0.4, a=0.5; budget=40000 rollouts; seeds=10

### Final validation return, sample efficiency, runtime
| method | final_val_mean | final_val_sd | final_val_ci_lo | final_val_ci_hi | rollouts_to_90pct_mean | rollouts_to_95pct_mean | reached_95pct_share | val@25%budget | val@100%budget | runtime_per_iter_ms |
|---|---|---|---|---|---|---|---|---|---|---|
| central | 3.048 | 0.000 | 3.048 | 3.048 | 1126.400 | 1510.400 | 1.000 | 2.929 | 2.729 | 13.328 |
| forward | 3.048 | 0.000 | 3.048 | 3.048 | 1152.000 | 1536.000 | 1.000 | 2.940 | 2.743 | 13.227 |
| gaussian | 3.048 | 0.000 | 3.048 | 3.048 | 2560.000 | 3276.800 | 1.000 | 3.017 | 2.825 | 22.131 |
| regression | 3.048 | 0.000 | 3.048 | 3.048 | 1689.600 | 2227.200 | 1.000 | 2.984 | 2.779 | 17.783 |
| spsa | 3.048 | 0.000 | 3.048 | 3.048 | 4710.400 | 6144.000 | 1.000 | 3.010 | 2.930 | 39.568 |
| central_nocrn | 3.048 | 0.000 | 3.048 | 3.048 | 1177.600 | 1561.600 | 1.000 | 2.943 | 2.721 | 13.329 |
| grid | 2.600 | 0.030 | 2.583 | 2.618 | 13107.200 | 14745.600 | 1.000 | 2.047 | 2.600 | 4.644 |
| neldermead | 2.716 | 0.241 | 2.576 | 2.857 | 16313.600 | 40179.200 | 0.500 | 2.716 | 2.716 | 4.792 |
| random | 2.600 | 0.030 | 2.583 | 2.619 | 48025.600 | 56032.000 | 0.300 | 2.600 | 2.600 | 4.829 |
| spsa_nocrn | 3.048 | 0.000 | 3.048 | 3.048 | 4608.000 | 6041.600 | 1.000 | 3.011 | 2.927 | 42.242 |

### Stability (share of seeds)
| method | converged | unconverged | bound | stall | diverge |
|---|---|---|---|---|---|
| central | 0.900 | 0.000 | 0.000 | 0.100 | 0.000 |
| forward | 0.900 | 0.000 | 0.000 | 0.100 | 0.000 |
| gaussian | 0.700 | 0.200 | 0.000 | 0.100 | 0.000 |
| regression | 0.800 | 0.100 | 0.000 | 0.100 | 0.000 |
| spsa | 0.900 | 0.000 | 0.000 | 0.100 | 0.000 |
| central_nocrn | 0.600 | 0.300 | 0.000 | 0.100 | 0.000 |
| grid | 0.000 | 0.500 | 0.000 | 0.200 | 0.300 |
| neldermead | 0.500 | 0.000 | 0.200 | 0.300 | 0.000 |
| random | 0.500 | 0.000 | 0.000 | 0.200 | 0.300 |
| spsa_nocrn | 0.700 | 0.200 | 0.000 | 0.100 | 0.000 |

### Gradient quality vs reference (rows: point x estimator)
| point | estimator | cosine | rel_l2 | sign_agree | bias_l2 | variance |
|---|---|---|---|---|---|---|
| start | forward | 1.000 | 0.040 | 1.000 | 0.010 | 0.004 |
| start | central | 1.000 | 0.051 | 1.000 | 0.034 | 0.004 |
| start | spsa | 1.000 | 0.032 | 1.000 | 0.027 | 0.001 |
| start | regression | 1.000 | 0.063 | 1.000 | 0.045 | 0.006 |
| start | gaussian | 1.000 | 0.547 | 1.000 | 0.141 | 0.656 |
| start | dense_grid_derivative | nan | nan | nan | 0.011 | 0.000 |
| mid | forward | 1.000 | 0.142 | 1.000 | 0.026 | 0.002 |
| mid | central | 1.000 | 0.121 | 1.000 | 0.021 | 0.002 |
| mid | spsa | 1.000 | 0.084 | 1.000 | 0.014 | 0.001 |
| mid | regression | 1.000 | 0.185 | 1.000 | 0.023 | 0.004 |
| mid | gaussian | 1.000 | 0.585 | 1.000 | 0.021 | 0.049 |
| mid | dense_grid_derivative | nan | nan | nan | 0.004 | 0.000 |
| final | forward | 1.000 | 0.060 | 1.000 | 0.036 | 0.003 |
| final | central | 1.000 | 0.048 | 1.000 | 0.009 | 0.003 |
| final | spsa | 1.000 | 0.030 | 1.000 | 0.006 | 0.001 |
| final | regression | 1.000 | 0.068 | 1.000 | 0.014 | 0.005 |
| final | gaussian | 1.000 | 0.542 | 1.000 | 0.112 | 0.319 |
| final | dense_grid_derivative | nan | nan | nan | 0.004 | 0.000 |

### Common random numbers
| estimator | crn | variance | cosine | bias_l2 |
|---|---|---|---|---|
| forward | True | 0.008 | 1.000 | 0.020 |
| forward | False | 0.232 | 0.933 | 0.017 |
| central | True | 0.008 | 1.000 | 0.022 |
| central | False | 0.049 | 1.000 | 0.004 |
| spsa | True | 0.002 | 1.000 | 0.007 |
| spsa | False | 0.008 | 1.000 | 0.012 |
| regression | True | 0.010 | 1.000 | 0.041 |
| regression | False | 0.396 | 0.933 | 0.110 |
| gaussian | True | 1.927 | 1.000 | 0.162 |
| gaussian | False | 1.472 | 1.000 | 0.122 |

| estimator | crn | final_val_mean | final_val_sd | n |
|---|---|---|---|---|
| central | True | 3.048 | 0.000 | 10 |
| central | False | 3.048 | 0.000 | 10 |
| spsa | True | 3.048 | 0.000 | 10 |
| spsa | False | 3.048 | 0.000 | 10 |

### Optimality gap vs exhaustive grid on validation paths
| method | gap_mean | gap_sd | grid_best_val |
|---|---|---|---|
| central | 0.000 | 0.000 | 3.048 |
| central_nocrn | 0.000 | 0.000 | 3.048 |
| forward | 0.000 | 0.000 | 3.048 |
| gaussian | 0.000 | 0.000 | 3.048 |
| grid | 0.448 | 0.030 | 3.048 |
| neldermead | 0.333 | 0.241 | 3.048 |
| random | 0.448 | 0.030 | 3.048 |
| regression | 0.000 | 0.000 | 3.048 |
| spsa | 0.000 | 0.000 | 3.048 |
| spsa_nocrn | 0.000 | 0.000 | 3.048 |

### Paired Wilcoxon tests (Holm; 52 comparisons in total, 10 significant at 0.05)
| outcome | A | B | mean_diff_A_minus_B | p_holm | rank_biserial | cohens_dz |
|---|---|---|---|---|---|---|
| final_validation_return | forward | random | 0.448 | 0.041 | 1.000 | 14.787 |
| final_validation_return | forward | neldermead | 0.332 | 0.041 | 1.000 | 1.379 |
| final_validation_return | central | random | 0.448 | 0.041 | 1.000 | 14.830 |
| final_validation_return | central | neldermead | 0.332 | 0.041 | 1.000 | 1.380 |
| final_validation_return | spsa | random | 0.448 | 0.041 | 1.000 | 14.813 |
| final_validation_return | spsa | neldermead | 0.332 | 0.041 | 1.000 | 1.379 |
| final_validation_return | regression | random | 0.448 | 0.041 | 1.000 | 14.846 |
| final_validation_return | regression | neldermead | 0.332 | 0.041 | 1.000 | 1.380 |
| final_validation_return | gaussian | random | 0.448 | 0.041 | 1.000 | 14.781 |
| final_validation_return | gaussian | neldermead | 0.332 | 0.041 | 1.000 | 1.379 |

### Sensitivity (see CSV; plots saved)
| param | value | estimator | bias_l2 | variance | cosine | final_val_mean | final_val_sd |
|---|---|---|---|---|---|---|---|
| eps | 0.010 | forward | 0.034 | 0.005 | 1.000 |  |  |
| eps | 0.030 | forward | 0.003 | 0.005 | 1.000 |  |  |
| eps | 0.100 | forward | 0.024 | 0.005 | 1.000 |  |  |
| eps | 0.300 | forward | 0.026 | 0.005 | 1.000 |  |  |
| eps | 1.000 | forward | 0.092 | 0.004 | 1.000 |  |  |
| eps | 0.010 | central | 0.034 | 0.005 | 1.000 |  |  |
| eps | 0.030 | central | 0.019 | 0.005 | 1.000 |  |  |
| eps | 0.100 | central | 0.038 | 0.005 | 1.000 |  |  |
| eps | 0.300 | central | 0.007 | 0.005 | 1.000 |  |  |
| eps | 1.000 | central | 0.036 | 0.006 | 1.000 |  |  |
| eps | 0.010 | spsa | 0.032 | 0.002 | 1.000 |  |  |
| eps | 0.030 | spsa | 0.018 | 0.002 | 1.000 |  |  |
| eps | 0.100 | spsa | 0.036 | 0.002 | 1.000 |  |  |
| eps | 0.300 | spsa | 0.005 | 0.002 | 1.000 |  |  |
| eps | 1.000 | spsa | 0.033 | 0.002 | 1.000 |  |  |
| N_paths | 8.000 | central | 0.011 | 0.059 | 1.000 |  |  |
| N_paths | 16.000 | central | 0.022 | 0.031 | 1.000 |  |  |
| N_paths | 32.000 | central | 0.017 | 0.015 | 1.000 |  |  |
| N_paths | 64.000 | central | 0.016 | 0.008 | 1.000 |  |  |
| N_paths | 128.000 | central | 0.004 | 0.003 | 1.000 |  |  |
| N_paths | 8.000 | spsa | 0.021 | 0.014 | 1.000 |  |  |
| N_paths | 16.000 | spsa | 0.015 | 0.007 | 1.000 |  |  |
| N_paths | 32.000 | spsa | 0.000 | 0.003 | 1.000 |  |  |
| N_paths | 64.000 | spsa | 0.002 | 0.001 | 1.000 |  |  |
| N_paths | 128.000 | spsa | 0.007 | 0.001 | 1.000 |  |  |
| N_paths | 8.000 | regression | 0.015 | 0.064 | 1.000 |  |  |
| N_paths | 16.000 | regression | 0.031 | 0.032 | 1.000 |  |  |
| N_paths | 32.000 | regression | 0.017 | 0.014 | 1.000 |  |  |
| N_paths | 64.000 | regression | 0.009 | 0.011 | 1.000 |  |  |
| N_paths | 128.000 | regression | 0.016 | 0.005 | 1.000 |  |  |
| spsa_m | 1.000 | spsa | 0.014 | 0.009 | 1.000 |  |  |
| spsa_m | 2.000 | spsa | 0.018 | 0.004 | 1.000 |  |  |
| spsa_m | 4.000 | spsa | 0.026 | 0.002 | 1.000 |  |  |
| spsa_m | 8.000 | spsa | 0.014 | 0.001 | 1.000 |  |  |
| spsa_m | 16.000 | spsa | 0.010 | 0.001 | 1.000 |  |  |
| regression_K | 1.000 | regression | 0.024 | 0.006 | 1.000 |  |  |
| regression_K | 2.000 | regression | 0.021 | 0.006 | 1.000 |  |  |
| regression_K | 4.000 | regression | 0.021 | 0.007 | 1.000 |  |  |
| regression_K | 8.000 | regression | 0.037 | 0.007 | 1.000 |  |  |
| step_size_a_mult | 0.250 | central |  |  |  | 2.924 | 0.100 |
| step_size_a_mult | 1.000 | central |  |  |  | 3.048 | 0.000 |
| step_size_a_mult | 4.000 | central |  |  |  | 3.045 | 0.004 |
| param_scale_mult | 0.250 | central |  |  |  | 1.774 | 0.750 |
| param_scale_mult | 1.000 | central |  |  |  | 3.048 | 0.000 |
| param_scale_mult | 4.000 | central |  |  |  | 2.973 | 0.055 |

## inventory__HOBBIES_1_371_WI_1_validation__I2
tuned hyperparameters (validation only): eps=0.05, a=2.0; budget=40000 rollouts; seeds=10

### Final validation return, sample efficiency, runtime
| method | final_val_mean | final_val_sd | final_val_ci_lo | final_val_ci_hi | rollouts_to_90pct_mean | rollouts_to_95pct_mean | reached_95pct_share | val@25%budget | val@100%budget | runtime_per_iter_ms |
|---|---|---|---|---|---|---|---|---|---|---|
| central | 3.345 | 0.054 | 3.314 | 3.376 | 8716.800 | 48460.800 | 0.400 | 3.034 | 2.964 | 26.365 |
| forward | 3.362 | 0.067 | 3.323 | 3.401 | 8537.600 | 24576.000 | 0.700 | 3.001 | 2.947 | 21.325 |
| gaussian | 3.356 | 0.089 | 3.304 | 3.406 | 17024.000 | 32870.400 | 0.600 | 3.047 | 2.951 | 26.357 |
| regression | 3.356 | 0.041 | 3.333 | 3.381 | 1088.000 | 25088.000 | 0.700 | 3.063 | 2.965 | 31.509 |
| spsa | 3.334 | 0.075 | 3.291 | 3.381 | 17433.600 | 48819.200 | 0.400 | 3.112 | 2.970 | 46.698 |
| central_nocrn | 3.356 | 0.087 | 3.304 | 3.405 | 18304.000 | 33843.200 | 0.600 | 2.866 | 2.884 | 26.717 |
| grid | 2.975 | 0.052 | 2.948 | 3.007 | 13120.000 | 13120.000 | 1.000 | 3.065 | 2.975 | 5.487 |
| neldermead | 2.454 | 0.812 | 1.927 | 2.877 | 80000.000 | 80000.000 | 0.000 | 2.454 | 2.454 | 5.812 |
| random | 2.978 | 0.045 | 2.953 | 3.006 | 48057.600 | 64025.600 | 0.200 | 2.945 | 2.978 | 5.718 |
| spsa_nocrn | 3.390 | 0.063 | 3.353 | 3.425 | 12505.600 | 28505.600 | 0.700 | 3.172 | 2.977 | 47.964 |

### Stability (share of seeds)
| method | converged | unconverged | bound | stall | diverge |
|---|---|---|---|---|---|
| central | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| forward | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| gaussian | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| regression | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| spsa | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| central_nocrn | 0.000 | 0.800 | 0.200 | 0.000 | 0.000 |
| grid | 0.000 | 0.700 | 0.000 | 0.100 | 0.200 |
| neldermead | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| random | 0.300 | 0.400 | 0.000 | 0.100 | 0.200 |
| spsa_nocrn | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |

### Gradient quality vs reference (rows: point x estimator)
| point | estimator | cosine | rel_l2 | sign_agree | bias_l2 | variance |
|---|---|---|---|---|---|---|
| start | forward | 1.000 | 0.043 | 1.000 | 0.022 | 0.005 |
| start | central | 1.000 | 0.057 | 1.000 | 0.058 | 0.005 |
| start | spsa | 0.937 | 0.338 | 0.867 | 0.072 | 0.408 |
| start | regression | 0.996 | 0.101 | 1.000 | 0.076 | 0.022 |
| start | gaussian | 0.814 | 0.840 | 0.800 | 0.093 | 2.430 |
| mid | forward | -0.458 | 13.749 | 0.517 | 0.463 | 0.078 |
| mid | central | -0.342 | 6.867 | 0.533 | 0.230 | 0.020 |
| mid | spsa | -0.394 | 6.215 | 0.333 | 0.197 | 0.012 |
| mid | regression | 0.149 | 5.466 | 0.550 | 0.145 | 0.063 |
| mid | gaussian | 0.017 | 3.825 | 0.467 | 0.104 | 0.018 |
| final | forward | 1.000 | 0.074 | 1.000 | 0.046 | 0.004 |
| final | central | 1.000 | 0.072 | 1.000 | 0.041 | 0.004 |
| final | spsa | 0.951 | 0.377 | 0.983 | 0.064 | 0.186 |
| final | regression | 0.953 | 0.252 | 0.950 | 0.145 | 0.103 |
| final | gaussian | 0.776 | 0.952 | 0.783 | 0.262 | 1.061 |

### Common random numbers
| estimator | crn | variance | cosine | bias_l2 |
|---|---|---|---|---|
| forward | True | 0.008 | 1.000 | 0.042 |
| forward | False | 15.045 | 0.422 | 0.342 |
| central | True | 0.010 | 1.000 | 0.079 |
| central | False | 3.763 | 0.518 | 0.266 |
| spsa | True | 0.388 | 0.924 | 0.171 |
| spsa | False | 1.127 | 0.835 | 0.042 |
| regression | True | 0.043 | 0.995 | 0.120 |
| regression | False | 11.347 | 0.621 | 0.581 |
| gaussian | True | 3.815 | 0.782 | 0.190 |
| gaussian | False | 3.270 | 0.513 | 0.319 |

| estimator | crn | final_val_mean | final_val_sd | n |
|---|---|---|---|---|
| central | True | 3.345 | 0.052 | 10 |
| central | False | 3.356 | 0.083 | 10 |
| spsa | True | 3.334 | 0.071 | 10 |
| spsa | False | 3.390 | 0.060 | 10 |

### Optimality gap vs exhaustive grid on validation paths
| method | gap_mean | gap_sd | grid_best_val |
|---|---|---|---|
| central | 0.134 | 0.054 | 3.480 |
| central_nocrn | 0.124 | 0.087 | 3.480 |
| forward | 0.118 | 0.067 | 3.480 |
| gaussian | 0.123 | 0.089 | 3.480 |
| grid | 0.505 | 0.052 | 3.480 |
| neldermead | 1.026 | 0.812 | 3.480 |
| random | 0.501 | 0.045 | 3.480 |
| regression | 0.124 | 0.041 | 3.480 |
| spsa | 0.146 | 0.075 | 3.480 |
| spsa_nocrn | 0.090 | 0.063 | 3.480 |

### Paired Wilcoxon tests (Holm; 52 comparisons in total, 20 significant at 0.05)
| outcome | A | B | mean_diff_A_minus_B | p_holm | rank_biserial | cohens_dz |
|---|---|---|---|---|---|---|
| final_validation_return | forward | random | 0.383 | 0.041 | 1.000 | 3.904 |
| final_validation_return | forward | neldermead | 0.908 | 0.041 | 1.000 | 1.087 |
| final_validation_return | central | random | 0.367 | 0.041 | 1.000 | 4.133 |
| final_validation_return | central | neldermead | 0.891 | 0.041 | 1.000 | 1.084 |
| final_validation_return | spsa | random | 0.356 | 0.041 | 1.000 | 3.520 |
| final_validation_return | spsa | neldermead | 0.880 | 0.041 | 1.000 | 1.041 |
| final_validation_return | regression | random | 0.377 | 0.041 | 1.000 | 4.922 |
| final_validation_return | regression | neldermead | 0.902 | 0.041 | 1.000 | 1.102 |
| final_validation_return | gaussian | random | 0.378 | 0.041 | 1.000 | 3.031 |
| final_validation_return | gaussian | neldermead | 0.903 | 0.041 | 1.000 | 1.088 |
| gradient_cosine_start | forward | central | 0.000 | 0.000 | 0.944 | 1.249 |
| gradient_cosine_start | forward | spsa | 0.063 | 0.000 | 0.961 | 0.582 |
| gradient_cosine_start | forward | regression | 0.004 | 0.000 | 0.918 | 0.495 |
| gradient_cosine_start | forward | gaussian | 0.186 | 0.000 | 1.000 | 0.812 |
| gradient_cosine_start | central | spsa | 0.062 | 0.000 | 0.888 | 0.580 |
| gradient_cosine_start | central | regression | 0.004 | 0.000 | 0.759 | 0.476 |
| gradient_cosine_start | central | gaussian | 0.185 | 0.000 | 0.996 | 0.812 |
| gradient_cosine_start | spsa | regression | -0.059 | 0.001 | -0.686 | -0.548 |
| gradient_cosine_start | spsa | gaussian | 0.123 | 0.003 | 0.596 | 0.475 |
| gradient_cosine_start | regression | gaussian | 0.182 | 0.000 | 0.940 | 0.791 |

### Sensitivity (see CSV; plots saved)
| param | value | estimator | bias_l2 | variance | cosine | final_val_mean | final_val_sd |
|---|---|---|---|---|---|---|---|
| eps | 0.010 | forward | 0.321 | 0.032 | 0.998 |  |  |
| eps | 0.030 | forward | 0.020 | 0.009 | 1.000 |  |  |
| eps | 0.100 | forward | 0.286 | 0.012 | 0.981 |  |  |
| eps | 0.300 | forward | 0.109 | 0.007 | 0.999 |  |  |
| eps | 1.000 | forward | 0.176 | 0.006 | 1.000 |  |  |
| eps | 0.010 | central | 0.118 | 0.013 | 1.000 |  |  |
| eps | 0.030 | central | 0.060 | 0.009 | 1.000 |  |  |
| eps | 0.100 | central | 0.167 | 0.007 | 0.997 |  |  |
| eps | 0.300 | central | 0.115 | 0.007 | 0.998 |  |  |
| eps | 1.000 | central | 0.111 | 0.007 | 0.998 |  |  |
| eps | 0.010 | spsa | 0.144 | 0.685 | 0.904 |  |  |
| eps | 0.030 | spsa | 0.070 | 0.532 | 0.906 |  |  |
| eps | 0.100 | spsa | 0.131 | 0.467 | 0.899 |  |  |
| eps | 0.300 | spsa | 0.071 | 0.505 | 0.902 |  |  |
| eps | 1.000 | spsa | 0.052 | 0.522 | 0.902 |  |  |
| N_paths | 8.000 | central | 0.106 | 0.062 | 0.999 |  |  |
| N_paths | 16.000 | central | 0.072 | 0.035 | 1.000 |  |  |
| N_paths | 32.000 | central | 0.086 | 0.016 | 1.000 |  |  |
| N_paths | 64.000 | central | 0.089 | 0.009 | 1.000 |  |  |
| N_paths | 128.000 | central | 0.095 | 0.003 | 1.000 |  |  |
| N_paths | 8.000 | spsa | 0.118 | 0.583 | 0.901 |  |  |
| N_paths | 16.000 | spsa | 0.153 | 0.238 | 0.963 |  |  |
| N_paths | 32.000 | spsa | 0.229 | 0.585 | 0.849 |  |  |
| N_paths | 64.000 | spsa | 0.148 | 0.324 | 0.940 |  |  |
| N_paths | 128.000 | spsa | 0.292 | 0.436 | 0.885 |  |  |
| N_paths | 8.000 | regression | 0.106 | 0.183 | 0.986 |  |  |
| N_paths | 16.000 | regression | 0.090 | 0.051 | 0.997 |  |  |
| N_paths | 32.000 | regression | 0.092 | 0.029 | 0.997 |  |  |
| N_paths | 64.000 | regression | 0.116 | 0.023 | 0.997 |  |  |
| N_paths | 128.000 | regression | 0.114 | 0.029 | 0.997 |  |  |
| spsa_m | 1.000 | spsa | 0.676 | 1.306 | 0.792 |  |  |
| spsa_m | 2.000 | spsa | 0.274 | 0.936 | 0.866 |  |  |
| spsa_m | 4.000 | spsa | 0.072 | 0.437 | 0.919 |  |  |
| spsa_m | 8.000 | spsa | 0.077 | 0.222 | 0.966 |  |  |
| spsa_m | 16.000 | spsa | 0.082 | 0.112 | 0.983 |  |  |
| regression_K | 2.000 | regression | 0.082 | 0.849 | 0.861 |  |  |
| regression_K | 4.000 | regression | 0.087 | 0.014 | 0.998 |  |  |
| regression_K | 8.000 | regression | 0.099 | 0.012 | 0.998 |  |  |
| regression_K | 16.000 | regression | 0.095 | 0.009 | 0.998 |  |  |
| step_size_a_mult | 0.250 | central |  |  |  | 3.348 | 0.062 |
| step_size_a_mult | 1.000 | central |  |  |  | 3.394 | 0.067 |
| step_size_a_mult | 4.000 | central |  |  |  | 3.275 | 0.026 |
| param_scale_mult | 0.250 | central |  |  |  | 2.842 | 0.413 |
| param_scale_mult | 1.000 | central |  |  |  | 3.394 | 0.067 |
| param_scale_mult | 4.000 | central |  |  |  | 3.001 | 0.032 |

## inventory__HOBBIES_1_371_WI_1_validation__I3
tuned hyperparameters (validation only): eps=0.05, a=2.0; budget=40000 rollouts; seeds=10

### Final validation return, sample efficiency, runtime
| method | final_val_mean | final_val_sd | final_val_ci_lo | final_val_ci_hi | rollouts_to_90pct_mean | rollouts_to_95pct_mean | reached_95pct_share | val@25%budget | val@100%budget | runtime_per_iter_ms |
|---|---|---|---|---|---|---|---|---|---|---|
| central | 3.165 | 0.072 | 3.123 | 3.209 | 42560.000 | 72768.000 | 0.100 | 3.157 | 3.003 | 140.325 |
| forward | 3.167 | 0.067 | 3.128 | 3.207 | 41408.000 | 64704.000 | 0.200 | 3.055 | 3.006 | 78.902 |
| gaussian | 3.183 | 0.078 | 3.140 | 3.230 | 42304.000 | 57792.000 | 0.300 | 3.159 | 2.990 | 134.063 |
| regression | 3.160 | 0.068 | 3.122 | 3.199 | 42688.000 | 72537.600 | 0.100 | 3.147 | 3.007 | 140.195 |
| spsa | 3.192 | 0.075 | 3.151 | 3.234 | 33126.400 | 56512.000 | 0.300 | 2.992 | 2.946 | 59.986 |
| central_nocrn | 3.226 | 0.110 | 3.159 | 3.290 | 34560.000 | 37120.000 | 0.600 | 3.084 | 2.973 | 132.208 |
| neldermead | 2.103 | 1.073 | 1.413 | 2.668 | 80000.000 | 80000.000 | 0.000 | 2.103 | 2.103 | 6.832 |
| random | 2.971 | 0.087 | 2.920 | 3.020 | 72012.800 | 72012.800 | 0.100 | 2.896 | 2.971 | 6.959 |
| spsa_nocrn | 3.290 | 0.051 | 3.259 | 3.318 | 1945.600 | 17843.200 | 0.800 | 3.063 | 2.949 | 57.904 |

### Stability (share of seeds)
| method | converged | unconverged | bound | stall | diverge |
|---|---|---|---|---|---|
| central | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| forward | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| gaussian | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| regression | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| spsa | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| central_nocrn | 0.000 | 0.700 | 0.200 | 0.100 | 0.000 |
| neldermead | 0.100 | 0.000 | 0.000 | 0.900 | 0.000 |
| random | 0.300 | 0.500 | 0.000 | 0.100 | 0.100 |
| spsa_nocrn | 0.000 | 0.700 | 0.300 | 0.000 | 0.000 |

### Gradient quality vs reference (rows: point x estimator)
| point | estimator | cosine | rel_l2 | sign_agree | bias_l2 | variance |
|---|---|---|---|---|---|---|
| start | forward | 0.978 | 0.216 | 0.887 | 0.296 | 0.009 |
| start | central | 0.986 | 0.174 | 0.897 | 0.235 | 0.008 |
| start | spsa | 0.563 | 1.542 | 0.623 | 0.290 | 5.339 |
| start | regression | 0.983 | 0.191 | 0.930 | 0.242 | 0.018 |
| start | gaussian | 0.728 | 1.047 | 0.723 | 0.283 | 2.476 |
| mid | forward | 0.907 | 0.487 | 0.703 | 0.076 | 0.003 |
| mid | central | 0.955 | 0.385 | 0.717 | 0.053 | 0.003 |
| mid | spsa | 0.487 | 1.928 | 0.617 | 0.094 | 0.147 |
| mid | regression | 0.917 | 0.517 | 0.703 | 0.062 | 0.007 |
| mid | gaussian | 0.703 | 1.303 | 0.653 | 0.078 | 0.071 |
| final | forward | 0.993 | 0.132 | 0.923 | 0.078 | 0.006 |
| final | central | 0.996 | 0.116 | 0.890 | 0.063 | 0.005 |
| final | spsa | 0.546 | 1.606 | 0.617 | 0.236 | 1.712 |
| final | regression | 0.992 | 0.151 | 0.823 | 0.087 | 0.008 |
| final | gaussian | 0.728 | 1.082 | 0.643 | 0.196 | 0.779 |

### Common random numbers
| estimator | crn | variance | cosine | bias_l2 |
|---|---|---|---|---|
| forward | True | 0.013 | 0.978 | 0.292 |
| forward | False | 99.305 | 0.159 | 1.340 |
| central | True | 0.012 | 0.986 | 0.233 |
| central | False | 23.430 | 0.222 | 0.909 |
| spsa | True | 4.790 | 0.580 | 0.486 |
| spsa | False | 11.243 | 0.449 | 0.654 |
| regression | True | 0.029 | 0.983 | 0.230 |
| regression | False | 9.053 | 0.500 | 0.602 |
| gaussian | True | 2.003 | 0.719 | 0.297 |
| gaussian | False | 3.803 | 0.583 | 0.323 |

| estimator | crn | final_val_mean | final_val_sd | n |
|---|---|---|---|---|
| central | True | 3.165 | 0.069 | 10 |
| central | False | 3.226 | 0.104 | 10 |
| spsa | True | 3.192 | 0.071 | 10 |
| spsa | False | 3.290 | 0.048 | 10 |

### Paired Wilcoxon tests (Holm; 52 comparisons in total, 20 significant at 0.05)
| outcome | A | B | mean_diff_A_minus_B | p_holm | rank_biserial | cohens_dz |
|---|---|---|---|---|---|---|
| final_validation_return | forward | random | 0.197 | 0.041 | 1.000 | 2.159 |
| final_validation_return | forward | neldermead | 1.064 | 0.041 | 1.000 | 0.938 |
| final_validation_return | central | random | 0.194 | 0.041 | 1.000 | 2.005 |
| final_validation_return | central | neldermead | 1.062 | 0.041 | 1.000 | 0.934 |
| final_validation_return | spsa | random | 0.221 | 0.041 | 1.000 | 1.846 |
| final_validation_return | spsa | neldermead | 1.089 | 0.041 | 1.000 | 0.985 |
| final_validation_return | regression | random | 0.189 | 0.047 | 0.964 | 1.961 |
| final_validation_return | regression | neldermead | 1.057 | 0.041 | 1.000 | 0.933 |
| final_validation_return | gaussian | random | 0.212 | 0.041 | 1.000 | 1.800 |
| final_validation_return | gaussian | neldermead | 1.080 | 0.041 | 1.000 | 0.962 |
| gradient_cosine_start | forward | central | -0.008 | 0.000 | -1.000 | -6.378 |
| gradient_cosine_start | forward | spsa | 0.415 | 0.000 | 1.000 | 4.169 |
| gradient_cosine_start | forward | regression | -0.005 | 0.000 | -0.802 | -0.983 |
| gradient_cosine_start | forward | gaussian | 0.250 | 0.000 | 1.000 | 2.577 |
| gradient_cosine_start | central | spsa | 0.423 | 0.000 | 1.000 | 4.254 |
| gradient_cosine_start | central | regression | 0.003 | 0.002 | 0.626 | 0.696 |
| gradient_cosine_start | central | gaussian | 0.258 | 0.000 | 1.000 | 2.680 |
| gradient_cosine_start | spsa | regression | -0.420 | 0.000 | -1.000 | -4.225 |
| gradient_cosine_start | spsa | gaussian | -0.165 | 0.000 | -0.880 | -1.124 |
| gradient_cosine_start | regression | gaussian | 0.255 | 0.000 | 1.000 | 2.654 |

### Sensitivity (see CSV; plots saved)
| param | value | estimator | bias_l2 | variance | cosine | final_val_mean | final_val_sd |
|---|---|---|---|---|---|---|---|
| eps | 0.010 | forward | 0.517 | 0.024 | 0.927 |  |  |
| eps | 0.030 | forward | 0.297 | 0.015 | 0.977 |  |  |
| eps | 0.100 | forward | 0.259 | 0.012 | 0.983 |  |  |
| eps | 0.300 | forward | 0.275 | 0.011 | 0.981 |  |  |
| eps | 1.000 | forward | 0.396 | 0.009 | 0.961 |  |  |
| eps | 0.010 | central | 0.273 | 0.016 | 0.979 |  |  |
| eps | 0.030 | central | 0.165 | 0.012 | 0.992 |  |  |
| eps | 0.100 | central | 0.234 | 0.009 | 0.986 |  |  |
| eps | 0.300 | central | 0.248 | 0.009 | 0.984 |  |  |
| eps | 1.000 | central | 0.249 | 0.008 | 0.984 |  |  |
| eps | 0.010 | spsa | 0.439 | 4.055 | 0.529 |  |  |
| eps | 0.030 | spsa | 0.592 | 4.430 | 0.517 |  |  |
| eps | 0.100 | spsa | 0.607 | 4.251 | 0.518 |  |  |
| eps | 0.300 | spsa | 0.589 | 3.890 | 0.520 |  |  |
| eps | 1.000 | spsa | 0.597 | 2.820 | 0.522 |  |  |
| N_paths | 8.000 | central | 0.232 | 0.106 | 0.982 |  |  |
| N_paths | 16.000 | central | 0.228 | 0.055 | 0.985 |  |  |
| N_paths | 32.000 | central | 0.229 | 0.026 | 0.986 |  |  |
| N_paths | 64.000 | central | 0.232 | 0.014 | 0.986 |  |  |
| N_paths | 128.000 | central | 0.227 | 0.005 | 0.987 |  |  |
| N_paths | 8.000 | spsa | 0.585 | 4.650 | 0.542 |  |  |
| N_paths | 16.000 | spsa | 0.370 | 4.896 | 0.544 |  |  |
| N_paths | 32.000 | spsa | 0.395 | 4.472 | 0.575 |  |  |
| N_paths | 64.000 | spsa | 0.669 | 3.736 | 0.577 |  |  |
| N_paths | 128.000 | spsa | 0.317 | 4.604 | 0.598 |  |  |
| N_paths | 8.000 | regression | 0.218 | 0.126 | 0.977 |  |  |
| N_paths | 16.000 | regression | 0.240 | 0.077 | 0.979 |  |  |
| N_paths | 32.000 | regression | 0.244 | 0.045 | 0.981 |  |  |
| N_paths | 64.000 | regression | 0.247 | 0.027 | 0.982 |  |  |
| N_paths | 128.000 | regression | 0.239 | 0.019 | 0.983 |  |  |
| spsa_m | 1.000 | spsa | 1.016 | 19.005 | 0.283 |  |  |
| spsa_m | 2.000 | spsa | 0.496 | 9.772 | 0.404 |  |  |
| spsa_m | 4.000 | spsa | 0.516 | 4.491 | 0.551 |  |  |
| spsa_m | 8.000 | spsa | 0.375 | 2.362 | 0.674 |  |  |
| spsa_m | 16.000 | spsa | 0.275 | 1.244 | 0.796 |  |  |
| regression_K | 10.000 | regression | 1.255 | 52.573 | 0.779 |  |  |
| regression_K | 20.000 | regression | 0.238 | 0.026 | 0.982 |  |  |
| regression_K | 40.000 | regression | 0.232 | 0.014 | 0.985 |  |  |
| regression_K | 80.000 | regression | 0.238 | 0.011 | 0.985 |  |  |
| step_size_a_mult | 0.250 | central |  |  |  | 3.045 | 0.106 |
| step_size_a_mult | 1.000 | central |  |  |  | 3.173 | 0.036 |
| step_size_a_mult | 4.000 | central |  |  |  | 3.006 | 0.031 |
| param_scale_mult | 0.250 | central |  |  |  | 1.796 | 0.731 |
| param_scale_mult | 1.000 | central |  |  |  | 3.173 | 0.036 |
| param_scale_mult | 4.000 | central |  |  |  | 2.984 | 0.075 |

## trading__TECHM__T1
tuned hyperparameters (validation only): eps=0.4, a=8.0; budget=40000 rollouts; seeds=10

### Final validation return, sample efficiency, runtime
| method | final_val_mean | final_val_sd | final_val_ci_lo | final_val_ci_hi | rollouts_to_90pct_mean | rollouts_to_95pct_mean | reached_95pct_share | val@25%budget | val@100%budget | runtime_per_iter_ms |
|---|---|---|---|---|---|---|---|---|---|---|
| central | 0.749 | 0.051 | 0.720 | 0.781 | 65702.400 | 68211.200 | 0.200 | 0.701 | 0.726 | 143.711 |
| forward | 0.768 | 0.078 | 0.727 | 0.817 | 64972.800 | 66918.400 | 0.200 | 0.732 | 0.739 | 92.852 |
| gaussian | 0.751 | 0.043 | 0.727 | 0.774 | 67584.000 | 80000.000 | 0.000 | 0.682 | 0.722 | 215.141 |
| regression | 0.743 | 0.062 | 0.709 | 0.780 | 66784.000 | 67456.000 | 0.200 | 0.669 | 0.731 | 214.019 |
| spsa | 0.753 | 0.064 | 0.719 | 0.794 | 66304.000 | 67737.600 | 0.200 | 0.679 | 0.732 | 91.078 |
| central_nocrn | 0.849 | 0.186 | 0.727 | 0.943 | 24332.800 | 25228.800 | 0.800 | 0.627 | 0.565 | 138.777 |
| neldermead | 0.267 | 0.121 | 0.193 | 0.337 | 80000.000 | 80000.000 | 0.000 | 0.267 | 0.267 | 8.962 |
| random | 0.768 | 0.158 | 0.672 | 0.860 | 29187.200 | 29862.400 | 0.700 | 0.755 | 0.768 | 9.211 |
| spsa_nocrn | 0.855 | 0.111 | 0.791 | 0.918 | 28403.200 | 29427.200 | 0.700 | 0.748 | 0.723 | 80.625 |

### Stability (share of seeds)
| method | converged | unconverged | bound | stall | diverge |
|---|---|---|---|---|---|
| central | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| forward | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| gaussian | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| regression | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| spsa | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| central_nocrn | 0.400 | 0.600 | 0.000 | 0.000 | 0.000 |
| neldermead | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| random | 0.300 | 0.700 | 0.000 | 0.000 | 0.000 |
| spsa_nocrn | 0.300 | 0.700 | 0.000 | 0.000 | 0.000 |

### Gradient quality vs reference (rows: point x estimator)
| point | estimator | cosine | rel_l2 | sign_agree | bias_l2 | variance |
|---|---|---|---|---|---|---|
| start | forward | 0.947 | 0.365 | 0.971 | 0.023 | 0.010 |
| start | central | 0.943 | 0.369 | 0.976 | 0.038 | 0.009 |
| start | spsa | 0.564 | 1.096 | 0.690 | 0.077 | 0.082 |
| start | regression | 0.924 | 0.418 | 0.948 | 0.065 | 0.009 |
| start | gaussian | 0.674 | 0.864 | 0.781 | 0.098 | 0.046 |
| mid | forward | 0.712 | 1.679 | 0.676 | 0.058 | 0.001 |
| mid | central | 0.540 | 0.987 | 0.690 | 0.024 | 0.001 |
| mid | spsa | 0.226 | 1.757 | 0.586 | 0.036 | 0.003 |
| mid | regression | 0.175 | 2.888 | 0.567 | 0.035 | 0.014 |
| mid | gaussian | 0.170 | 1.833 | 0.586 | 0.047 | 0.004 |
| final | forward | 0.564 | 1.253 | 0.748 | 0.053 | 0.001 |
| final | central | 0.652 | 0.867 | 0.686 | 0.026 | 0.001 |
| final | spsa | 0.406 | 1.641 | 0.610 | 0.034 | 0.006 |
| final | regression | 0.342 | 2.246 | 0.590 | 0.032 | 0.014 |
| final | gaussian | 0.422 | 1.613 | 0.581 | 0.045 | 0.006 |

### Common random numbers
| estimator | crn | variance | cosine | bias_l2 |
|---|---|---|---|---|
| forward | True | 0.012 | 0.940 | 0.012 |
| forward | False | 0.897 | 0.304 | 0.097 |
| central | True | 0.011 | 0.940 | 0.017 |
| central | False | 0.225 | 0.375 | 0.093 |
| spsa | True | 0.085 | 0.578 | 0.077 |
| spsa | False | 0.142 | 0.484 | 0.064 |
| regression | True | 0.010 | 0.924 | 0.048 |
| regression | False | 0.157 | 0.469 | 0.071 |
| gaussian | True | 0.050 | 0.691 | 0.071 |
| gaussian | False | 0.077 | 0.566 | 0.089 |

| estimator | crn | final_val_mean | final_val_sd | n |
|---|---|---|---|---|
| central | True | 0.749 | 0.048 | 10 |
| central | False | 0.849 | 0.177 | 10 |
| spsa | True | 0.753 | 0.061 | 10 |
| spsa | False | 0.855 | 0.106 | 10 |

### Paired Wilcoxon tests (Holm; 52 comparisons in total, 14 significant at 0.05)
| outcome | A | B | mean_diff_A_minus_B | p_holm | rank_biserial | cohens_dz |
|---|---|---|---|---|---|---|
| final_validation_return | forward | neldermead | 0.501 | 0.041 | 1.000 | 4.893 |
| final_validation_return | central | neldermead | 0.482 | 0.041 | 1.000 | 4.966 |
| final_validation_return | spsa | neldermead | 0.486 | 0.041 | 1.000 | 3.849 |
| final_validation_return | regression | neldermead | 0.476 | 0.041 | 1.000 | 4.476 |
| final_validation_return | gaussian | neldermead | 0.484 | 0.041 | 1.000 | 5.071 |
| final_validation_return | random | neldermead | 0.501 | 0.041 | 1.000 | 2.640 |
| gradient_cosine_start | forward | spsa | 0.383 | 0.000 | 1.000 | 1.989 |
| gradient_cosine_start | forward | regression | 0.023 | 0.002 | 0.699 | 0.632 |
| gradient_cosine_start | forward | gaussian | 0.274 | 0.000 | 1.000 | 2.029 |
| gradient_cosine_start | central | spsa | 0.379 | 0.000 | 1.000 | 1.976 |
| gradient_cosine_start | central | regression | 0.019 | 0.002 | 0.682 | 0.517 |
| gradient_cosine_start | central | gaussian | 0.269 | 0.000 | 1.000 | 2.047 |
| gradient_cosine_start | spsa | regression | -0.360 | 0.000 | -0.996 | -1.752 |
| gradient_cosine_start | regression | gaussian | 0.250 | 0.000 | 1.000 | 1.862 |

### Sensitivity (see CSV; plots saved)
| param | value | estimator | bias_l2 | variance | cosine | final_val_mean | final_val_sd |
|---|---|---|---|---|---|---|---|
| eps | 0.010 | forward | 0.037 | 0.015 | 0.920 |  |  |
| eps | 0.030 | forward | 0.037 | 0.015 | 0.920 |  |  |
| eps | 0.100 | forward | 0.036 | 0.016 | 0.921 |  |  |
| eps | 0.300 | forward | 0.038 | 0.016 | 0.917 |  |  |
| eps | 1.000 | forward | 0.075 | 0.013 | 0.892 |  |  |
| eps | 0.010 | central | 0.037 | 0.015 | 0.920 |  |  |
| eps | 0.030 | central | 0.037 | 0.015 | 0.920 |  |  |
| eps | 0.100 | central | 0.038 | 0.015 | 0.919 |  |  |
| eps | 0.300 | central | 0.047 | 0.015 | 0.912 |  |  |
| eps | 1.000 | central | 0.083 | 0.010 | 0.899 |  |  |
| eps | 0.010 | spsa | 0.059 | 0.110 | 0.592 |  |  |
| eps | 0.030 | spsa | 0.059 | 0.109 | 0.592 |  |  |
| eps | 0.100 | spsa | 0.059 | 0.107 | 0.592 |  |  |
| eps | 0.300 | spsa | 0.075 | 0.087 | 0.584 |  |  |
| eps | 1.000 | spsa | 0.144 | 0.032 | 0.564 |  |  |
| N_paths | 8.000 | central | 0.031 | 0.061 | 0.712 |  |  |
| N_paths | 16.000 | central | 0.037 | 0.025 | 0.867 |  |  |
| N_paths | 32.000 | central | 0.026 | 0.014 | 0.929 |  |  |
| N_paths | 64.000 | central | 0.031 | 0.006 | 0.961 |  |  |
| N_paths | 128.000 | central | 0.028 | 0.003 | 0.982 |  |  |
| N_paths | 8.000 | spsa | 0.041 | 0.151 | 0.557 |  |  |
| N_paths | 16.000 | spsa | 0.053 | 0.088 | 0.581 |  |  |
| N_paths | 32.000 | spsa | 0.082 | 0.079 | 0.575 |  |  |
| N_paths | 64.000 | spsa | 0.101 | 0.060 | 0.563 |  |  |
| N_paths | 128.000 | spsa | 0.076 | 0.077 | 0.598 |  |  |
| N_paths | 8.000 | regression | 0.063 | 0.046 | 0.703 |  |  |
| N_paths | 16.000 | regression | 0.071 | 0.018 | 0.865 |  |  |
| N_paths | 32.000 | regression | 0.059 | 0.012 | 0.906 |  |  |
| N_paths | 64.000 | regression | 0.067 | 0.006 | 0.943 |  |  |
| N_paths | 128.000 | regression | 0.061 | 0.003 | 0.964 |  |  |
| spsa_m | 1.000 | spsa | 0.172 | 0.234 | 0.293 |  |  |
| spsa_m | 2.000 | spsa | 0.078 | 0.175 | 0.433 |  |  |
| spsa_m | 4.000 | spsa | 0.057 | 0.082 | 0.626 |  |  |
| spsa_m | 8.000 | spsa | 0.065 | 0.038 | 0.750 |  |  |
| spsa_m | 16.000 | spsa | 0.040 | 0.018 | 0.882 |  |  |
| regression_K | 7.000 | regression | 0.166 | 0.760 | 0.507 |  |  |
| regression_K | 14.000 | regression | 0.078 | 0.008 | 0.919 |  |  |
| regression_K | 28.000 | regression | 0.074 | 0.007 | 0.937 |  |  |
| regression_K | 56.000 | regression | 0.074 | 0.006 | 0.941 |  |  |
| step_size_a_mult | 0.250 | central |  |  |  | 0.644 | 0.065 |
| step_size_a_mult | 1.000 | central |  |  |  | 0.758 | 0.059 |
| step_size_a_mult | 4.000 | central |  |  |  | 0.838 | 0.064 |
| param_scale_mult | 0.250 | central |  |  |  | 0.382 | 0.103 |
| param_scale_mult | 1.000 | central |  |  |  | 0.758 | 0.059 |
| param_scale_mult | 4.000 | central |  |  |  | 0.856 | 0.082 |

## trading__TECHM__T2
tuned hyperparameters (validation only): eps=0.4, a=8.0; budget=40000 rollouts; seeds=10

### Final validation return, sample efficiency, runtime
| method | final_val_mean | final_val_sd | final_val_ci_lo | final_val_ci_hi | rollouts_to_90pct_mean | rollouts_to_95pct_mean | reached_95pct_share | val@25%budget | val@100%budget | runtime_per_iter_ms |
|---|---|---|---|---|---|---|---|---|---|---|
| central | 0.637 | 0.000 | 0.637 | 0.637 | 1689.600 | 1804.800 | 1.000 | 0.637 | 0.637 | 69.609 |
| forward | 0.613 | 0.002 | 0.611 | 0.614 | 1433.600 | 5350.400 | 1.000 | 0.606 | 0.605 | 50.020 |
| gaussian | 0.637 | 0.000 | 0.637 | 0.637 | 1804.800 | 1958.400 | 1.000 | 0.635 | 0.636 | 81.054 |
| regression | 0.637 | 0.000 | 0.637 | 0.637 | 2598.400 | 2777.600 | 1.000 | 0.636 | 0.635 | 73.544 |
| spsa | 0.637 | 0.000 | 0.637 | 0.637 | 2304.000 | 2611.200 | 1.000 | 0.637 | 0.637 | 90.217 |
| central_nocrn | 0.636 | 0.001 | 0.636 | 0.637 | 2803.200 | 3916.800 | 1.000 | 0.579 | 0.605 | 71.782 |
| neldermead | 0.102 | 0.140 | 0.020 | 0.182 | 80000.000 | 80000.000 | 0.000 | 0.102 | 0.102 | 11.367 |
| random | 0.609 | 0.022 | 0.597 | 0.621 | 23081.600 | 49728.000 | 0.500 | 0.573 | 0.609 | 11.792 |
| spsa_nocrn | 0.637 | 0.001 | 0.636 | 0.637 | 4659.200 | 5017.600 | 1.000 | 0.572 | 0.614 | 82.861 |

### Stability (share of seeds)
| method | converged | unconverged | bound | stall | diverge |
|---|---|---|---|---|---|
| central | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| forward | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| gaussian | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| regression | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| spsa | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| central_nocrn | 0.000 | 0.100 | 0.900 | 0.000 | 0.000 |
| neldermead | 0.100 | 0.000 | 0.000 | 0.900 | 0.000 |
| random | 0.800 | 0.200 | 0.000 | 0.000 | 0.000 |
| spsa_nocrn | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |

### Gradient quality vs reference (rows: point x estimator)
| point | estimator | cosine | rel_l2 | sign_agree | bias_l2 | variance |
|---|---|---|---|---|---|---|
| start | forward | 0.976 | 0.294 | 0.956 | 0.036 | 0.009 |
| start | central | 0.958 | 0.360 | 0.989 | 0.082 | 0.006 |
| start | spsa | 0.836 | 0.576 | 0.833 | 0.077 | 0.023 |
| start | regression | 0.820 | 0.569 | 0.867 | 0.110 | 0.018 |
| start | gaussian | 0.752 | 0.813 | 0.756 | 0.103 | 0.050 |
| mid | forward | 0.840 | 0.591 | 0.989 | 0.014 | 0.000 |
| mid | central | -0.136 | 4.270 | 0.667 | 0.107 | 0.001 |
| mid | spsa | -0.113 | 5.157 | 0.478 | 0.108 | 0.006 |
| mid | regression | 0.025 | 9.990 | 0.522 | 0.179 | 0.071 |
| mid | gaussian | -0.004 | 10.529 | 0.444 | 0.205 | 0.081 |
| final | forward | 0.842 | 0.587 | 0.989 | 0.014 | 0.000 |
| final | central | -0.134 | 4.254 | 0.667 | 0.107 | 0.001 |
| final | spsa | -0.111 | 5.139 | 0.478 | 0.108 | 0.006 |
| final | regression | 0.027 | 9.954 | 0.522 | 0.179 | 0.071 |
| final | gaussian | -0.004 | 10.492 | 0.433 | 0.205 | 0.081 |

### Common random numbers
| estimator | crn | variance | cosine | bias_l2 |
|---|---|---|---|---|
| forward | True | 0.007 | 0.972 | 0.039 |
| forward | False | 0.059 | 0.806 | 0.038 |
| central | True | 0.006 | 0.946 | 0.084 |
| central | False | 0.013 | 0.880 | 0.087 |
| spsa | True | 0.024 | 0.855 | 0.066 |
| spsa | False | 0.027 | 0.829 | 0.068 |
| regression | True | 0.024 | 0.748 | 0.120 |
| regression | False | 0.029 | 0.687 | 0.122 |
| gaussian | True | 0.034 | 0.746 | 0.133 |
| gaussian | False | 0.036 | 0.711 | 0.128 |

| estimator | crn | final_val_mean | final_val_sd | n |
|---|---|---|---|---|
| central | True | 0.637 | 0.000 | 10 |
| central | False | 0.636 | 0.001 | 10 |
| spsa | True | 0.637 | 0.000 | 10 |
| spsa | False | 0.637 | 0.001 | 10 |

### Optimality gap vs exhaustive grid on validation paths
| method | gap_mean | gap_sd | grid_best_val |
|---|---|---|---|
| central | 0.000 | 0.000 | 0.637 |
| central_nocrn | 0.000 | 0.001 | 0.637 |
| forward | 0.024 | 0.002 | 0.637 |
| gaussian | 0.000 | 0.000 | 0.637 |
| neldermead | 0.535 | 0.140 | 0.637 |
| random | 0.028 | 0.022 | 0.637 |
| regression | 0.000 | 0.000 | 0.637 |
| spsa | 0.000 | 0.000 | 0.637 |
| spsa_nocrn | 0.000 | 0.001 | 0.637 |

### Paired Wilcoxon tests (Holm; 52 comparisons in total, 33 significant at 0.05)
| outcome | A | B | mean_diff_A_minus_B | p_holm | rank_biserial | cohens_dz |
|---|---|---|---|---|---|---|
| final_validation_return | forward | central | -0.024 | 0.041 | -1.000 | -9.988 |
| final_validation_return | forward | spsa | -0.024 | 0.041 | -1.000 | -10.312 |
| final_validation_return | forward | regression | -0.024 | 0.041 | -1.000 | -10.328 |
| final_validation_return | forward | gaussian | -0.024 | 0.041 | -1.000 | -10.353 |
| final_validation_return | forward | neldermead | 0.511 | 0.041 | 1.000 | 3.666 |
| final_validation_return | central | random | 0.028 | 0.041 | 1.000 | 1.278 |
| final_validation_return | central | neldermead | 0.535 | 0.041 | 1.000 | 3.811 |
| final_validation_return | spsa | gaussian | -0.000 | 0.041 | -0.964 | -0.494 |
| final_validation_return | spsa | random | 0.028 | 0.041 | 1.000 | 1.282 |
| final_validation_return | spsa | neldermead | 0.535 | 0.041 | 1.000 | 3.812 |
| final_validation_return | regression | random | 0.028 | 0.041 | 1.000 | 1.282 |
| final_validation_return | regression | neldermead | 0.535 | 0.041 | 1.000 | 3.812 |
| final_validation_return | gaussian | random | 0.028 | 0.041 | 1.000 | 1.282 |
| final_validation_return | gaussian | neldermead | 0.535 | 0.041 | 1.000 | 3.812 |
| final_validation_return | random | neldermead | 0.508 | 0.041 | 1.000 | 3.633 |
| rollouts_to_95pct | forward | central | 3545.600 | 0.041 | 1.000 | 1.214 |
| rollouts_to_95pct | forward | gaussian | 3392.000 | 0.041 | 1.000 | 1.133 |
| rollouts_to_95pct | forward | random | -44377.600 | 0.041 | -1.000 | -1.354 |
| rollouts_to_95pct | forward | neldermead | -74649.600 | 0.041 | -1.000 | -20.637 |
| rollouts_to_95pct | central | random | -47923.200 | 0.041 | -1.000 | -1.481 |
| rollouts_to_95pct | central | neldermead | -78195.200 | 0.041 | -1.000 | -63.656 |
| rollouts_to_95pct | spsa | random | -47116.800 | 0.041 | -1.000 | -1.444 |
| rollouts_to_95pct | spsa | neldermead | -77388.800 | 0.041 | -1.000 | -45.142 |
| rollouts_to_95pct | regression | random | -46950.400 | 0.041 | -1.000 | -1.433 |
| rollouts_to_95pct | regression | neldermead | -77222.400 | 0.041 | -1.000 | -32.601 |
| rollouts_to_95pct | gaussian | random | -47769.600 | 0.041 | -1.000 | -1.498 |
| rollouts_to_95pct | gaussian | neldermead | -78041.600 | 0.041 | -1.000 | -48.988 |
| gradient_cosine_start | forward | spsa | 0.140 | 0.000 | 0.996 | 1.048 |
| gradient_cosine_start | forward | regression | 0.156 | 0.000 | 0.819 | 0.474 |
| gradient_cosine_start | forward | gaussian | 0.224 | 0.000 | 0.987 | 1.183 |
| gradient_cosine_start | central | spsa | 0.122 | 0.000 | 0.910 | 0.859 |
| gradient_cosine_start | central | regression | 0.137 | 0.001 | 0.725 | 0.460 |
| gradient_cosine_start | central | gaussian | 0.205 | 0.000 | 0.884 | 1.018 |

### Sensitivity (see CSV; plots saved)
| param | value | estimator | bias_l2 | variance | cosine | final_val_mean | final_val_sd |
|---|---|---|---|---|---|---|---|
| eps | 0.010 | forward | 0.032 | 0.013 | 0.949 |  |  |
| eps | 0.030 | forward | 0.028 | 0.013 | 0.949 |  |  |
| eps | 0.100 | forward | 0.016 | 0.015 | 0.948 |  |  |
| eps | 0.300 | forward | 0.034 | 0.015 | 0.946 |  |  |
| eps | 1.000 | forward | 0.192 | 0.004 | 0.753 |  |  |
| eps | 0.010 | central | 0.035 | 0.012 | 0.949 |  |  |
| eps | 0.030 | central | 0.035 | 0.012 | 0.949 |  |  |
| eps | 0.100 | central | 0.037 | 0.012 | 0.951 |  |  |
| eps | 0.300 | central | 0.078 | 0.009 | 0.954 |  |  |
| eps | 1.000 | central | 0.211 | 0.002 | 0.732 |  |  |
| eps | 0.010 | spsa | 0.037 | 0.036 | 0.870 |  |  |
| eps | 0.030 | spsa | 0.038 | 0.036 | 0.869 |  |  |
| eps | 0.100 | spsa | 0.040 | 0.036 | 0.868 |  |  |
| eps | 0.300 | spsa | 0.061 | 0.030 | 0.859 |  |  |
| eps | 1.000 | spsa | 0.165 | 0.011 | 0.804 |  |  |
| N_paths | 8.000 | central | 0.119 | 0.036 | 0.657 |  |  |
| N_paths | 16.000 | central | 0.120 | 0.014 | 0.826 |  |  |
| N_paths | 32.000 | central | 0.093 | 0.006 | 0.969 |  |  |
| N_paths | 64.000 | central | 0.102 | 0.003 | 0.981 |  |  |
| N_paths | 128.000 | central | 0.092 | 0.001 | 0.987 |  |  |
| N_paths | 8.000 | spsa | 0.094 | 0.028 | 0.782 |  |  |
| N_paths | 16.000 | spsa | 0.090 | 0.021 | 0.846 |  |  |
| N_paths | 32.000 | spsa | 0.077 | 0.035 | 0.829 |  |  |
| N_paths | 64.000 | spsa | 0.099 | 0.021 | 0.800 |  |  |
| N_paths | 128.000 | spsa | 0.083 | 0.028 | 0.796 |  |  |
| N_paths | 8.000 | regression | 0.136 | 0.048 | 0.582 |  |  |
| N_paths | 16.000 | regression | 0.134 | 0.024 | 0.646 |  |  |
| N_paths | 32.000 | regression | 0.134 | 0.016 | 0.828 |  |  |
| N_paths | 64.000 | regression | 0.107 | 0.017 | 0.891 |  |  |
| N_paths | 128.000 | regression | 0.115 | 0.019 | 0.806 |  |  |
| spsa_m | 1.000 | spsa | 0.072 | 0.109 | 0.558 |  |  |
| spsa_m | 2.000 | spsa | 0.072 | 0.057 | 0.693 |  |  |
| spsa_m | 4.000 | spsa | 0.067 | 0.031 | 0.800 |  |  |
| spsa_m | 8.000 | spsa | 0.081 | 0.018 | 0.850 |  |  |
| spsa_m | 16.000 | spsa | 0.075 | 0.008 | 0.925 |  |  |
| regression_K | 3.000 | regression | 0.191 | 1.427 | 0.661 |  |  |
| regression_K | 6.000 | regression | 0.144 | 0.015 | 0.790 |  |  |
| regression_K | 12.000 | regression | 0.155 | 0.007 | 0.828 |  |  |
| regression_K | 24.000 | regression | 0.150 | 0.005 | 0.883 |  |  |
| step_size_a_mult | 0.250 | central |  |  |  | 0.630 | 0.009 |
| step_size_a_mult | 1.000 | central |  |  |  | 0.637 | 0.000 |
| step_size_a_mult | 4.000 | central |  |  |  | 0.637 | 0.000 |
| param_scale_mult | 0.250 | central |  |  |  | 0.420 | 0.079 |
| param_scale_mult | 1.000 | central |  |  |  | 0.637 | 0.000 |
| param_scale_mult | 4.000 | central |  |  |  | 0.637 | 0.000 |
