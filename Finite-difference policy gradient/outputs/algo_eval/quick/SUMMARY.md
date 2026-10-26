# Algorithm-performance summary (quick)
analysis_plan.yaml sha256[:16] = faf34b6d2602a63c

## trading__TECHM__T1
tuned hyperparameters (validation only): eps=0.1, a=4.0; budget=6000 rollouts; seeds=3

### Final validation return, sample efficiency, runtime
| method | final_val_mean | final_val_sd | final_val_ci_lo | final_val_ci_hi | rollouts_to_90pct_mean | rollouts_to_95pct_mean | reached_95pct_share | val@25%budget | val@100%budget | runtime_per_iter_ms |
|---|---|---|---|---|---|---|---|---|---|---|
| central | 0.697 | 0.036 | 0.672 | 0.739 | 9493.333 | 9941.333 | 0.333 | 0.528 | 0.695 | 79.296 |
| forward | 0.732 | 0.018 | 0.712 | 0.747 | 6560.000 | 9109.333 | 0.333 | 0.600 | 0.728 | 55.410 |
| gaussian | 0.687 | 0.037 | 0.660 | 0.729 | 9642.667 | 12000.000 | 0.000 | 0.535 | 0.683 | 70.476 |
| regression | 0.691 | 0.030 | 0.660 | 0.721 | 9280.000 | 12000.000 | 0.000 | 0.529 | 0.691 | 79.566 |
| spsa | 0.751 | 0.015 | 0.734 | 0.765 | 3072.000 | 6901.333 | 0.667 | 0.587 | 0.738 | 45.195 |
| central_nocrn | 0.711 | 0.137 | 0.598 | 0.863 | 8746.667 | 8746.667 | 0.333 | 0.445 | 0.270 | 69.977 |
| neldermead | 0.320 | 0.095 | 0.213 | 0.395 | 12000.000 | 12000.000 | 0.000 | 0.320 | 0.320 | 4.886 |
| random | 0.737 | 0.272 | 0.424 | 0.916 | 922.667 | 922.667 | 1.000 | 0.645 | 0.737 | 4.850 |
| spsa_nocrn | 0.759 | 0.152 | 0.657 | 0.933 | 8768.000 | 9024.000 | 0.333 | 0.447 | 0.593 | 42.245 |

### Stability (share of seeds)
| method | converged | unconverged | bound | stall | diverge |
|---|---|---|---|---|---|
| central | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| forward | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| gaussian | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| regression | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| spsa | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| central_nocrn | 0.333 | 0.000 | 0.667 | 0.000 | 0.000 |
| neldermead | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| random | 0.667 | 0.333 | 0.000 | 0.000 | 0.000 |
| spsa_nocrn | 0.000 | 0.667 | 0.333 | 0.000 | 0.000 |

### Gradient quality vs reference (rows: point x estimator)
| point | estimator | cosine | rel_l2 | sign_agree | bias_l2 | variance |
|---|---|---|---|---|---|---|
| start | forward | 0.924 | 0.456 | 0.957 | 0.045 | 0.017 |
| start | central | 0.923 | 0.456 | 0.957 | 0.044 | 0.017 |
| start | spsa | 0.677 | 1.518 | 0.843 | 0.104 | 0.217 |
| start | regression | 0.915 | 0.464 | 0.957 | 0.050 | 0.017 |
| start | gaussian | 0.685 | 0.969 | 0.800 | 0.094 | 0.072 |
| mid | forward | 0.806 | 0.714 | 0.829 | 0.023 | 0.005 |
| mid | central | 0.784 | 0.745 | 0.829 | 0.033 | 0.005 |
| mid | spsa | 0.568 | 1.507 | 0.671 | 0.062 | 0.023 |
| mid | regression | 0.704 | 0.949 | 0.771 | 0.039 | 0.007 |
| mid | gaussian | 0.589 | 1.154 | 0.743 | 0.046 | 0.014 |
| final | forward | 0.795 | 0.779 | 0.814 | 0.020 | 0.003 |
| final | central | 0.772 | 0.779 | 0.843 | 0.027 | 0.003 |
| final | spsa | 0.551 | 1.662 | 0.671 | 0.059 | 0.018 |
| final | regression | 0.621 | 1.196 | 0.743 | 0.035 | 0.008 |
| final | gaussian | 0.588 | 1.169 | 0.743 | 0.041 | 0.011 |

### Common random numbers
| estimator | crn | variance | cosine | bias_l2 |
|---|---|---|---|---|
| forward | True | 0.039 | 0.866 | 0.016 |
| forward | False | 17.053 | -0.166 | 1.464 |
| central | True | 0.038 | 0.865 | 0.015 |
| central | False | 6.192 | 0.154 | 0.958 |
| spsa | True | 0.144 | 0.478 | 0.133 |
| spsa | False | 1.107 | 0.271 | 0.288 |
| regression | True | 0.040 | 0.859 | 0.015 |
| regression | False | 6.633 | 0.298 | 0.502 |
| gaussian | True | 0.136 | 0.639 | 0.056 |
| gaussian | False | 1.053 | 0.228 | 0.347 |

| estimator | crn | final_val_mean | final_val_sd | n |
|---|---|---|---|---|
| central | True | 0.697 | 0.030 | 3 |
| central | False | 0.711 | 0.112 | 3 |
| spsa | True | 0.751 | 0.013 | 3 |
| spsa | False | 0.759 | 0.124 | 3 |

### Paired Wilcoxon tests (Holm; 52 comparisons in total, 6 significant at 0.05)
| outcome | A | B | mean_diff_A_minus_B | p_holm | rank_biserial | cohens_dz |
|---|---|---|---|---|---|---|
| gradient_cosine_start | forward | spsa | 0.247 | 0.039 | 0.964 | 1.630 |
| gradient_cosine_start | forward | gaussian | 0.239 | 0.039 | 0.964 | 1.565 |
| gradient_cosine_start | central | spsa | 0.246 | 0.039 | 0.964 | 1.608 |
| gradient_cosine_start | central | gaussian | 0.239 | 0.039 | 0.964 | 1.577 |
| gradient_cosine_start | spsa | regression | -0.238 | 0.039 | -0.964 | -1.493 |
| gradient_cosine_start | regression | gaussian | 0.230 | 0.039 | 0.964 | 1.473 |

### Sensitivity (see CSV; plots saved)
| param | value | estimator | bias_l2 | variance | cosine | final_val_mean | final_val_sd |
|---|---|---|---|---|---|---|---|
| eps | 0.010 | forward | 0.015 | 0.027 | 0.852 |  |  |
| eps | 0.030 | forward | 0.014 | 0.027 | 0.853 |  |  |
| eps | 0.100 | forward | 0.014 | 0.027 | 0.854 |  |  |
| eps | 0.300 | forward | 0.018 | 0.028 | 0.853 |  |  |
| eps | 1.000 | forward | 0.058 | 0.022 | 0.840 |  |  |
| eps | 0.010 | central | 0.015 | 0.027 | 0.852 |  |  |
| eps | 0.030 | central | 0.015 | 0.027 | 0.852 |  |  |
| eps | 0.100 | central | 0.016 | 0.027 | 0.851 |  |  |
| eps | 0.300 | central | 0.027 | 0.026 | 0.842 |  |  |
| eps | 1.000 | central | 0.065 | 0.018 | 0.838 |  |  |
| eps | 0.010 | spsa | 0.117 | 0.095 | 0.568 |  |  |
| eps | 0.030 | spsa | 0.117 | 0.095 | 0.568 |  |  |
| eps | 0.100 | spsa | 0.116 | 0.093 | 0.569 |  |  |
| eps | 0.300 | spsa | 0.115 | 0.076 | 0.569 |  |  |
| eps | 1.000 | spsa | 0.156 | 0.030 | 0.554 |  |  |
| N_paths | 8.000 | central | 0.095 | 0.077 | 0.810 |  |  |
| N_paths | 16.000 | central | 0.060 | 0.027 | 0.902 |  |  |
| N_paths | 32.000 | central | 0.034 | 0.015 | 0.932 |  |  |
| N_paths | 64.000 | central | 0.021 | 0.006 | 0.965 |  |  |
| N_paths | 128.000 | central | 0.012 | 0.003 | 0.983 |  |  |
| N_paths | 8.000 | spsa | 0.149 | 0.260 | 0.558 |  |  |
| N_paths | 16.000 | spsa | 0.086 | 0.111 | 0.562 |  |  |
| N_paths | 32.000 | spsa | 0.097 | 0.099 | 0.560 |  |  |
| N_paths | 64.000 | spsa | 0.067 | 0.098 | 0.598 |  |  |
| N_paths | 128.000 | spsa | 0.151 | 0.181 | 0.665 |  |  |
| N_paths | 8.000 | regression | 0.098 | 0.072 | 0.807 |  |  |
| N_paths | 16.000 | regression | 0.054 | 0.026 | 0.899 |  |  |
| N_paths | 32.000 | regression | 0.039 | 0.017 | 0.921 |  |  |
| N_paths | 64.000 | regression | 0.020 | 0.006 | 0.968 |  |  |
| N_paths | 128.000 | regression | 0.013 | 0.004 | 0.978 |  |  |
| spsa_m | 1.000 | spsa | 0.229 | 0.356 | 0.298 |  |  |
| spsa_m | 2.000 | spsa | 0.197 | 0.231 | 0.373 |  |  |
| spsa_m | 4.000 | spsa | 0.116 | 0.100 | 0.589 |  |  |
| spsa_m | 8.000 | spsa | 0.075 | 0.055 | 0.710 |  |  |
| spsa_m | 16.000 | spsa | 0.052 | 0.036 | 0.794 |  |  |
| regression_K | 7.000 | regression | 0.045 | 0.064 | 0.814 |  |  |
| regression_K | 14.000 | regression | 0.030 | 0.018 | 0.921 |  |  |
| regression_K | 28.000 | regression | 0.032 | 0.017 | 0.927 |  |  |
| regression_K | 56.000 | regression | 0.033 | 0.017 | 0.924 |  |  |
| step_size_a_mult | 0.250 | central |  |  |  | 0.368 | 0.094 |
| step_size_a_mult | 1.000 | central |  |  |  | 0.633 | 0.066 |
| step_size_a_mult | 4.000 | central |  |  |  | 0.747 | 0.061 |
| param_scale_mult | 0.250 | central |  |  |  | 0.281 | 0.108 |
| param_scale_mult | 1.000 | central |  |  |  | 0.633 | 0.066 |
| param_scale_mult | 4.000 | central |  |  |  | 0.819 | 0.063 |

## trading__TECHM__T2
tuned hyperparameters (validation only): eps=0.3, a=4.0; budget=6000 rollouts; seeds=3

### Final validation return, sample efficiency, runtime
| method | final_val_mean | final_val_sd | final_val_ci_lo | final_val_ci_hi | rollouts_to_90pct_mean | rollouts_to_95pct_mean | reached_95pct_share | val@25%budget | val@100%budget | runtime_per_iter_ms |
|---|---|---|---|---|---|---|---|---|---|---|
| central | 0.633 | 0.006 | 0.626 | 0.637 | 2304.000 | 2624.000 | 1.000 | 0.521 | 0.633 | 45.566 |
| forward | 0.612 | 0.000 | 0.612 | 0.612 | 1792.000 | 3072.000 | 1.000 | 0.506 | 0.609 | 31.044 |
| gaussian | 0.625 | 0.020 | 0.603 | 0.637 | 2880.000 | 3072.000 | 1.000 | 0.502 | 0.625 | 37.610 |
| regression | 0.628 | 0.013 | 0.613 | 0.637 | 2762.667 | 2986.667 | 1.000 | 0.510 | 0.628 | 43.228 |
| spsa | 0.562 | 0.128 | 0.414 | 0.637 | 4768.000 | 5024.000 | 0.667 | 0.513 | 0.562 | 55.931 |
| central_nocrn | 0.606 | 0.013 | 0.592 | 0.617 | 1728.000 | 6240.000 | 0.667 | 0.269 | 0.376 | 39.224 |
| neldermead | 0.062 | 0.075 | -0.020 | 0.129 | 12000.000 | 12000.000 | 0.000 | 0.062 | 0.062 | 4.949 |
| random | 0.581 | 0.034 | 0.543 | 0.609 | 4474.667 | 8165.333 | 0.333 | 0.582 | 0.581 | 6.261 |
| spsa_nocrn | 0.551 | 0.149 | 0.379 | 0.637 | 5024.000 | 5365.333 | 0.667 | 0.445 | 0.524 | 49.503 |

### Stability (share of seeds)
| method | converged | unconverged | bound | stall | diverge |
|---|---|---|---|---|---|
| central | 0.000 | 0.333 | 0.667 | 0.000 | 0.000 |
| forward | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| gaussian | 0.000 | 0.333 | 0.667 | 0.000 | 0.000 |
| regression | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| spsa | 0.333 | 0.000 | 0.667 | 0.000 | 0.000 |
| central_nocrn | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| neldermead | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| random | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| spsa_nocrn | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |

### Gradient quality vs reference (rows: point x estimator)
| point | estimator | cosine | rel_l2 | sign_agree | bias_l2 | variance |
|---|---|---|---|---|---|---|
| start | forward | 0.941 | 0.498 | 1.000 | 0.034 | 0.025 |
| start | central | 0.974 | 0.384 | 1.000 | 0.034 | 0.014 |
| start | spsa | 0.834 | 0.677 | 0.700 | 0.059 | 0.048 |
| start | regression | 0.858 | 0.571 | 0.900 | 0.061 | 0.034 |
| start | gaussian | 0.719 | 0.789 | 0.867 | 0.134 | 0.040 |
| mid | forward | 0.895 | 0.544 | 1.000 | 0.022 | 0.000 |
| mid | central | 0.601 | 1.515 | 0.700 | 0.067 | 0.003 |
| mid | spsa | 0.500 | 1.888 | 0.567 | 0.068 | 0.006 |
| mid | regression | 0.338 | 4.432 | 0.567 | 0.158 | 0.065 |
| mid | gaussian | 0.573 | 3.555 | 0.567 | 0.121 | 0.039 |
| final | forward | 0.890 | 0.542 | 1.000 | 0.012 | 0.000 |
| final | central | 0.289 | 2.439 | 0.733 | 0.061 | 0.002 |
| final | spsa | 0.069 | 3.100 | 0.500 | 0.063 | 0.004 |
| final | regression | 0.177 | 7.163 | 0.533 | 0.152 | 0.056 |
| final | gaussian | 0.419 | 5.295 | 0.600 | 0.110 | 0.030 |

### Common random numbers
| estimator | crn | variance | cosine | bias_l2 |
|---|---|---|---|---|
| forward | True | 0.028 | 0.943 | 0.032 |
| forward | False | 0.168 | 0.314 | 0.133 |
| central | True | 0.014 | 0.972 | 0.033 |
| central | False | 0.022 | 0.923 | 0.056 |
| spsa | True | 0.048 | 0.841 | 0.065 |
| spsa | False | 0.097 | 0.699 | 0.039 |
| regression | True | 0.021 | 0.927 | 0.060 |
| regression | False | 0.041 | 0.766 | 0.081 |
| gaussian | True | 0.114 | 0.785 | 0.131 |
| gaussian | False | 0.064 | 0.818 | 0.086 |

| estimator | crn | final_val_mean | final_val_sd | n |
|---|---|---|---|---|
| central | True | 0.633 | 0.005 | 3 |
| central | False | 0.606 | 0.010 | 3 |
| spsa | True | 0.562 | 0.105 | 3 |
| spsa | False | 0.551 | 0.122 | 3 |

### Optimality gap vs exhaustive grid on validation paths
| method | gap_mean | gap_sd | grid_best_val |
|---|---|---|---|
| central | 0.004 | 0.006 | 0.637 |
| central_nocrn | 0.031 | 0.013 | 0.637 |
| forward | 0.025 | 0.000 | 0.637 |
| gaussian | 0.011 | 0.020 | 0.637 |
| neldermead | 0.575 | 0.075 | 0.637 |
| random | 0.056 | 0.034 | 0.637 |
| regression | 0.009 | 0.013 | 0.637 |
| spsa | 0.075 | 0.128 | 0.637 |
| spsa_nocrn | 0.086 | 0.149 | 0.637 |

### Paired Wilcoxon tests (Holm; 52 comparisons in total, 3 significant at 0.05)
| outcome | A | B | mean_diff_A_minus_B | p_holm | rank_biserial | cohens_dz |
|---|---|---|---|---|---|---|
| gradient_cosine_start | forward | regression | 0.084 | 0.020 | 1.000 | 0.531 |
| gradient_cosine_start | central | regression | 0.116 | 0.020 | 1.000 | 0.604 |
| gradient_cosine_start | central | gaussian | 0.255 | 0.020 | 1.000 | 1.302 |

### Sensitivity (see CSV; plots saved)
| param | value | estimator | bias_l2 | variance | cosine | final_val_mean | final_val_sd |
|---|---|---|---|---|---|---|---|
| eps | 0.010 | forward | 0.010 | 0.017 | 0.969 |  |  |
| eps | 0.030 | forward | 0.008 | 0.018 | 0.972 |  |  |
| eps | 0.100 | forward | 0.024 | 0.021 | 0.977 |  |  |
| eps | 0.300 | forward | 0.019 | 0.022 | 0.975 |  |  |
| eps | 1.000 | forward | 0.174 | 0.006 | 0.782 |  |  |
| eps | 0.010 | central | 0.012 | 0.017 | 0.968 |  |  |
| eps | 0.030 | central | 0.012 | 0.017 | 0.968 |  |  |
| eps | 0.100 | central | 0.014 | 0.017 | 0.968 |  |  |
| eps | 0.300 | central | 0.051 | 0.012 | 0.954 |  |  |
| eps | 1.000 | central | 0.202 | 0.004 | 0.717 |  |  |
| eps | 0.010 | spsa | 0.073 | 0.029 | 0.860 |  |  |
| eps | 0.030 | spsa | 0.074 | 0.029 | 0.860 |  |  |
| eps | 0.100 | spsa | 0.076 | 0.029 | 0.861 |  |  |
| eps | 0.300 | spsa | 0.084 | 0.022 | 0.864 |  |  |
| eps | 1.000 | spsa | 0.167 | 0.005 | 0.858 |  |  |
| N_paths | 8.000 | central | 0.035 | 0.056 | 0.944 |  |  |
| N_paths | 16.000 | central | 0.016 | 0.020 | 0.965 |  |  |
| N_paths | 32.000 | central | 0.033 | 0.008 | 0.983 |  |  |
| N_paths | 64.000 | central | 0.055 | 0.003 | 0.993 |  |  |
| N_paths | 128.000 | central | 0.058 | 0.002 | 0.994 |  |  |
| N_paths | 8.000 | spsa | 0.074 | 0.023 | 0.832 |  |  |
| N_paths | 16.000 | spsa | 0.052 | 0.031 | 0.867 |  |  |
| N_paths | 32.000 | spsa | 0.080 | 0.063 | 0.774 |  |  |
| N_paths | 64.000 | spsa | 0.081 | 0.025 | 0.808 |  |  |
| N_paths | 128.000 | spsa | 0.040 | 0.047 | 0.765 |  |  |
| N_paths | 8.000 | regression | 0.043 | 0.081 | 0.879 |  |  |
| N_paths | 16.000 | regression | 0.025 | 0.039 | 0.797 |  |  |
| N_paths | 32.000 | regression | 0.057 | 0.014 | 0.948 |  |  |
| N_paths | 64.000 | regression | 0.058 | 0.009 | 0.953 |  |  |
| N_paths | 128.000 | regression | 0.099 | 0.008 | 0.942 |  |  |
| spsa_m | 1.000 | spsa | 0.100 | 0.163 | 0.613 |  |  |
| spsa_m | 2.000 | spsa | 0.100 | 0.077 | 0.707 |  |  |
| spsa_m | 4.000 | spsa | 0.063 | 0.036 | 0.857 |  |  |
| spsa_m | 8.000 | spsa | 0.043 | 0.024 | 0.902 |  |  |
| spsa_m | 16.000 | spsa | 0.044 | 0.005 | 0.969 |  |  |
| regression_K | 3.000 | regression | 0.030 | 0.229 | 0.548 |  |  |
| regression_K | 6.000 | regression | 0.074 | 0.028 | 0.768 |  |  |
| regression_K | 12.000 | regression | 0.101 | 0.022 | 0.804 |  |  |
| regression_K | 24.000 | regression | 0.127 | 0.011 | 0.803 |  |  |
| step_size_a_mult | 0.250 | central |  |  |  | 0.448 | 0.097 |
| step_size_a_mult | 1.000 | central |  |  |  | 0.612 | 0.032 |
| step_size_a_mult | 4.000 | central |  |  |  | 0.637 | 0.000 |
| param_scale_mult | 0.250 | central |  |  |  | 0.233 | 0.137 |
| param_scale_mult | 1.000 | central |  |  |  | 0.612 | 0.032 |
| param_scale_mult | 4.000 | central |  |  |  | 0.637 | 0.000 |
