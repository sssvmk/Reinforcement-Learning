# Results summary (auto-generated)

main runs: 1 (series x repetitions); ablation runs: 0

## Estimator x metric (overall pool, mean ± std over series x repetitions)

| estimator | spearman | kendall | regret@1_norm | p@3 | top1_in_true_top3 | pairwise_acc | mae | bias |
|---|---|---|---|---|---|---|---|---|
| fqe | 0.198 ± 0.000 | 0.261 ± 0.000 | 0.127 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.633 ± 0.000 | 576.615 ± 0.000 | -213.414 ± 0.000 |
| self | -0.254 ± 0.000 | -0.255 ± 0.000 | 0.094 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.372 ± 0.000 | 399.107 ± 0.000 | 194.159 ± 0.000 |
| ce_ope | -0.099 ± 0.000 | -0.067 ± 0.000 | 0.011 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.468 ± 0.000 | 229.075 ± 0.000 | 188.666 ± 0.000 |
| wis | 0.123 ± 0.000 | 0.079 ± 0.000 | 0.002 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.438 ± 0.000 | 469.512 ± 0.000 | 10.474 ± 0.000 |
| dr | 0.382 ± 0.000 | 0.365 ± 0.000 | 0.002 ± 0.000 | 0.333 ± 0.000 | 1.000 ± 0.000 | 0.687 ± 0.000 | 537.574 ± 0.000 | -203.470 ± 0.000 |
| behavior | 0.193 ± 0.000 | 0.133 ± 0.000 | 0.005 ± 0.000 | 0.667 ± 0.000 | 0.000 ± 0.000 | 0.567 ± 0.000 | 312.514 ± 0.000 | 305.335 ± 0.000 |
| random | 0.172 ± 0.000 | 0.104 ± 0.000 | 0.000 ± 0.000 | 0.333 ± 0.000 | 1.000 ± 0.000 | 0.553 ± 0.000 | 692.388 ± 0.000 | -291.886 ± 0.000 |

## Per run (series x repetition)

| run | estimator | spearman | regret@1_norm | p@3 |
|---|---|---|---|---|
| FOODS_3_228_CA_1_validation_rep0_medium_n600_main | fqe | 0.198 | 0.127 | 0.0 |
| FOODS_3_228_CA_1_validation_rep0_medium_n600_main | self | -0.254 | 0.094 | 0.0 |
| FOODS_3_228_CA_1_validation_rep0_medium_n600_main | ce_ope | -0.099 | 0.011 | 0.0 |
| FOODS_3_228_CA_1_validation_rep0_medium_n600_main | wis | 0.123 | 0.002 | 0.0 |
| FOODS_3_228_CA_1_validation_rep0_medium_n600_main | dr | 0.382 | 0.002 | 0.333 |

## Method-family leaderboard (true test-period value; best/mean/worst across candidates and runs)

| family | n | best_true | mean_true | worst_true |
|---|---|---|---|---|
| BC | 8 | 610.35 | 602.16 | 586.83 |
| CE | 6 | 600.16 | 570.15 | 554.04 |
| CQL | 5 | 609.41 | 600.89 | 569.74 |
| DQN | 6 | 620.68 | 311.73 | -209.75 |
| DynaQ | 8 | 602.22 | 559.36 | 542.88 |
| MCTS | 5 | 606.8 | 551.87 | 494.29 |
| ref | 11 | 624.48 | -547.29 | -5921.56 |

## Value error and within-family Spearman per estimator and family (bias = estimate - truth)

| estimator | family | bias | mae | within_spearman |
|---|---|---|---|---|
| ce_ope | BC | -27.96 | 30.81 | 0.83 |
| ce_ope | CE | 363.95 | 363.95 | -0.77 |
| ce_ope | CQL | -60.31 | 60.31 | 0.6 |
| ce_ope | DQN | 323.74 | 383.01 | -0.58 |
| ce_ope | DynaQ | 144.11 | 144.11 | 0.57 |
| ce_ope | MCTS | 225.12 | 225.12 | -0.1 |
| ce_ope | ref | 305.93 | 356.04 | 0.35 |
| dr | BC | -399.43 | 399.43 | 0.71 |
| dr | CE | -471.53 | 471.53 | 0.94 |
| dr | CQL | -438.94 | 438.94 | 0.9 |
| dr | DQN | -163.61 | 476.69 | -0.41 |
| dr | DynaQ | -482.57 | 482.57 | 0.95 |
| dr | MCTS | -403.22 | 403.22 | -0.1 |
| dr | ref | 464.33 | 853.18 | 0.5 |
| fqe | BC | -482.81 | 482.81 | 0.95 |
| fqe | CE | -479.21 | 479.21 | 0.89 |
| fqe | CQL | -479.96 | 479.96 | 1.0 |
| fqe | DQN | -166.19 | 476.98 | -0.41 |
| fqe | DynaQ | -487.51 | 487.51 | 0.9 |
| fqe | MCTS | -404.37 | 404.37 | -0.1 |
| fqe | ref | 509.03 | 939.34 | 0.35 |
| self | BC | nan | nan | nan |
| self | CE | 379.83 | 379.83 | -0.77 |
| self | CQL | -322.58 | 322.58 | 0.6 |
| self | DQN | -25.11 | 211.72 | 0.99 |
| self | DynaQ | 542.32 | 601.94 | -0.81 |
| self | MCTS | nan | nan | nan |
| self | ref | nan | nan | nan |
| wis | BC | -280.27 | 280.27 | 0.76 |
| wis | CE | -259.71 | 259.71 | -0.85 |
| wis | CQL | -292.43 | 292.43 | 0.2 |
| wis | DQN | -8.52 | 340.45 | 0.05 |
| wis | DynaQ | -242.91 | 242.91 | -0.58 |
| wis | MCTS | -295.04 | 295.04 | 0.71 |
| wis | ref | 840.49 | 1116.58 | -0.08 |

## Cross-family fairness check

| estimator | mean within-family Spearman | across-family (overall) Spearman |
|---|---|---|
| fqe | 0.511 | 0.198 |
| ce_ope | 0.129 | -0.099 |
| self | 0.002 | -0.254 |

## Within-family ranking quality (families with >= 4 candidates)

| estimator | family | spearman | regret@1_norm |
|---|---|---|---|
| ce_ope | BC | 0.833 | 0.089 |
| ce_ope | CE | -0.771 | 0.997 |
| ce_ope | CQL | 0.6 | 0.008 |
| ce_ope | DQN | -0.58 | 1.0 |
| ce_ope | DynaQ | 0.571 | 0.802 |
| ce_ope | MCTS | -0.1 | 0.207 |
| ce_ope | ref | 0.351 | 0.127 |
| dr | BC | 0.714 | 0.0 |
| dr | CE | 0.943 | 0.0 |
| dr | CQL | 0.9 | 0.002 |
| dr | DQN | -0.406 | 1.0 |
| dr | DynaQ | 0.952 | 0.0 |
| dr | MCTS | -0.1 | 0.207 |
| dr | ref | 0.5 | 0.002 |
| fqe | BC | 0.952 | 0.014 |
| fqe | CE | 0.886 | 0.005 |
| fqe | CQL | 1.0 | 0.0 |
| fqe | DQN | -0.406 | 1.0 |
| fqe | DynaQ | 0.905 | 0.0 |
| fqe | MCTS | -0.1 | 0.207 |
| fqe | ref | 0.345 | 0.127 |
| self | BC | nan | nan |
| self | CE | -0.771 | 0.995 |
| self | CQL | 0.6 | 0.002 |
| self | DQN | 0.986 | 0.0 |
| self | DynaQ | -0.81 | 1.0 |
| self | MCTS | nan | nan |
| self | ref | nan | nan |
| wis | BC | 0.762 | 0.014 |
| wis | CE | -0.845 | 0.995 |
| wis | CQL | 0.2 | 0.002 |
| wis | DQN | 0.051 | 1.0 |
| wis | DynaQ | -0.577 | 0.849 |
| wis | MCTS | 0.707 | 0.0 |
| wis | ref | -0.08 | 0.002 |

## Pool with vs. without clearly-bad anchors (never_order, order_max)

| run | estimator | spearman_all | spearman_no_anchors |
|---|---|---|---|
| FOODS_3_228_CA_1_validation_rep0_medium_n600_main | fqe | 0.198 | 0.211 |
| FOODS_3_228_CA_1_validation_rep0_medium_n600_main | ce_ope | -0.099 | -0.142 |
| FOODS_3_228_CA_1_validation_rep0_medium_n600_main | self | -0.254 | -0.254 |
| FOODS_3_228_CA_1_validation_rep0_medium_n600_main | dr | 0.382 | 0.409 |

## Hyper-parameter sweeps (n_min / lambda, n_planning, CQL alpha, MCTS budget, DQN steps): full table in table_hyperparameter_views.csv

## MCTS: FQE on subsample vs. distilled policy

| name | truth | fqe | fqe_distilled | gap(FQE_sub - FQE_distilled) |
|---|---|---|---|---|
| mcts_bootstrap_b50 | 494.29 | 145.45 | 234.76 | -89.31 |
| mcts_bootstrap_b200 | 606.8 | 127.22 | 138.38 | -11.16 |
| mcts_mean_b50 | 571.37 | 158.99 | 242.34 | -83.35 |
| mcts_mean_b200 | 583.56 | 161.17 | 253.95 | -92.78 |
| mcts_negbin_b50 | 503.31 | 144.67 | 217.51 | -72.84 |
