# Results (auto-generated)

M5 inventory results: primary series FOODS_2_197_CA_4 plus two replication series; costs are assumptions. Medians over repetitions.

## Replication series (base condition)

## Primary fold, base condition

| method        |   nrmse_J |   spearman |   nregret1 |    prec3 |   pair_acc |   diverged_frac |   pick_test_cum |
|:--------------|----------:|-----------:|-----------:|---------:|-----------:|----------------:|----------------:|
| TD            |    0.0543 |     0.3462 |     0.0443 |   0      |     0.6502 |          0      |         -0.1576 |
| GTD2          |    0.1902 |     0.3162 |     0.025  |   0.1667 |     0.6225 |          0      |          0.0683 |
| TDC           |    0.1898 |     0.2282 |     0.049  |   0      |     0.5766 |          0      |         -0.2111 |
| ETD           |   51.278  |     0.3172 |     0.0251 |   0      |     0.6254 |          0.0208 |          0.1699 |
| ETD_clip      |    0.0664 |     0.3312 |     0.0394 |   0      |     0.6403 |          0      |         -0.104  |
| ETD_noreset   |   51.2746 |     0.5044 |     0.0144 |   0.1667 |     0.6924 |          0.0208 |          0.4422 |
| LSTD          |    0.0682 |     0.3343 |     0.0443 |   0      |     0.6442 |          0      |         -0.1576 |
| FQE           |    0.1426 |     0.2871 |     0.0443 |   0      |     0.6122 |          0      |         -0.1576 |
| WIS           |    0.2097 |    -0.154  |     0.1425 |   0      |     0.4485 |          0      |         -0.1788 |
| OracleProj    |    0.0049 |     0.9958 |     0      |   1      |     0.9872 |          0      |          0.3476 |
| BehaviorValue |    0.1981 |   nan      |   nan      | nan      |   nan      |          0      |         -0.104  |
| Random        |    0.1974 |    -0.1127 |     0.0533 |   0      |     0.4415 |          0      |          0.1198 |


## Ablations: median nrmse_J

| cond                        |    TD |   GTD2 |   TDC |     ETD |   ETD_clip |   ETD_noreset |   LSTD |   FQE |   WIS |   OracleProj |   BehaviorValue |   Random |
|:----------------------------|------:|-------:|------:|--------:|-----------:|--------------:|-------:|------:|------:|-------------:|----------------:|---------:|
| ablation:coverage_eps_b_0.1 | 0.123 |  0.194 | 0.191 | 528.011 |      77.02 |       352.289 |  0.103 | 0.176 | 0.204 |        0.005 |           0.206 |    0.197 |


## Ablations: median spearman

| cond                        |    TD |   GTD2 |   TDC |   ETD |   ETD_clip |   ETD_noreset |   LSTD |   FQE |    WIS |   OracleProj |   BehaviorValue |   Random |
|:----------------------------|------:|-------:|------:|------:|-----------:|--------------:|-------:|------:|-------:|-------------:|----------------:|---------:|
| ablation:coverage_eps_b_0.1 | 0.526 |  0.346 | 0.458 |  0.81 |      0.186 |         0.905 |  0.535 | 0.289 | -0.229 |        0.993 |             nan |   -0.238 |


## Ablations: median nregret1

| cond                        |   TD |   GTD2 |   TDC |   ETD |   ETD_clip |   ETD_noreset |   LSTD |   FQE |   WIS |   OracleProj |   BehaviorValue |   Random |
|:----------------------------|-----:|-------:|------:|------:|-----------:|--------------:|-------:|------:|------:|-------------:|----------------:|---------:|
| ablation:coverage_eps_b_0.1 |    0 |  0.024 | 0.024 |     0 |      0.213 |             0 |      0 |  0.05 | 0.213 |            0 |             nan |    0.058 |


## Paired tests on nrmse_J (primary, base)

Friedman: {'stat': nan, 'p': nan, 'n': 2}

| a        | b        |   n |   median_diff |   cliffs_delta |   p |   p_holm |
|:---------|:---------|----:|--------------:|---------------:|----:|---------:|
| ETD      | ETD_clip |   2 |       51.2116 |            1   | nan |      nan |
| ETD      | FQE      |   2 |       51.1354 |            1   | nan |      nan |
| ETD      | GTD2     |   2 |       51.0878 |            1   | nan |      nan |
| ETD      | LSTD     |   2 |       51.2099 |            1   | nan |      nan |
| ETD      | TD       |   2 |       51.2237 |            1   | nan |      nan |
| ETD      | TDC      |   2 |       51.0882 |            1   | nan |      nan |
| ETD      | WIS      |   2 |       51.0683 |            1   | nan |      nan |
| ETD_clip | FQE      |   2 |       -0.0762 |           -1   | nan |      nan |
| ETD_clip | GTD2     |   2 |       -0.1238 |           -1   | nan |      nan |
| ETD_clip | LSTD     |   2 |       -0.0018 |            0   | nan |      nan |
| ETD_clip | TD       |   2 |        0.0121 |            0.5 | nan |      nan |
| ETD_clip | TDC      |   2 |       -0.1235 |           -1   | nan |      nan |
| ETD_clip | WIS      |   2 |       -0.1433 |           -1   | nan |      nan |
| FQE      | GTD2     |   2 |       -0.0476 |           -1   | nan |      nan |
| FQE      | LSTD     |   2 |        0.0745 |            1   | nan |      nan |
| FQE      | TD       |   2 |        0.0883 |            1   | nan |      nan |
| FQE      | TDC      |   2 |       -0.0472 |           -1   | nan |      nan |
| FQE      | WIS      |   2 |       -0.0671 |           -1   | nan |      nan |
| GTD2     | LSTD     |   2 |        0.122  |            1   | nan |      nan |
| GTD2     | TD       |   2 |        0.1359 |            1   | nan |      nan |
| GTD2     | TDC      |   2 |        0.0003 |            0   | nan |      nan |
| GTD2     | WIS      |   2 |       -0.0195 |           -1   | nan |      nan |
| LSTD     | TD       |   2 |        0.0138 |            1   | nan |      nan |
| LSTD     | TDC      |   2 |       -0.1217 |           -1   | nan |      nan |
| LSTD     | WIS      |   2 |       -0.1415 |           -1   | nan |      nan |
| TD       | TDC      |   2 |       -0.1355 |           -1   | nan |      nan |
| TD       | WIS      |   2 |       -0.1554 |           -1   | nan |      nan |
| TDC      | WIS      |   2 |       -0.0198 |           -1   | nan |      nan |


## Paired tests on spearman (primary, base)

Friedman: {'stat': nan, 'p': nan, 'n': 2}

| a        | b        |   n |   median_diff |   cliffs_delta |   p |   p_holm |
|:---------|:---------|----:|--------------:|---------------:|----:|---------:|
| ETD      | ETD_clip |   2 |       -0.014  |           0    | nan |      nan |
| ETD      | FQE      |   2 |        0.0301 |           0    | nan |      nan |
| ETD      | GTD2     |   2 |        0.0009 |           0    | nan |      nan |
| ETD      | LSTD     |   2 |       -0.0171 |           0    | nan |      nan |
| ETD      | TD       |   2 |       -0.029  |          -0.5  | nan |      nan |
| ETD      | TDC      |   2 |        0.089  |           1    | nan |      nan |
| ETD      | WIS      |   2 |        0.4711 |           1    | nan |      nan |
| ETD_clip | FQE      |   2 |        0.0441 |           1    | nan |      nan |
| ETD_clip | GTD2     |   2 |        0.0149 |           0    | nan |      nan |
| ETD_clip | LSTD     |   2 |       -0.0031 |          -0.25 | nan |      nan |
| ETD_clip | TD       |   2 |       -0.015  |          -0.5  | nan |      nan |
| ETD_clip | TDC      |   2 |        0.103  |           1    | nan |      nan |
| ETD_clip | WIS      |   2 |        0.4851 |           1    | nan |      nan |
| FQE      | GTD2     |   2 |       -0.0291 |           0    | nan |      nan |
| FQE      | LSTD     |   2 |       -0.0471 |          -1    | nan |      nan |
| FQE      | TD       |   2 |       -0.0591 |          -1    | nan |      nan |
| FQE      | TDC      |   2 |        0.0589 |           1    | nan |      nan |
| FQE      | WIS      |   2 |        0.4411 |           1    | nan |      nan |
| GTD2     | LSTD     |   2 |       -0.018  |           0    | nan |      nan |
| GTD2     | TD       |   2 |       -0.03   |          -0.5  | nan |      nan |
| GTD2     | TDC      |   2 |        0.088  |           1    | nan |      nan |
| GTD2     | WIS      |   2 |        0.4702 |           1    | nan |      nan |
| LSTD     | TD       |   2 |       -0.0119 |          -0.5  | nan |      nan |
| LSTD     | TDC      |   2 |        0.106  |           1    | nan |      nan |
| LSTD     | WIS      |   2 |        0.4882 |           1    | nan |      nan |
| TD       | TDC      |   2 |        0.118  |           1    | nan |      nan |
| TD       | WIS      |   2 |        0.5002 |           1    | nan |      nan |
| TDC      | WIS      |   2 |        0.3822 |           1    | nan |      nan |


## Paired tests on nregret1 (primary, base)

Friedman: {'stat': nan, 'p': nan, 'n': 2}

| a        | b        |   n |   median_diff |   cliffs_delta |   p |   p_holm |
|:---------|:---------|----:|--------------:|---------------:|----:|---------:|
| ETD      | ETD_clip |   2 |       -0.0142 |          -0.75 | nan |      nan |
| ETD      | FQE      |   2 |       -0.0191 |          -0.75 | nan |      nan |
| ETD      | GTD2     |   2 |        0.0002 |           0    | nan |      nan |
| ETD      | LSTD     |   2 |       -0.0191 |          -0.75 | nan |      nan |
| ETD      | TD       |   2 |       -0.0191 |          -0.75 | nan |      nan |
| ETD      | TDC      |   2 |       -0.0239 |          -1    | nan |      nan |
| ETD      | WIS      |   2 |       -0.1174 |          -1    | nan |      nan |
| ETD_clip | FQE      |   2 |       -0.0049 |          -0.25 | nan |      nan |
| ETD_clip | GTD2     |   2 |        0.0144 |           0    | nan |      nan |
| ETD_clip | LSTD     |   2 |       -0.0049 |          -0.25 | nan |      nan |
| ETD_clip | TD       |   2 |       -0.0049 |          -0.25 | nan |      nan |
| ETD_clip | TDC      |   2 |       -0.0096 |          -1    | nan |      nan |
| ETD_clip | WIS      |   2 |       -0.1032 |          -1    | nan |      nan |
| FQE      | GTD2     |   2 |        0.0193 |           0.25 | nan |      nan |
| FQE      | LSTD     |   2 |        0      |           0    | nan |      nan |
| FQE      | TD       |   2 |        0      |           0    | nan |      nan |
| FQE      | TDC      |   2 |       -0.0047 |          -0.25 | nan |      nan |
| FQE      | WIS      |   2 |       -0.0982 |          -1    | nan |      nan |
| GTD2     | LSTD     |   2 |       -0.0193 |          -0.25 | nan |      nan |
| GTD2     | TD       |   2 |       -0.0193 |          -0.25 | nan |      nan |
| GTD2     | TDC      |   2 |       -0.0241 |          -0.25 | nan |      nan |
| GTD2     | WIS      |   2 |       -0.1176 |          -1    | nan |      nan |
| LSTD     | TD       |   2 |        0      |           0    | nan |      nan |
| LSTD     | TDC      |   2 |       -0.0047 |          -0.25 | nan |      nan |
| LSTD     | WIS      |   2 |       -0.0982 |          -1    | nan |      nan |
| TD       | TDC      |   2 |       -0.0047 |          -0.25 | nan |      nan |
| TD       | WIS      |   2 |       -0.0982 |          -1    | nan |      nan |
| TDC      | WIS      |   2 |       -0.0935 |          -1    | nan |      nan |


## ETD emphasis diagnostics (primary, base; across repetitions)

|       |   rho_max |   rho_mean |   ess_rho |   gamma2_Erho2 |   F_mean |     F_var |     F_max |     M_var |
|:------|----------:|-----------:|----------:|---------------:|---------:|----------:|----------:|----------:|
| count |    48     |     48     |    48     |         48     |   48     |    48     |    48     |    48     |
| mean  |    23.371 |      0.99  |  1331.44  |          6.701 |    8.962 |  2964.38  |  1506.91  |  2964.38  |
| std   |     4.664 |      0.024 |   973.021 |          5.925 |    3.137 | 11246.6   |  2669.67  | 11246.6   |
| min   |     1.237 |      0.922 |   253.999 |          1.036 |    2.671 |   223.25  |   437.229 |   223.25  |
| 25%   |    24.333 |      0.974 |   857.125 |          3.468 |    9.3   |   285.143 |   437.229 |   285.143 |
| 50%   |    24.333 |      0.991 |  1279.79  |          4.259 |    9.434 |   361.355 |   567.242 |   361.355 |
| 75%   |    24.333 |      1.003 |  1480.89  |          5.725 |    9.471 |   910.418 |   908.669 |   910.418 |
| max   |    24.333 |      1.058 |  5195.58  |         21.309 |   17.82  | 73604.8   | 13114.7   | 73604.8   |

