# Results (auto-generated)

All trading results rest on ONE instrument (TECHM). Medians over repetitions.

## Primary fold, base condition

| method        |   nrmse_J |   spearman |   nregret1 |    prec3 |   pair_acc |   diverged_frac |   pick_test_cum |
|:--------------|----------:|-----------:|-----------:|---------:|-----------:|----------------:|----------------:|
| TD            |    0.1959 |     0.7517 |     0.1975 |   0.3333 |     0.7827 |          0      |          0.7418 |
| GTD2          |    0.3267 |     0.647  |     0.1802 |   0.3333 |     0.7432 |          0      |          0.434  |
| TDC           |    0.3385 |     0.6122 |     0.1788 |   0.3333 |     0.7358 |          0      |          0.2268 |
| ETD           | 2158.04   |     0.417  |     0.4135 |   0.1667 |     0.6575 |          0.069  |          0.23   |
| ETD_clip      |    0.4967 |     0.6186 |     0.1172 |   0.5    |     0.7333 |          0      |          0.3356 |
| ETD_noreset   | 2179.48   |     0.3398 |     0.4135 |   0.3333 |     0.6263 |          0.0862 |          0.23   |
| LSTD          |    0.269  |     0.8283 |     0.1707 |   0.3333 |     0.8346 |          0      |          1.0236 |
| FQE           |    0.1508 |     0.8386 |     0.1707 |   0.8333 |     0.8321 |          0      |          1.0236 |
| WIS           |    1.1794 |     0.3242 |     0.3879 |   0.3333 |     0.6185 |          0      |          0.5217 |
| OracleProj    |    0.1772 |     0.8953 |     0.1489 |   0.6667 |     0.879  |          0      |          0.6174 |
| BehaviorValue |    0.2498 |   nan      |   nan      | nan      |   nan      |          0      |          1.0236 |
| Random        |   16.685  |    -0.2829 |     0.5404 |   0      |     0.4074 |          0      |          0.363  |


## Ablations: median nrmse_J

| cond                        |    TD |   GTD2 |   TDC |   ETD |   ETD_clip |   ETD_noreset |   LSTD |   FQE |   WIS |   OracleProj |   BehaviorValue |   Random |
|:----------------------------|------:|-------:|------:|------:|-----------:|--------------:|-------:|------:|------:|-------------:|----------------:|---------:|
| ablation:coverage_eps_b_0.1 | 0.314 |   0.37 | 0.361 | 0.523 |      0.456 |          0.51 |  0.662 | 0.178 | 0.985 |        0.154 |           0.259 |   18.212 |


## Ablations: median spearman

| cond                        |   TD |   GTD2 |   TDC |   ETD |   ETD_clip |   ETD_noreset |   LSTD |   FQE |   WIS |   OracleProj |   BehaviorValue |   Random |
|:----------------------------|-----:|-------:|------:|------:|-----------:|--------------:|-------:|------:|------:|-------------:|----------------:|---------:|
| ablation:coverage_eps_b_0.1 | 0.33 |   0.05 | 0.447 | 0.246 |      0.233 |          0.29 |  0.758 | 0.711 | 0.431 |        0.886 |             nan |   -0.045 |


## Ablations: median nregret1

| cond                        |    TD |   GTD2 |   TDC |   ETD |   ETD_clip |   ETD_noreset |   LSTD |   FQE |   WIS |   OracleProj |   BehaviorValue |   Random |
|:----------------------------|------:|-------:|------:|------:|-----------:|--------------:|-------:|------:|------:|-------------:|----------------:|---------:|
| ablation:coverage_eps_b_0.1 | 0.112 |  0.289 | 0.166 | 0.604 |      0.112 |         0.604 |  0.112 | 0.112 | 0.166 |        0.112 |             nan |    0.522 |


## Paired tests on nrmse_J (primary, base)

Friedman: {'stat': nan, 'p': nan, 'n': 2}

| a        | b        |   n |   median_diff |   cliffs_delta |   p |   p_holm |
|:---------|:---------|----:|--------------:|---------------:|----:|---------:|
| ETD      | ETD_clip |   2 |     2157.54   |            1   | nan |      nan |
| ETD      | FQE      |   2 |     2157.89   |            1   | nan |      nan |
| ETD      | GTD2     |   2 |     2157.71   |            1   | nan |      nan |
| ETD      | LSTD     |   2 |     2157.77   |            1   | nan |      nan |
| ETD      | TD       |   2 |     2157.84   |            1   | nan |      nan |
| ETD      | TDC      |   2 |     2157.7    |            1   | nan |      nan |
| ETD      | WIS      |   2 |     2156.86   |            1   | nan |      nan |
| ETD_clip | FQE      |   2 |        0.3458 |            1   | nan |      nan |
| ETD_clip | GTD2     |   2 |        0.17   |            0   | nan |      nan |
| ETD_clip | LSTD     |   2 |        0.2277 |            0.5 | nan |      nan |
| ETD_clip | TD       |   2 |        0.3007 |            1   | nan |      nan |
| ETD_clip | TDC      |   2 |        0.1582 |            0   | nan |      nan |
| ETD_clip | WIS      |   2 |       -0.6827 |           -1   | nan |      nan |
| FQE      | GTD2     |   2 |       -0.1759 |           -1   | nan |      nan |
| FQE      | LSTD     |   2 |       -0.1181 |           -1   | nan |      nan |
| FQE      | TD       |   2 |       -0.0451 |           -0.5 | nan |      nan |
| FQE      | TDC      |   2 |       -0.1876 |           -1   | nan |      nan |
| FQE      | WIS      |   2 |       -1.0285 |           -1   | nan |      nan |
| GTD2     | LSTD     |   2 |        0.0578 |            0.5 | nan |      nan |
| GTD2     | TD       |   2 |        0.1308 |            1   | nan |      nan |
| GTD2     | TDC      |   2 |       -0.0117 |           -0.5 | nan |      nan |
| GTD2     | WIS      |   2 |       -0.8526 |           -1   | nan |      nan |
| LSTD     | TD       |   2 |        0.073  |            1   | nan |      nan |
| LSTD     | TDC      |   2 |       -0.0695 |           -1   | nan |      nan |
| LSTD     | WIS      |   2 |       -0.9104 |           -1   | nan |      nan |
| TD       | TDC      |   2 |       -0.1425 |           -1   | nan |      nan |
| TD       | WIS      |   2 |       -0.9834 |           -1   | nan |      nan |
| TDC      | WIS      |   2 |       -0.8409 |           -1   | nan |      nan |


## Paired tests on spearman (primary, base)

Friedman: {'stat': nan, 'p': nan, 'n': 0}

| a        | b        |   n |   median_diff |   cliffs_delta |   p |   p_holm |
|:---------|:---------|----:|--------------:|---------------:|----:|---------:|
| ETD      | ETD_clip |   2 |       -0.2016 |           -0.5 | nan |      nan |
| ETD      | FQE      |   2 |       -0.4216 |           -1   | nan |      nan |
| ETD      | GTD2     |   2 |       -0.2299 |           -0.5 | nan |      nan |
| ETD      | LSTD     |   2 |       -0.4112 |           -1   | nan |      nan |
| ETD      | TD       |   2 |       -0.3346 |           -1   | nan |      nan |
| ETD      | TDC      |   2 |       -0.1952 |           -0.5 | nan |      nan |
| ETD      | WIS      |   2 |        0.0928 |            0   | nan |      nan |
| ETD_clip | FQE      |   2 |       -0.22   |           -1   | nan |      nan |
| ETD_clip | GTD2     |   2 |       -0.0283 |            0   | nan |      nan |
| ETD_clip | LSTD     |   2 |       -0.2097 |           -0.5 | nan |      nan |
| ETD_clip | TD       |   2 |       -0.133  |           -0.5 | nan |      nan |
| ETD_clip | TDC      |   2 |        0.0064 |            0   | nan |      nan |
| ETD_clip | WIS      |   2 |        0.2944 |            1   | nan |      nan |
| FQE      | GTD2     |   2 |        0.1917 |            1   | nan |      nan |
| FQE      | LSTD     |   2 |        0.0103 |            0   | nan |      nan |
| FQE      | TD       |   2 |        0.087  |            0.5 | nan |      nan |
| FQE      | TDC      |   2 |        0.2264 |            1   | nan |      nan |
| FQE      | WIS      |   2 |        0.5144 |            1   | nan |      nan |
| GTD2     | LSTD     |   2 |       -0.1813 |           -1   | nan |      nan |
| GTD2     | TD       |   2 |       -0.1047 |           -0.5 | nan |      nan |
| GTD2     | TDC      |   2 |        0.0347 |            0   | nan |      nan |
| GTD2     | WIS      |   2 |        0.3227 |            1   | nan |      nan |
| LSTD     | TD       |   2 |        0.0766 |            0.5 | nan |      nan |
| LSTD     | TDC      |   2 |        0.2161 |            1   | nan |      nan |
| LSTD     | WIS      |   2 |        0.5041 |            1   | nan |      nan |
| TD       | TDC      |   2 |        0.1394 |            0.5 | nan |      nan |
| TD       | WIS      |   2 |        0.4274 |            1   | nan |      nan |
| TDC      | WIS      |   2 |        0.288  |            1   | nan |      nan |


## Paired tests on nregret1 (primary, base)

Friedman: {'stat': nan, 'p': nan, 'n': 0}

| a        | b        |   n |   median_diff |   cliffs_delta |   p |   p_holm |
|:---------|:---------|----:|--------------:|---------------:|----:|---------:|
| ETD      | ETD_clip |   2 |        0.2963 |           1    | nan |      nan |
| ETD      | FQE      |   2 |        0.2428 |           0.5  | nan |      nan |
| ETD      | GTD2     |   2 |        0.2333 |           0.5  | nan |      nan |
| ETD      | LSTD     |   2 |        0.2428 |           0.5  | nan |      nan |
| ETD      | TD       |   2 |        0.216  |           0.5  | nan |      nan |
| ETD      | TDC      |   2 |        0.2347 |           0.5  | nan |      nan |
| ETD      | WIS      |   2 |        0.0256 |           0.5  | nan |      nan |
| ETD_clip | FQE      |   2 |       -0.0535 |          -0.5  | nan |      nan |
| ETD_clip | GTD2     |   2 |       -0.063  |          -0.5  | nan |      nan |
| ETD_clip | LSTD     |   2 |       -0.0535 |          -0.5  | nan |      nan |
| ETD_clip | TD       |   2 |       -0.0803 |          -0.75 | nan |      nan |
| ETD_clip | TDC      |   2 |       -0.0616 |          -0.25 | nan |      nan |
| ETD_clip | WIS      |   2 |       -0.2706 |          -0.75 | nan |      nan |
| FQE      | GTD2     |   2 |       -0.0095 |           0    | nan |      nan |
| FQE      | LSTD     |   2 |        0      |           0    | nan |      nan |
| FQE      | TD       |   2 |       -0.0268 |          -0.25 | nan |      nan |
| FQE      | TDC      |   2 |       -0.0081 |           0    | nan |      nan |
| FQE      | WIS      |   2 |       -0.2172 |          -0.5  | nan |      nan |
| GTD2     | LSTD     |   2 |        0.0095 |           0    | nan |      nan |
| GTD2     | TD       |   2 |       -0.0173 |           0    | nan |      nan |
| GTD2     | TDC      |   2 |        0.0014 |           0.5  | nan |      nan |
| GTD2     | WIS      |   2 |       -0.2077 |          -0.5  | nan |      nan |
| LSTD     | TD       |   2 |       -0.0268 |          -0.25 | nan |      nan |
| LSTD     | TDC      |   2 |       -0.0081 |           0    | nan |      nan |
| LSTD     | WIS      |   2 |       -0.2172 |          -0.5  | nan |      nan |
| TD       | TDC      |   2 |        0.0187 |           0    | nan |      nan |
| TD       | WIS      |   2 |       -0.1904 |          -0.25 | nan |      nan |
| TDC      | WIS      |   2 |       -0.2091 |          -0.5  | nan |      nan |


## ETD emphasis diagnostics (primary, base; across repetitions)

|       |   rho_max |   rho_mean |   ess_rho |   gamma2_Erho2 |   F_mean |            F_var |     F_max |      M_var |
|:------|----------:|-----------:|----------:|---------------:|---------:|-----------------:|----------:|-----------:|
| count |    58     |     58     |    58     |         58     |   58     |     58           |    58     |     58     |
| mean  |     5.974 |      1.002 |  2543.02  |          2.643 |   16.396 | 307759           | 16041.5   |  76939.8   |
| std   |     1.333 |      0.022 |  1332.52  |          1.102 |   13.054 | 799537           | 24968.1   | 199884     |
| min   |     1.118 |      0.957 |   966.019 |          0.965 |    4.034 |    151.198       |   183.7   |     37.8   |
| 25%   |     6.333 |      0.99  |  1603.29  |          1.815 |    8.653 |    582.812       |   960.616 |    145.703 |
| 50%   |     6.333 |      1.004 |  1956.62  |          2.725 |   12.307 |   5384.38        |  3442.85  |   1346.1   |
| 75%   |     6.333 |      1.012 |  2764.34  |          3.399 |   17.489 | 145334           | 17888.2   |  36333.5   |
| max   |     6.333 |      1.049 |  5579.36  |          5.706 |   71.236 |      3.47517e+06 | 91426     | 868792     |

