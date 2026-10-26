# Neural (R-I4, R-T3) vs linear REINFORCE policies and equal-budget comparison methods on the neural policy

| key | policy | baseline | n_seeds | val_J_mean | val_J_sd | test_mean | test_ci_lo | test_ci_hi | test_unit |
|---|---|---|---|---|---|---|---|---|---|
| FOODS_3_124_CA_3_validation | I4 | cm-random | 2 | 2.7527 | 0.0206 | 736.8188 | 721.2805 | 752.3570 | profit/window |
| FOODS_3_124_CA_3_validation | I4 | cm-spsa | 2 | 2.8351 | 0.1308 | 751.4713 | 721.1253 | 781.8173 | profit/window |
| FOODS_3_124_CA_3_validation | I4 | linear | 2 | 2.9286 | 0.0304 | 774.6197 | 774.5602 | 774.6793 | profit/window |
| FOODS_3_124_CA_3_validation | I4 | loo_time | 2 | 2.9513 | 0.0091 | 778.5885 | 774.9716 | 782.2055 | profit/window |
| FOODS_3_124_CA_3_validation | I4 | none | 2 | 2.7556 | 0.1081 | 734.3031 | 728.5297 | 740.0766 | profit/window |
| FOODS_3_124_CA_3_validation | I2 | linear | 3 | 2.8094 | 0.2350 | 765.2218 | 718.4683 | 791.1494 | profit/window |
| FOODS_3_124_CA_3_validation | I2 | loo_time | 3 | 2.9080 | 0.0768 | 776.6889 | 761.0948 | 786.0534 | profit/window |
| FOODS_3_124_CA_3_validation | I2 | none | 3 | 2.9751 | 0.0507 | 780.6488 | 766.2595 | 791.4679 | profit/window |
| FOODS_3_124_CA_3_validation | I3 | linear | 3 | 2.9947 | 0.1097 | 780.3572 | 770.7888 | 797.3756 | profit/window |
| FOODS_3_124_CA_3_validation | I3 | loo_time | 3 | 3.0142 | 0.0067 | 796.8574 | 795.5015 | 798.0502 | profit/window |
| FOODS_3_124_CA_3_validation | I3 | none | 3 | -0.1810 | 5.5555 | 290.4129 | -716.7229 | 798.3051 | profit/window |
| FOODS_2_128_WI_2_validation | I4 | cm-random | 2 | 0.6661 | 0.0471 | 159.0832 | 157.2972 | 160.8692 | profit/window |
| FOODS_2_128_WI_2_validation | I4 | cm-spsa | 2 | 0.6718 | 0.0434 | 151.4150 | 149.3273 | 153.5027 | profit/window |
| FOODS_2_128_WI_2_validation | I4 | linear | 2 | 0.7750 | 0.0280 | 166.9320 | 164.4432 | 169.4209 | profit/window |
| FOODS_2_128_WI_2_validation | I4 | loo_time | 2 | 0.7251 | 0.1188 | 161.7002 | 150.5931 | 172.8073 | profit/window |
| FOODS_2_128_WI_2_validation | I4 | none | 2 | 0.7156 | 0.0362 | 163.7786 | 160.9381 | 166.6191 | profit/window |
| FOODS_2_128_WI_2_validation | I2 | linear | 3 | 0.6825 | 0.0176 | 157.8421 | 156.4671 | 159.3535 | profit/window |
| FOODS_2_128_WI_2_validation | I2 | loo_time | 3 | 0.6934 | 0.0379 | 160.1945 | 156.4671 | 167.5550 | profit/window |
| FOODS_2_128_WI_2_validation | I2 | none | 3 | 0.6783 | 0.0135 | 158.3031 | 156.4671 | 160.2844 | profit/window |
| FOODS_2_128_WI_2_validation | I3 | linear | 3 | -1.0853 | 1.4668 | 13.0209 | -55.5024 | 147.1595 | profit/window |
| FOODS_2_128_WI_2_validation | I3 | loo_time | 3 | -1.0672 | 1.4272 | 16.2927 | -53.4776 | 154.4426 | profit/window |
| FOODS_2_128_WI_2_validation | I3 | none | 3 | -1.9664 | 0.1083 | -55.9874 | -66.2444 | -49.3392 | profit/window |
| HOBBIES_1_371_WI_1_validation | I4 | cm-random | 2 | 2.9341 | 0.0391 | 107.8826 | 105.9962 | 109.7690 | profit/window |
| HOBBIES_1_371_WI_1_validation | I4 | cm-spsa | 2 | 2.8467 | 0.3792 | 76.7113 | 68.1787 | 85.2438 | profit/window |
| HOBBIES_1_371_WI_1_validation | I4 | linear | 2 | 3.0264 | 0.0097 | 98.1685 | 94.5741 | 101.7629 | profit/window |
| HOBBIES_1_371_WI_1_validation | I4 | loo_time | 2 | 2.9854 | 0.0404 | 100.0388 | 99.2513 | 100.8263 | profit/window |
| HOBBIES_1_371_WI_1_validation | I4 | none | 2 | 3.1250 | 0.2148 | 95.1260 | 94.5852 | 95.6669 | profit/window |
| HOBBIES_1_371_WI_1_validation | I2 | linear | 3 | 3.1551 | 0.2299 | 100.2856 | 95.6222 | 102.9043 | profit/window |
| HOBBIES_1_371_WI_1_validation | I2 | loo_time | 3 | 3.1628 | 0.0722 | 100.1456 | 91.4475 | 106.0015 | profit/window |
| HOBBIES_1_371_WI_1_validation | I2 | none | 3 | 3.2592 | 0.0698 | 96.2719 | 89.7082 | 104.9170 | profit/window |
| HOBBIES_1_371_WI_1_validation | I3 | linear | 3 | 3.1216 | 0.2262 | 97.6145 | 90.5329 | 104.6603 | profit/window |
| HOBBIES_1_371_WI_1_validation | I3 | loo_time | 3 | 3.0144 | 0.2159 | 94.1478 | 92.4358 | 95.7085 | profit/window |
| HOBBIES_1_371_WI_1_validation | I3 | none | 3 | 1.6824 | 1.9770 | 89.2130 | 81.5784 | 100.9371 | profit/window |
| TECHM | T3 | cm-random | 2 | 0.8887 | 0.0715 | 0.7008 | 0.6506 | 0.7510 | cum. return |
| TECHM | T3 | cm-spsa | 2 | 0.6002 | 0.0704 | 0.7624 | 0.6009 | 0.9239 | cum. return |
| TECHM | T3 | linear | 2 | 0.9484 | 0.0668 | 0.6564 | 0.5270 | 0.7857 | cum. return |
| TECHM | T3 | loo_time | 2 | 0.8906 | 0.1445 | 0.2526 | 0.0499 | 0.4553 | cum. return |
| TECHM | T3 | none | 2 | 0.8774 | 0.1108 | 0.3751 | 0.3312 | 0.4189 | cum. return |
| TECHM | T1 | linear | 3 | 0.7471 | 0.2881 | 0.6265 | 0.1016 | 0.9701 | cum. return |
| TECHM | T1 | loo_time | 3 | 0.7397 | 0.2713 | 0.5557 | 0.1016 | 0.8176 | cum. return |
| TECHM | T1 | none | 3 | 0.7187 | 0.2083 | 0.7380 | 0.5603 | 1.0618 | cum. return |
| TECHM | T2 | linear | 3 | 0.8035 | 0.0194 | 0.5245 | 0.0569 | 0.7839 | cum. return |
| TECHM | T2 | loo_time | 3 | 0.8446 | 0.0589 | 0.7310 | 0.0681 | 1.1663 | cum. return |
| TECHM | T2 | none | 3 | 0.9120 | 0.0424 | 0.7342 | 0.6383 | 0.9143 | cum. return |

## Mann-Whitney: neural vs linear (loo_time baseline, over seeds)

| key | neural | vs | val_p | test_p | test_diff | cliffs_test | test_p_holm |
|---|---|---|---|---|---|---|---|
| FOODS_3_124_CA_3_validation | I4 | I2 | 0.8000 | 0.8000 | 1.8996 | -0.3333 | 1.0000 |
| FOODS_3_124_CA_3_validation | I4 | I3 | 0.2000 | 0.2000 | -18.2688 | -1.0000 | 1.0000 |
| FOODS_2_128_WI_2_validation | I4 | I2 | 1.0000 | 1.0000 | 1.5057 | 0.0000 | 1.0000 |
| FOODS_2_128_WI_2_validation | I4 | I3 | 0.2000 | 0.4000 | 145.4074 | 0.6667 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I4 | I2 | 0.2000 | 0.8000 | -0.1068 | -0.3333 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I4 | I3 | 0.8000 | 0.2000 | 5.8910 | 1.0000 | 1.0000 |
| TECHM | T3 | T1 | 0.8000 | 0.4000 | -0.3031 | -0.6667 | 1.0000 |
| TECHM | T3 | T2 | 0.8000 | 0.4000 | -0.4784 | -0.6667 | 1.0000 |

## Sensitivity (exploratory; fewer seeds and episodes than the linear study)

| policy | variable | value | n | val_J_mean | val_J_sd |
|---|---|---|---|---|---|
| I4 | clip | 1.0000 | 2 | 2.8189 | 0.0161 |
| I4 | clip | 5.0000 | 2 | 2.8061 | 0.0385 |
| I4 | clip | 25.0000 | 2 | 2.8616 | 0.0143 |
| I4 | gamma | 0.9000 | 2 | 2.9864 | 0.0776 |
| I4 | gamma | 0.9900 | 2 | 2.8061 | 0.0385 |
| I4 | gamma | 1.0000 | 2 | 2.2259 | 0.6368 |
| I4 | horizon | 30.0000 | 2 | 2.4606 | 0.1420 |
| I4 | horizon | 90.0000 | 2 | 2.8061 | 0.0385 |
| I4 | horizon | 180.0000 | 2 | 4.0519 | 0.0278 |
| I4 | lr | 0.0100 | 2 | 2.5558 | 0.0148 |
| I4 | lr | 0.0300 | 2 | 2.8061 | 0.0385 |
| I4 | lr | 0.0900 | 2 | 2.9143 | 0.1319 |
| I4 | sigma0 | 0.2500 | 2 | 2.3839 | 0.5124 |
| I4 | sigma0 | 0.5000 | 2 | 2.8061 | 0.0385 |
| I4 | sigma0 | 1.0000 | 2 | 2.9105 | 0.0579 |
| T3 | clip | 1.0000 | 2 | 0.7486 | 0.3147 |
| T3 | clip | 5.0000 | 2 | 0.8014 | 0.2430 |
| T3 | clip | 25.0000 | 2 | 0.8014 | 0.2430 |
| T3 | gamma | 0.9000 | 2 | 1.0517 | 0.0272 |
| T3 | gamma | 0.9900 | 2 | 0.6557 | 0.3648 |
| T3 | gamma | 1.0000 | 2 | 0.5791 | 0.2711 |
| T3 | horizon | 125.0000 | 2 | 0.3866 | 0.0943 |
| T3 | horizon | 250.0000 | 2 | 0.8014 | 0.2430 |
| T3 | horizon | 500.0000 | 2 | 1.8069 | 0.2307 |
| T3 | lr | 0.0100 | 2 | 0.5642 | 0.4720 |
| T3 | lr | 0.0300 | 2 | 0.8014 | 0.2430 |
| T3 | lr | 0.0900 | 2 | 0.8009 | 0.0427 |
