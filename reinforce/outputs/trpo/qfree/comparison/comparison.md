# TRPO vs previous methods (qfree mode)

Baselines are **re-used** from `outputs/qfree` of this folder (same seeds, same theta0). TRPO counts pilot + training batches against the same budget; validation/diagnostic rollouts are excluded for all methods. Tuning effort differs: TRPO tunes delta x lambda (12 configs) and VPG lr (3); REINFORCE tuned lr x B x entropy (18).

Previous results unchanged during comparison (hash check of outputs/full): **True**

## Summary (validation return, deterministic policy, last checkpoint)

| key | policy | method | n | final | ci_lo | ci_hi | eps95_median | reached95 |
|---|---|---|---|---|---|---|---|---|
| FOODS_2_128_WI_2_validation | I1 | cm:central | 10 | 0.3648 | 0.3522 | 0.3803 | 1280.0000 | 0.4000 |
| FOODS_2_128_WI_2_validation | I1 | cm:neldermead | 10 | 0.7101 | 0.7084 | 0.7114 | 2048.0000 | 0.9000 |
| FOODS_2_128_WI_2_validation | I1 | cm:random | 10 | 0.4772 | 0.4427 | 0.5056 | 1536.0000 | 0.6000 |
| FOODS_2_128_WI_2_validation | I1 | cm:spsa | 10 | 0.7158 | 0.7070 | 0.7240 | 3072.0000 | 1.0000 |
| FOODS_2_128_WI_2_validation | I1 | reinforce:loo_time | 10 | 0.7210 | 0.6961 | 0.7430 | 2480.0000 | 0.6000 |
| FOODS_2_128_WI_2_validation | I1 | reinforce:none | 10 | 0.7328 | 0.7112 | 0.7520 | 5952.0000 | 0.5000 |
| FOODS_2_128_WI_2_validation | I1 | trpo:trpo | 10 | 0.6775 | 0.6594 | 0.6972 | 3008.0000 | 1.0000 |
| FOODS_2_128_WI_2_validation | I1 | trpo:vpg | 10 | 0.7471 | 0.7339 | 0.7593 | 5984.0000 | 0.7000 |
| FOODS_2_128_WI_2_validation | I2 | cm:central | 10 | 0.3367 | 0.3078 | 0.3594 | nan | 0.0000 |
| FOODS_2_128_WI_2_validation | I2 | cm:neldermead | 10 | 0.5382 | 0.4977 | 0.5722 | nan | 0.0000 |
| FOODS_2_128_WI_2_validation | I2 | cm:random | 10 | 0.3948 | 0.3353 | 0.4561 | nan | 0.0000 |
| FOODS_2_128_WI_2_validation | I2 | cm:spsa | 10 | 0.3220 | 0.2541 | 0.3900 | 2048.0000 | 0.2000 |
| FOODS_2_128_WI_2_validation | I2 | reinforce:loo_time | 10 | 0.6496 | 0.6079 | 0.6895 | 5456.0000 | 0.8000 |
| FOODS_2_128_WI_2_validation | I2 | reinforce:none | 10 | -2731465011756176404119552.0000 | -8194395035268528675487744.0000 | 0.6270 | 8928.0000 | 0.7000 |
| FOODS_2_128_WI_2_validation | I2 | trpo:trpo | 10 | 0.7572 | 0.7260 | 0.7841 | 10944.0000 | 1.0000 |
| FOODS_2_128_WI_2_validation | I2 | trpo:vpg | 10 | -30485786332809123848129953289460046942438940233789352181760.0000 | -91457358998427360394017260603068570059457684377187303555072.0000 | 0.6319 | 2016.0000 | 0.5000 |
| FOODS_3_124_CA_3_validation | I1 | cm:central | 10 | 2.4529 | 2.4381 | 2.4674 | 2560.0000 | 0.7000 |
| FOODS_3_124_CA_3_validation | I1 | cm:neldermead | 10 | 2.4167 | 2.3730 | 2.4604 | 0.0000 | 0.3000 |
| FOODS_3_124_CA_3_validation | I1 | cm:random | 10 | 2.4988 | 2.4727 | 2.5286 | 1024.0000 | 0.8000 |
| FOODS_3_124_CA_3_validation | I1 | cm:spsa | 10 | 2.6066 | 2.5937 | 2.6203 | 1024.0000 | 0.9000 |
| FOODS_3_124_CA_3_validation | I1 | reinforce:loo_time | 10 | 2.7925 | 2.7754 | 2.8070 | 5280.0000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I1 | reinforce:none | 10 | 2.8659 | 2.8544 | 2.8774 | 8640.0000 | 0.9000 |
| FOODS_3_124_CA_3_validation | I1 | trpo:trpo | 10 | 2.8143 | 2.8027 | 2.8268 | 2016.0000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I1 | trpo:vpg | 10 | 2.8618 | 2.8436 | 2.8785 | 1024.0000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I2 | cm:central | 10 | 2.6943 | 2.6621 | 2.7241 | 16896.0000 | 0.5000 |
| FOODS_3_124_CA_3_validation | I2 | cm:neldermead | 10 | 2.5586 | 2.4924 | 2.6107 | 8192.0000 | 0.1000 |
| FOODS_3_124_CA_3_validation | I2 | cm:random | 10 | 2.7234 | 2.6820 | 2.7646 | 15360.0000 | 0.7000 |
| FOODS_3_124_CA_3_validation | I2 | cm:spsa | 10 | 2.4628 | 2.3281 | 2.5626 | 1024.0000 | 0.6000 |
| FOODS_3_124_CA_3_validation | I2 | reinforce:loo_time | 10 | -573771199881622960052300936506094516026084100985563045794330563291946009737801895385003192221696.0000 | -1721313599644868880156902809518283548078252302956689137382991689875838029213405686155009576665088.0000 | 2.6038 | 2976.0000 | 0.9000 |
| FOODS_3_124_CA_3_validation | I2 | reinforce:none | 10 | -524407187209019505341080677561306636227167024199957462951194844138193054340881580032.0000 | -1573221561627049565320948538671508079232492182082488741321636248944515436624042000384.0000 | 2.5596 | 1984.0000 | 0.9000 |
| FOODS_3_124_CA_3_validation | I2 | trpo:trpo | 10 | 2.7785 | 2.7097 | 2.8468 | 1024.0000 | 1.0000 |
| FOODS_3_124_CA_3_validation | I2 | trpo:vpg | 10 | 2.8187 | 2.6643 | 2.9404 | 1024.0000 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I1 | cm:central | 10 | 2.9813 | 2.9559 | 3.0050 | 7680.0000 | 0.9000 |
| HOBBIES_1_371_WI_1_validation | I1 | cm:neldermead | 10 | 2.0468 | 1.4089 | 2.6294 | nan | 0.0000 |
| HOBBIES_1_371_WI_1_validation | I1 | cm:random | 10 | 3.0113 | 2.9926 | 3.0286 | 3584.0000 | 0.4000 |
| HOBBIES_1_371_WI_1_validation | I1 | cm:spsa | 10 | 3.0088 | 2.9872 | 3.0298 | 4096.0000 | 0.2000 |
| HOBBIES_1_371_WI_1_validation | I1 | reinforce:loo_time | 10 | 3.1617 | 3.1186 | 3.2049 | 2480.0000 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I1 | reinforce:none | 10 | 3.0525 | 3.0151 | 3.0968 | 2976.0000 | 0.9000 |
| HOBBIES_1_371_WI_1_validation | I1 | trpo:trpo | 10 | 3.2803 | 3.2577 | 3.3045 | 3008.0000 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I1 | trpo:vpg | 10 | 3.1084 | 3.0380 | 3.1800 | 1520.0000 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I2 | cm:central | 10 | 3.0190 | 2.9731 | 3.0554 | nan | 0.0000 |
| HOBBIES_1_371_WI_1_validation | I2 | cm:neldermead | 10 | 2.6206 | 2.5753 | 2.6566 | nan | 0.0000 |
| HOBBIES_1_371_WI_1_validation | I2 | cm:random | 10 | 2.9968 | 2.9524 | 3.0464 | 9216.0000 | 0.3000 |
| HOBBIES_1_371_WI_1_validation | I2 | cm:spsa | 10 | 2.8779 | 2.8391 | 2.9138 | 1024.0000 | 0.3000 |
| HOBBIES_1_371_WI_1_validation | I2 | reinforce:loo_time | 10 | -330254073912182299053184468659335895445627726216730626752512.0000 | -990762006688306829547047585137342882017904980649731719954432.0000 | -412405481053897952854016.0000 | 2976.0000 | 0.8000 |
| HOBBIES_1_371_WI_1_validation | I2 | reinforce:none | 10 | -295977881543351847694742839361047651487572959424160631624372154095632384.0000 | -887933644630055248845084671832482149660089665413118640075205003035279360.0000 | 1.4638 | 4960.0000 | 0.7000 |
| HOBBIES_1_371_WI_1_validation | I2 | trpo:trpo | 10 | 3.2417 | 3.1873 | 3.2865 | 2512.0000 | 1.0000 |
| HOBBIES_1_371_WI_1_validation | I2 | trpo:vpg | 10 | -37039534782185721026363305434880764320850680879937068512863120366592711741068365791232.0000 | -111118604346557156177343569514078505527796180362785753087480388929391580060680873574400.0000 | 2.9303 | 2016.0000 | 1.0000 |

## TRPO vs each method (paired by seed, Wilcoxon, Holm over all rows)

| key | policy | vs | n_pairs | diff_trpo_minus | p_wilcoxon_paired | p_mannwhitney | cliffs_delta | p_holm | verdict |
|---|---|---|---|---|---|---|---|---|---|
| FOODS_2_128_WI_2_validation | I1 | cm:central | 10 | 0.3127 | 0.0020 | 0.0002 | 1.0000 | 0.0820 | no significant difference |
| FOODS_2_128_WI_2_validation | I1 | cm:neldermead | 10 | -0.0326 | 0.0195 | 0.0257 | -0.6000 | 0.1953 | no significant difference |
| FOODS_2_128_WI_2_validation | I1 | cm:random | 10 | 0.2003 | 0.0020 | 0.0002 | 1.0000 | 0.0820 | no significant difference |
| FOODS_2_128_WI_2_validation | I1 | cm:spsa | 10 | -0.0383 | 0.0039 | 0.0113 | -0.6800 | 0.0820 | no significant difference |
| FOODS_2_128_WI_2_validation | I1 | reinforce:loo_time | 10 | -0.0434 | 0.0488 | 0.0140 | -0.6600 | 0.3418 | no significant difference |
| FOODS_2_128_WI_2_validation | I1 | reinforce:none | 10 | -0.0553 | 0.0098 | 0.0046 | -0.7600 | 0.1074 | no significant difference |
| FOODS_2_128_WI_2_validation | I1 | trpo:vpg | 10 | -0.0696 | 0.0020 | 0.0004 | -0.9400 | 0.0820 | no significant difference |
| FOODS_2_128_WI_2_validation | I2 | cm:central | 10 | 0.4205 | 0.0020 | 0.0002 | 1.0000 | 0.0820 | no significant difference |
| FOODS_2_128_WI_2_validation | I2 | cm:neldermead | 10 | 0.2190 | 0.0020 | 0.0002 | 1.0000 | 0.0820 | no significant difference |
| FOODS_2_128_WI_2_validation | I2 | cm:random | 10 | 0.3624 | 0.0020 | 0.0002 | 1.0000 | 0.0820 | no significant difference |
| FOODS_2_128_WI_2_validation | I2 | cm:spsa | 10 | 0.4352 | 0.0020 | 0.0002 | 1.0000 | 0.0820 | no significant difference |
| FOODS_2_128_WI_2_validation | I2 | reinforce:loo_time | 10 | 0.1076 | 0.0059 | 0.0017 | 0.8400 | 0.0820 | no significant difference |
| FOODS_2_128_WI_2_validation | I2 | reinforce:none | 10 | 2731465011756176404119552.0000 | 0.0039 | 0.0013 | 0.8600 | 0.0820 | no significant difference |
| FOODS_2_128_WI_2_validation | I2 | trpo:vpg | 10 | 30485786332809123848129953289460046942438940233789352181760.0000 | 0.0020 | 0.0013 | 0.8600 | 0.0820 | no significant difference |
| FOODS_3_124_CA_3_validation | I1 | cm:central | 10 | 0.3614 | 0.0020 | 0.0002 | 1.0000 | 0.0820 | no significant difference |
| FOODS_3_124_CA_3_validation | I1 | cm:neldermead | 10 | 0.3976 | 0.0020 | 0.0002 | 1.0000 | 0.0820 | no significant difference |
| FOODS_3_124_CA_3_validation | I1 | cm:random | 10 | 0.3155 | 0.0020 | 0.0002 | 1.0000 | 0.0820 | no significant difference |
| FOODS_3_124_CA_3_validation | I1 | cm:spsa | 10 | 0.2078 | 0.0020 | 0.0002 | 1.0000 | 0.0820 | no significant difference |
| FOODS_3_124_CA_3_validation | I1 | reinforce:loo_time | 10 | 0.0218 | 0.1602 | 0.1405 | 0.4000 | 0.5234 | no significant difference |
| FOODS_3_124_CA_3_validation | I1 | reinforce:none | 10 | -0.0516 | 0.0020 | 0.0004 | -0.9400 | 0.0820 | no significant difference |
| FOODS_3_124_CA_3_validation | I1 | trpo:vpg | 10 | -0.0475 | 0.0020 | 0.0028 | -0.8000 | 0.0820 | no significant difference |
| FOODS_3_124_CA_3_validation | I2 | cm:central | 10 | 0.0842 | 0.1309 | 0.1041 | 0.4400 | 0.5234 | no significant difference |
| FOODS_3_124_CA_3_validation | I2 | cm:neldermead | 10 | 0.2199 | 0.0039 | 0.0022 | 0.8200 | 0.0820 | no significant difference |
| FOODS_3_124_CA_3_validation | I2 | cm:random | 10 | 0.0551 | 0.3223 | 0.2730 | 0.3000 | 0.6445 | no significant difference |
| FOODS_3_124_CA_3_validation | I2 | cm:spsa | 10 | 0.3157 | 0.0039 | 0.0003 | 0.9600 | 0.0820 | no significant difference |
| FOODS_3_124_CA_3_validation | I2 | reinforce:loo_time | 10 | 573771199881622960052300936506094516026084100985563045794330563291946009737801895385003192221696.0000 | 0.0488 | 0.0539 | 0.5200 | 0.3418 | no significant difference |
| FOODS_3_124_CA_3_validation | I2 | reinforce:none | 10 | 524407187209019505341080677561306636227167024199957462951194844138193054340881580032.0000 | 0.0645 | 0.1859 | 0.3600 | 0.3418 | no significant difference |
| FOODS_3_124_CA_3_validation | I2 | trpo:vpg | 10 | -0.0402 | 0.4316 | 0.2730 | -0.3000 | 0.6445 | no significant difference |
| HOBBIES_1_371_WI_1_validation | I1 | cm:central | 10 | 0.2990 | 0.0020 | 0.0002 | 1.0000 | 0.0820 | no significant difference |
| HOBBIES_1_371_WI_1_validation | I1 | cm:neldermead | 10 | 1.2335 | 0.0020 | 0.0002 | 1.0000 | 0.0820 | no significant difference |
| HOBBIES_1_371_WI_1_validation | I1 | cm:random | 10 | 0.2690 | 0.0020 | 0.0002 | 1.0000 | 0.0820 | no significant difference |
| HOBBIES_1_371_WI_1_validation | I1 | cm:spsa | 10 | 0.2715 | 0.0020 | 0.0002 | 1.0000 | 0.0820 | no significant difference |
| HOBBIES_1_371_WI_1_validation | I1 | reinforce:loo_time | 10 | 0.1186 | 0.0039 | 0.0010 | 0.8800 | 0.0820 | no significant difference |
| HOBBIES_1_371_WI_1_validation | I1 | reinforce:none | 10 | 0.2278 | 0.0020 | 0.0002 | 0.9800 | 0.0820 | no significant difference |
| HOBBIES_1_371_WI_1_validation | I1 | trpo:vpg | 10 | 0.1719 | 0.0039 | 0.0017 | 0.8400 | 0.0820 | no significant difference |
| HOBBIES_1_371_WI_1_validation | I2 | cm:central | 10 | 0.2227 | 0.0020 | 0.0003 | 0.9600 | 0.0820 | no significant difference |
| HOBBIES_1_371_WI_1_validation | I2 | cm:neldermead | 10 | 0.6212 | 0.0020 | 0.0002 | 1.0000 | 0.0820 | no significant difference |
| HOBBIES_1_371_WI_1_validation | I2 | cm:random | 10 | 0.2450 | 0.0020 | 0.0004 | 0.9400 | 0.0820 | no significant difference |
| HOBBIES_1_371_WI_1_validation | I2 | cm:spsa | 10 | 0.3638 | 0.0020 | 0.0002 | 1.0000 | 0.0820 | no significant difference |
| HOBBIES_1_371_WI_1_validation | I2 | reinforce:loo_time | 10 | 330254073912182299053184468659335895445627726216730626752512.0000 | 0.0039 | 0.0006 | 0.9200 | 0.0820 | no significant difference |
| HOBBIES_1_371_WI_1_validation | I2 | reinforce:none | 10 | 295977881543351847694742839361047651487572959424160631624372154095632384.0000 | 0.0195 | 0.0073 | 0.7200 | 0.1953 | no significant difference |
| HOBBIES_1_371_WI_1_validation | I2 | trpo:vpg | 10 | 37039534782185721026363305434880764320850680879937068512863120366592711741068365791232.0000 | 0.0273 | 0.0036 | 0.7800 | 0.2188 | no significant difference |

## Verdict counts

| vs | no significant difference |
|---|---|
| cm:central | 6 |
| cm:neldermead | 6 |
| cm:random | 6 |
| cm:spsa | 6 |
| reinforce:loo_time | 6 |
| reinforce:none | 6 |
| trpo:vpg | 6 |