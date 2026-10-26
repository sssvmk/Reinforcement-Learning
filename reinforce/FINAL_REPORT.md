# Final report: FDPG vs REINFORCE (linear and neural) on M5 inventory and TECHM trading

All numbers are test results, mean over seeds (10 unless stated). Neural, walk-forward, Q_max-free and price-audit work is EXPLORATORY (added after the analysis plan was frozen).

## Headline
1. **No learned TECHM policy beats buy-and-hold** (test cumulative return +100%): R-T1 ~0.81, R-T2 ~0.76, R-T3 (neural) ~0.57, FDPG F-T1 ~0.69, F-T2 0.99 (deterministic, no seed variance). Same conclusion under open-to-open returns and in walk-forward.
2. **Inventory: structured policies are as good as anything.** FDPG's F-I2 (FOODS_3 809, FOODS_2 172, HOBBIES 102) is at or above REINFORCE R-I2 (781 / 164 / 96) and the neural R-I4 (798 / 163 / 102).
3. **Neural policies show no clear gain.** R-I4 beats R-I2 only on FOODS_3 (+17.5, Holm p=0.029); it is equal to R-I3 and not better elsewhere. R-T3 has higher validation J but lower, noisier test return than R-T1 (p=0.054, not significant after Holm) which suggests overfitting.
4. **REINFORCE vs equal-budget derivative-free methods on the same network:** REINFORCE (linear/loo_time baseline) beats SPSA and random search on FOODS_3 and FOODS_2 (Holm-significant on FOODS_3). On HOBBIES random search is better (109 vs 101, p<0.001). On TECHM, SPSA/random have lower validation J but higher test return (0.87/0.79 vs 0.57), another sign of validation overfitting by REINFORCE.
5. **Baselines matter for variance:** `none` has larger seed sd on FOODS_3 (J sd 0.29 vs 0.03 with loo_time/linear).

## Walk-forward (FDPG's 3 expanding folds; fold dates identical to FDPG's CSV)
Pooled cumulative return (5 seeds): buy-and-hold 1.075, RSI rule 1.091, FDPG F-T2 1.060, FDPG F-T1 0.846, R-T1 0.905, R-T2 0.778, R-T3 0.686, central-FD 0.796, random 0.740, SPSA 0.629. No learned method beats buy-and-hold; differences among learned methods are not significant (all p>0.05 vs R-T1). FDPG rows come from FDPG's own saved CSVs (not paired with ours).

## Q_max-free REINFORCE (`outputs/qfree/eval/QMAX_FREE.md`)
Removing the 7·mu order cap changes nothing for R-I1 (cap never binds) and little for R-I2 (e.g. FOODS_3 768 vs 781, within seed sd 3-27); <0.3% of orders exceed 7·mu. The cap is not driving results.

## TECHM price-scale audit (`outputs/ropen/eval/PRICE_AUDIT.md`)
The level check fails (adjusted close vs raw mid). The close/mid ratio rises smoothly 0.65 (2007) to 0.97 (2026) with no step, consistent with cumulative dividend adjustment (total-return series), not a split. This is an inference: **please confirm with the data owner.** Adjusted close is kept as the primary series. Using open-to-open returns lowers absolute numbers (buy-and-hold 0.80, R-T1 0.44 trained and evaluated on open) but rankings are unchanged. Also note file dates run to 2026-10-23.

## Caveats
- Seeds, code and tuning differ between FDPG and REINFORCE: cross-project tests are unpaired (Mann-Whitney); within-REINFORCE comparisons share seeds and initial parameters (paired Wilcoxon).
- Neural tuning grid is small (1 tuning seed); neural sensitivity uses 2 seeds / 20k episodes (`outputs/neural_full/eval/sensitivity_summary.csv`).
- Single instrument and 3 inventory series.
- FDPG folder was only read (hashes unchanged).

## Where to look
`outputs/full/eval` (linear), `outputs/neural_full/eval` and `.../compare/NEURAL_VS_FDPG.md`, `outputs/wf_full/WALKFORWARD.md`, `outputs/qfree/eval`, `outputs/ropen/eval`, `FDPG_INVENTORY.md`, `analysis_plan.yaml`.
