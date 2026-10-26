# REINFORCE (Monte Carlo policy gradient with baselines) on inventory control and TECHM trading

Everything here was written and run in this folder; all outputs are under `outputs/` (`outputs/quick/`, `outputs/full/`). The FDPG project folder was only **read** (text/CSV/JSON; no import, execution or unpickling) and its 305 files were SHA-256 hashed before and after (asserted identical).

## Data and assumptions
See `DATA_NOTES.md` (audit, MDP, costs) and `FDPG_INVENTORY.md` (what was reused from FDPG and compatibility).
- Trading `TECHM_extend.csv.gzip`: price-scale audit **fails** (`close` is adjusted, OHLC are not). Deviation, as in FDPG: adjusted `close` for returns (`allow_adjusted_close`). Dates run to 2026-10-23 as supplied.
- M5: 3 series (primary FOODS_3_124_CA_3, intermittent FOODS_2_128_WI_2, seasonal HOBBIES_1_371_WI_1), H=90, L=2, Q_max=7mu, 60/20/20 splits. TECHM H=250, 5 bps.
- Packages: numpy, pandas, scipy, matplotlib, pytest, pyyaml (pyyaml was already installed; nothing was installed or downloaded). torch is needed only for the optional neural policies (R-I4, R-T3), see "Neural policies" below.

## Method
Policies (all with analytic score functions, unit-tested against numerical gradients):
R-I1 Gaussian order-up-to multiplier, R-I2 Gaussian direct quantity (log-sigma learned, floor 0.05), R-I3 softmax over 9 order levels {0..7}mu, R-T1 Bernoulli long/flat (sigmoid of standardised features and previous position), R-T2 softmax over {0, 0.5, 1}.

Gradient: `g = 1/B sum_i sum_t grad log pi(a_it|s_it) (G_it - b_it)` with rewards-to-go `G` (gamma 0.99 inventory / 0.995 trading; the gamma^t factor is omitted, flag `discount_weight`), Adam (clip 5, cosine LR), optional entropy bonus. Rewards are normalised (see DATA_NOTES) so J is in "norm units".
Baselines: `none`, `loo_const` (leave-one-out constant), `loo_time` (leave-one-out per time step; main), `linear` (ridge value function fitted on the previous iteration, so unbiased), `loo_time_std` (advantage standardisation; slightly biased).
Equal-budget comparison methods on the same mean-policy parameters: random search, Nelder-Mead, SPSA, central finite differences (common paths). For R-T1 they optimise the continuous position p (= FDPG's F-T1 form) because the deterministic p>=0.5 rule is piecewise constant.
Selection is always by **validation** J of the deterministic policy (best iterate); test data are used only in `evaluate_business.py` and the FDPG comparison. Budget: 40,000 training episodes, 10 seeds (full); validation rollouts are not counted. Hyper-parameters (lr x B x entropy, 18 configs x 2 seeds x 20,000 episodes) tuned on validation with `loo_time`, reused for all baselines (`outputs/full/models/hparams_rf*.json`); comparison-method settings tuned likewise. Frozen plan: `analysis_plan.yaml`.

## Run
```
python audit_data.py ; python audit_fdpg.py
python run_reinforce.py --domain both --quick|--full [--sensitivity]
python evaluate_reinforce.py --quick|--full ; python evaluate_business.py --quick|--full ; python compare_with_fdpg.py --quick|--full
python -m pytest tests -q
```
Each script prints an estimated runtime first (full training ~50 min on 8 workers; evaluations a few minutes). Reports: `outputs/full/eval/EVALUATION.md`, `outputs/full/business/BUSINESS.md`, `outputs/full/compare/COMPARISON.md`.

## Results (full mode, 10 seeds; numbers in the reports above)
**Baselines / gradient quality** (primary inventory, trained theta, B=32): variance (trace of covariance) vs no baseline: `loo_const` 0.22x, `loo_time` 0.035x (I1) / 0.055x (I2), `linear` 0.03-0.04x; the bias check (distance to the no-baseline mean in standard errors) is about 1 for all, i.e. no detectable bias (`loo_time_std` is biased by design). On TECHM T1 baselines give no variance gain (returns are near zero-mean noise), and the gradient is essentially noise at the trained point (cosine to the large-batch reference about 0).
**Learning**: on inventory R-I2, no baseline is unstable (mean final validation J -7.5, one diverged-scale seed) while every baseline gives 2.5-2.9 (paired Wilcoxon, Holm p=0.008-0.029); for R-I1, R-I3 and the trading policies, baselines did **not** significantly change final validation J (all Holm p>0.5), though they reach 90% of the best J sooner on I1 (3.7k vs 8.5k episodes).
**REINFORCE vs equal-budget methods** (validation J): REINFORCE (loo_time) beats Nelder-Mead and central FD on R-I2 and TECHM T1 after Holm correction, but random search and SPSA are close on R-I1 (no significant difference); SPSA/random reach 90% of their best quickly on R-I1 (fewer episodes) but plateau lower.
**Test period, primary inventory** (mean 90-day profit, bootstrap 95% CI about +/-100): R-I3 ~798-800, R-I2 ~781-791, R-I1 ~757, tuned (s,S) 746, cover rule 744, train-tuned base-stock 679 (mistuned by a train/test regime shift, as FDPG found). R-I2/R-I3 beat the train-tuned (s,S) (paired bootstrap p<0.01). The **validation**-tuned (s,S) (not a fair baseline) reaches 821 and beats every learned policy.
**Test period, TECHM** (compounded, 5 bps): buy-and-hold +100%; R-T1 loo_time +81%, R-T2 +76%, derivative-free methods 62-79%. No learned policy beats buy-and-hold (block-bootstrap CIs of excess return contain 0). Walk-forward R-T1 pooled +537% vs buy-and-hold +534% (4 folds; 2 folds below, 2 above buy-and-hold): no reliable edge. The train-tuned RSI rule degenerates to always-long.
**vs FDPG** (`compare_with_fdpg.py`): reproduction check re-running FDPG's saved thetas here with FDPG's windows and no Q_max matched all 11 reported test numbers (REPRODUCED, to 1e-15). With evidence B (same environment, FDPG's best estimator chosen on validation), same-family comparisons are mixed: inventory R-I1 equals F-I1 on the primary series (757 vs 759, p=0.48), is better on seasonal (100 vs 95) and worse on intermittent (166 vs 170); R-I3 beats FDPG on primary/seasonal (798 vs 767, 104 vs 98) but is far worse on the intermittent series (83 vs 167, softmax stuck on too-coarse order levels); FDPG's F-I2 (s,S) beats R-I2 on primary (809 vs 781). TECHM: R-T1 0.81 vs F-T1 0.69, not significant (Holm p=0.31); FDPG's F-T2 (0.99) beats R-T2 (0.76). FDPG's curves (evidence A) are in its own normalisation and only indicative.

## Caveats
- Seeds, code and tuning effort differ between projects (Mann-Whitney, not paired); R-I2/R-I3/R-T2 are not the same families as F-I2/F-I3/F-T2.
- Results are single-instrument/3-series; 10 seeds give wide intervals (e.g. R-I2 seed sd 0.15 in J).
- Simple baselines win in places: buy-and-hold on TECHM, validation-tuned (s,S) on inventory, and FDPG's structured policies (F-I2, F-T2) on several problems.
- Adjusted-close deviation and the 2026 date range in the trading file affect absolute return levels.
- Q_max-free variant, walk-forward vs FDPG, neural vs FDPG and the price-scale audit are done (see `FINAL_REPORT.md`); all are exploratory (`analysis_plan.yaml: exploratory_added_after_freeze`).
- Neural policies use a reduced tuning grid (1 tuning seed); FDPG comparisons are unpaired (different seeds/code).
- Sensitivity (gamma, clip, lr, B, sigma0, horizon) is exploratory: `outputs/full/eval/sensitivity_summary.csv`.

## Neural policies (R-I4, R-T3) - exploratory, PyTorch
- Policy mean/logit = linear skip + one 16-unit tanh hidden layer (float64 CPU; per-sample score gradients via `torch.func.vmap(grad)`). R-I4 is Gaussian over order quantity (R-I2 features); R-T3 is Bernoulli over the R-T1 features. Gradients are unit-tested against finite differences.
- Design: baselines none/loo_time/linear, 10 seeds, small tuning grid (lr {0.01,0.03}, B 32, entropy {0,0.01}), equal-budget SPSA and random-search on the same network (cm-spsa, cm-random), exploratory sensitivity (2 seeds, 20k episodes). Outputs: `outputs/neural_full/` (earlier 5-seed run: `outputs/neural_full_5seed/`).
- Run: `python -W ignore run_reinforce.py --neural --full --sensitivity --workers 10`, then `evaluate_neural.py --full` and `compare_neural_fdpg.py --full` (`--quick` for smoke tests).
- Extensions: `walkforward_compare.py --full` (FDPG fold structure), `run_reinforce.py --qmax-free` + `evaluate_qfree.py`, `run_reinforce.py --return-source open` + `evaluate_priceaudit.py`.
- 10-seed result summary: see `FINAL_REPORT.md` and `outputs/neural_full/eval/NEURAL.md`; the older 5-seed text below is superseded where numbers differ.
- Result (`outputs/neural_full/eval/neural_vs_linear.csv`): R-I4 is about on par with R-I2/R-I3 (best on FOODS_3 val J 3.06 and HOBBIES val J 3.34, test profit CIs overlap with R-I3; not better on FOODS_2). R-T3 has higher validation J (1.02-1.08 vs 0.93-0.98) but its test cumulative return (0.42-0.65) is no better and often below R-T1/R-T2 (0.62-0.81): no evidence of a gain, consistent with overfitting; still below buy-and-hold.
