# TRPO on M5 inventory control and TECHM trading

Phase 3 after FDPG and REINFORCE, same problems, splits, policies, seeds and budgets. Code: `trpo/` (package), `run_trpo.py`, `evaluate_trpo.py`, `evaluate_business_trpo.py`, `compare_with_previous.py`, `tests/test_trpo.py`. Outputs: `outputs/trpo/{quick,full}/` (`models`, `runs`, `iters`, `algo_eval`, `business_eval`, `comparison`). Plan: `analysis_plan_trpo.yaml`.

## Data (given by the user)
- TECHM: `C:/Users/username/OneDrive - XXYY/Muni/project/India/TECHM/TECHM_extend.csv.gzip` (adjusted close; the price-scale audit from phase 2 failed and the series is **still unconfirmed with the data owner**).
- M5: `C:\Users\username\OneDrive - XXYY\m5-forecasting-accuracy\` (FOODS_3_124_CA_3, FOODS_2_128_WI_2, HOBBIES_1_371_WI_1).
- Earlier phases: FDPG folder (read-only) and this folder's `outputs/full` (REINFORCE and derivative-free runs, re-used as baselines; hash-checked unchanged).

## About TRPO (background)
TRPO (Trust Region Policy Optimization; Schulman et al. 2015) is a policy-gradient method that limits how far each update can move the policy, measured by the KL divergence between the old and new action distributions. The limit gives stable, steady improvement where plain policy gradient can take one bad step and collapse. This section is written from background knowledge, not a fresh literature search.

**Where it has been applied**
- *Simulated robot locomotion and control.* Its original showcase was MuJoCo tasks such as hopping, swimming and walking, with neural-network policies. It is a standard baseline in continuous-control benchmarks.
- *Atari games.* The original paper also showed it learning from raw pixels with a convolutional policy.
- *Robot manipulation and sim-to-real work.* It was used for learning grasping, reaching and similar skills in simulation, where an unstable update can waste a long training run.
- *Research on stable RL.* It is the basis for later methods, especially PPO, which keeps the same idea with a simpler clipped objective. The same line of work shaped the policy optimization used in RLHF for language models, though PPO is what is mostly used there.
- *Operations and control problems with simulators.* In principle any problem with a stochastic policy and a simulator can use it. Examples are inventory ordering, pricing, scheduling and energy control, when a stable, hands-off learner matters more than speed.

**Why people chose it**
- Updates are bounded, so training is less sensitive to learning-rate choice.
- It is theoretically motivated, with an approximate guarantee that each update does not make the policy worse.
- It works with continuous actions and large neural policies.

**Limits**
- Complex and costly per update. It uses second-order information (conjugate gradient and a line search on the KL constraint), which is more work than first-order methods.
- Awkward with some architectures. Sharing parameters between the policy and the value network, or using dropout, doesn't fit it well.
- On-policy and sample-hungry, like REINFORCE.
- PPO has largely replaced it in practice, because PPO is simpler and performs similarly.

**Relation to these projects.** For the inventory and trading work, TRPO is a natural step after REINFORCE with a baseline: the same stochastic policies and simulators, but with a controlled step size. With policies of only about 10 parameters, it was expected to give little over a well-tuned REINFORCE or finite-difference method, so it is treated as a research comparison, not an expected improvement (the results below are consistent with this).

## What is TRPO here
Trust Region Policy Optimization (Schulman et al. 2015) with linear/tabular stochastic policies R-I1 (Gaussian base-stock), R-I2, R-I3 (softmax), R-T1 (Bernoulli), R-T2 (softmax).
Per iteration: sample B=32 episodes; GAE advantages from a ridge-linear value function fitted *before* the batch (V is refit afterwards on this batch plus the last 3); policy gradient g of the surrogate L(θ)=mean(π_θ/π_old · A); direction s=(F+dI)⁻¹g by 20 conjugate-gradient steps on analytic Fisher-vector products, d = 0.01·mean diag F; step β_max = sqrt(2δ/sᵀFs); backtracking line search (10 halvings) that accepts the first candidate with exact mean KL ≤ δ and surrogate gain > 0 (the σ floor is projected before the check).
Ablations: `vpg` (same advantages, Adam, no trust region), `ng_fixed` (natural gradient, fixed step, no KL check), `trpo_mc` (λ=1), `trpo_nostd` (no advantage standardisation).
Design choices: KL ignores order clipping (pre-clip Gaussian sample); Fisher and surrogate are means over all (episode, step) samples, so δ is a mean per-state KL; no γ^t weighting (as REINFORCE); V(s_H) uses the last features with time_remaining=0 (approximation); no entropy bonus in the main runs (see Extensions).

## Run
```
python -W ignore -m pytest tests -q                      # 58 tests
python -W ignore run_trpo.py --domain both --full --sensitivity --workers 8   # ~97 min on 8 workers (quick: ~5 min)
python -W ignore evaluate_trpo.py --full
python -W ignore compare_with_previous.py --full
python -W ignore evaluate_business_trpo.py --full
```
Budget: 40,000 training episodes per run (pilot batch included); validation and diagnostic rollouts are excluded, as for all other methods. 10 seeds, same seeds/θ0 as REINFORCE. Hyper-parameters (δ∈{.003,.01,.03,.1} × λ∈{.9,.97,1}; VPG lr∈{.01,.03,.1}) chosen on validation only (2 seeds, half budget).

## Results (full mode; validation return in normalised units, last checkpoint, deterministic policy)
- **Trust region works as designed.** All TRPO steps were accepted; realised mean KL ≤ δ every iteration (e.g. mean 0.005, max 0.010 at δ=0.01); ~0.6 backtracks per step; 100% of accepted steps raised the surrogate. CG matches the exact damped solve (cosine 0.98–1.00; ≥0.997 on inventory).
- **But the surrogate does not predict realised gain.** Correlation between surrogate improvement and realised common-path return change is ≈0 (−0.06…0.26). The value function is weak (R² 0.06 on TECHM, 0.3–0.8 on inventory). The monotone-improvement guarantee is only approximate here (sampled surrogate, noisy returns).
- **Stability.** TRPO has the smaller worst single-iteration drop than VPG on FOODS_2 I2 (−0.06 vs −0.56), FOODS_3 I2 (−0.17 vs −0.65), I3 on FOODS_2/FOODS_3; VPG drops less on TECHM and a few I1 cases. 21 of 44 ablation comparisons have raw p<0.05 (mostly λ=1), but none survive Holm (see `algo_eval/ablation_tests.csv`).
- **vs REINFORCE and derivative-free methods** (`comparison/comparison.md`): TRPO final validation return is ahead of random/SPSA/Nelder-Mead/central-FD on every inventory problem, and at or above REINFORCE except small deficits on the I1 problems of FOODS_2/FOODS_3 (differences 0.04–0.05, raw p<0.05). REINFORCE (loo_time) and `none` often have a collapsed *last* checkpoint on I2 (final return −5…−23), which TRPO and VPG avoid. On TECHM, REINFORCE loo_time (0.81) ≥ TRPO (0.77) > VPG (0.67) on T1, no significant difference. **After Holm correction over all 80 paired comparisons none is significant** (n=10 paired seeds can give p≥0.002; 80 tests make the correction very strict), so claims are directional only.
- **Ablations.** λ=1 (Monte-Carlo advantage) is clearly worse than GAE (9 of 11 problems raw p<0.05, return −0.1…−1.0). Standardisation matters little on inventory, helps on TECHM T1 (0.92 vs 0.82 in sensitivity). The fixed-step natural gradient is often slightly *better* than TRPO on I1/I2 (δ-limited steps are conservative) but has far worse worst-case steps (e.g. −17 on one I3 run in quick mode).
- **Sensitivity** (primary: FOODS_3 I2, TECHM T1; 5 seeds; `algo_eval/algo_eval.md`): inventory is flat in δ (0.003→0.1: 2.98→3.03), CG iterations (2–20), σ₀, damping and standardisation; B=8 hurts (2.78); λ=0 hurts (2.69). TECHM is noisy (sd ≈0.15) and prefers larger B and larger δ; γ=1 hurts. Horizon changes the problem (returns not comparable) and is shown only for completeness.
- **Business (test period)** (`business_eval/BUSINESS.md`): on TECHM no learned policy beats buy-and-hold (cum. return 0.99): R-T1 TRPO 0.46, VPG 0.76, REINFORCE 0.82; walk-forward TRPO R-T1 pooled +370% vs buy-and-hold +533%. TRPO is **worse** than REINFORCE and VPG on TECHM test despite similar validation return, consistent with validation overfitting on a single instrument. Inventory profit tables with integer-rounded orders are in the same file.
- **Cost.** An update costs ~3 ms (I1/I2), ~30–40 ms (softmax I3/T2) vs ~10–35 ms of simulation per batch, so TRPO's extra cost is small relative to rollouts for these small policies.

## Honest caveats
One instrument and three M5 series; 10 seeds; unequal tuning effort between projects (TRPO 12+3 configurations, REINFORCE 18, derivative-free fewer); cross-project comparison is unpaired apart from re-used seeds/θ0; "final" = last checkpoint (REINFORCE collapses at the last checkpoint are partly a checkpoint-selection effect, its best-checkpoint numbers in `outputs/full` look better); TECHM price series unconfirmed; the KL ignores order clipping; standardisation biases the gradient slightly; sensitivity seeds (200+) differ from the main-run seeds so sensitivity baselines differ slightly from main results.

## Not implemented
Dedicated tests for hash/read-only checks (done at run time by `compare_with_previous.py`, which asserts the reused results folder is unchanged) and for splits (covered in earlier test suites). Sensitivity sweeps for the three extensions below.

## Extensions (added later, EXPLORATORY; see `analysis_plan_trpo.yaml`)
Run (each ~7–150 min; outputs in `outputs/trpo/{qfree,ent,neural_full}`):
```
python -W ignore run_trpo.py --domain both --full --qmax-free
python -W ignore run_trpo.py --domain both --full --entropy
python -W ignore run_trpo.py --domain both --full --neural
python -W ignore evaluate_trpo.py --full --set qfree|entropy|neural
python -W ignore compare_with_previous.py --full --set qfree|entropy|neural
python -W ignore evaluate_extensions_trpo.py --full --set qfree|entropy|neural   # validation + TEST period
```
Tests: `tests/test_trpo_ext.py` (9; 58 in total).

Learning curves: `python plot_learning_curves.py` writes PNGs to `outputs/trpo/figures/{main,neural,qfree,entropy}/<problem>__<policy>.png` (mean validation return over seeds vs episodes; TRPO with a 95% band; the main set shows TRPO, VPG, REINFORCE loo_time, SPSA and random search; the y-axis is clipped after the first 20% of training, so early or collapsed values can be off-scale; diverged runs are excluded from the plots).

**Implementation.** *Neural* (PyTorch MLP policies R-I4/R-T3): Fisher is the exact Gauss-Newton form JᵀWJ with the per-sample Jacobian cached once per batch (torch.func), so CG Fisher-vector products are cheap; the KL is closed-form per state (Gaussian with learned log σ / Bernoulli); checked against finite differences. *Entropy bonus*: the mean per-state entropy times c is added to the surrogate and its gradient (c=0 is bit-identical to plain TRPO). *Q_max-free*: env var `QMAX_FREE=1` removes the order cap for training and evaluation.

**Q_max-free (I1/I2, 3 series, 10 seeds).** Removing the cap did not hurt TRPO: validation returns are the same as for the capped policies (the cap was not binding for I1: identical thetas), and ≤1.5% of orders exceed 7μ. Without the cap the last-checkpoint return of REINFORCE and VPG on I2 *diverges* on several series/seeds (returns of −1e24 and worse), whereas TRPO's I2 never did (final 2.78/3.24/0.76); the saved best checkpoints (used for test) are all fine, so the instability is hidden by best-checkpoint selection. On I1 REINFORCE-none/VPG are slightly better than TRPO on two series (raw p<0.05, differences 0.05–0.07). After Holm over 42 rows the best raw p is 0.082, so nothing is significant. Test profits are within noise of capped TRPO, REINFORCE and FDPG (`outputs/trpo/qfree/ext_eval/EXT_EVAL.md`).

**Entropy bonus (all 11 problems, 10 seeds).** Chosen c: 0.003 on 7 problems, 0.01 on 2, 0.03 on 2. It did **not** help: final validation return was equal or slightly lower (raw p<0.05 for the bonus being worse on FOODS_2 I1 and FOODS_3 I2, which also survive Holm at 0.021; otherwise no significant differences), the KL constraint is unaffected, and test profits/returns are within noise of plain TRPO (TECHM T1 0.535 vs 0.540). Caveat: plain TRPO's grid did not use c; the bonus could only help if exploration were limiting, which standardised advantages and a learned σ already avoid here.

**Neural TRPO (R-I4 on 3 series, R-T3 on TECHM; 10 seeds; reduced tuning: 12 configs, 1 seed).** Trust region holds (all steps accepted, KL ≤ δ); the surrogate-vs-realised correlation is again ≈0 (−0.02…0). Final validation return: TRPO neural is **not better than VPG+GAE neural or neural REINFORCE (loo_time)**: FOODS_2 0.65 / 0.77 / −5.1, FOODS_3 2.64 / 2.89 / 2.89, HOBBIES 3.16 / 3.15 / 3.24, TECHM T3 0.45 / 0.76 / 0.83. It beats random search and SPSA on FOODS_2 and HOBBIES (Holm-significant on FOODS_2, vs random and SPSA, and HOBBIES vs random) and beats REINFORCE only on FOODS_2 (raw p 0.049, not Holm-significant), where REINFORCE collapses at the last checkpoint. It is significantly worse than REINFORCE loo_time on FOODS_3 (Holm 0.039) and worse than every method on TECHM T3 (raw p ≤ 0.01; Holm significant only vs REINFORCE `none`). Test period: inventory profits of neural TRPO (780/167/105) are within noise of linear TRPO (769–808/166–170/102–106), i.e. the MLP adds nothing on these small problems; on TECHM neural TRPO returns 0.135 vs 0.54 for linear T1, 0.69 for VPG neural and 0.99 for buy-and-hold, so none beats buy-and-hold. Cost: ~93 min for the full run (PyTorch Jacobians). CG-vs-exact cosine was not computed (P>130). Files: `outputs/trpo/neural_full/{comparison,algo_eval,ext_eval}`.

**Overall verdict on the extensions.** None of the three extensions improves on plain linear TRPO. Q_max-free shows the one clear stability benefit of TRPO (no last-checkpoint divergence without the cap); the entropy bonus is neutral to slightly harmful; neural policies are no better and, for TRPO, worse than VPG on TECHM.

