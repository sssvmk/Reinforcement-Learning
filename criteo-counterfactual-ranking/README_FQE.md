# FQE on the Criteo Counterfactual-Ranking Logs -- Addendum

This file documents the **Fitted Q Evaluation (FQE)** work added on top of
the existing project in this folder (`parser.py`, `features.py`,
`policy.py`, `estimators.py`, `train.py`, `README.md`, `tests/*`). **None
of those pre-existing files were modified**, per the task instructions.
New files only: `fqe.py`, `run_experiment.py`, `tests/test_fqe.py`,
`requirements-fqe.txt` (additive, lists `matplotlib`, the only new
dependency), and this document. See [`README.md`](./README.md) for
everything already covered there (base file-format description, the
parser/features/policy design, the CRM pipeline, and the "scaling to
100M+" discussion) -- this addendum does not repeat that material except
where Step 0 uncovered something new.

> Note: `README.md` predates this addendum and was not modified (per the
> no-overwrite instruction), so it does not link back here. This document
> is the one-directional index: start at `README.md` for the base
> project, come here for the FQE-specific work.

## 1. Step 0 findings (re-run for this task)

Re-running the Step-0 inspection against the real `sample.txt`
reconfirmed everything already in `README.md` Section 3 (412 impressions,
raw CTR 6.068%, 10x-corrected CTR 0.6418%, slot-count distribution,
propensity range, 149 banner types, 409/412 `wasAdClicked` /
`wasProductClicked` consistency). One thing was **not** previously
documented and is new in this addendum:

### New finding: multi-valued features are repeated-key tokens, not comma-separated

`README.md` states "no multi-valued (comma-separated) features appear
anywhere in the 412-impression sample," which is true *for that specific
syntax*. However, roughly **3,469 of ~9,574 candidate lines (36%)**
contain a feature id appearing **two or more times on the same line with
different values**, e.g.:

```
0 exid:19168103 17:2 17:2 21:55 21:55 22:130 22:28 24:3039 25:721 33:386 33:386 35:963
```

This is a genuine multi-valued-categorical phenomenon (feature 17 takes
values {2, 2}, feature 22 takes {130, 28}, feature 33 takes {386, 386},
etc. for this one candidate), just expressed as **repeated
`featId:value` tokens** rather than `featId:v1,v2`. The existing
`parser.py::_parse_feat_pairs` builds a plain dict (`feats[fid] = val`),
so a repeated key **silently overwrites** the earlier value; downstream,
`features.py::featurize_candidate` therefore only ever sees the
last-occurring value per feature id within a candidate line, slightly
under-representing real multi-valued signal.

**This is documented here, not fixed in-place**, per the instruction not
to overwrite existing files. `fqe.py` reuses the existing (lossy)
featurization as-is for consistency with the rest of the pipeline
(`estimators.py`, `policy.py`, `train.py` all already depend on it). A
non-destructive fix would be straightforward (reparse tokens into
`{fid: [v1, v2, ...]}` by accumulating rather than overwriting) but is
left as a documented known limitation rather than silently patched.

## 2. Why the slot-MDP formulation

The raw data is a one-shot contextual bandit (one impression -> one
banner -> one `wasAdClicked` label). For FQE to produce a non-trivial,
temporal-difference-style value estimate, we reformulate each impression
as an `H = nb_slots`-step MDP that **exactly mirrors how the logging
policy generated the banner** (sequential Plackett-Luce sampling without
replacement):

- **Episode** = one impression, **state at slot k** = (display features,
  slot index k, horizon H, a summary of the k already-placed candidates).
  **Action at slot k** = the candidate placed there. **Reward at slot
  k** = that candidate's `wasProductClicked` (default; see below).
  **Transition**: deterministic -- the chosen candidate is removed from
  the remaining pool and k increases by 1. **Terminal**: after slot
  `nb_slots`, Q = 0.
- A key structural fact, already implicit in `policy.py`'s PL formula:
  because displayed candidates are always the first `nb_slots` entries of
  the candidate list (in slot order), **the remaining pool at slot k is
  exactly `candidates[k:]`**. This means the action actually taken at
  slot k is always local index 0 of that remaining list, and so is the
  *next* logged action at slot k+1 -- which makes the on-policy/SARSA FQE
  variant trivial to implement correctly (bootstrap on local row 0, no
  extra bookkeeping) and rules out a whole class of off-by-one bugs.
- **Reward choice**: Step 0 found `wasAdClicked` and "any displayed
  product clicked" agree in 409/412 (99.3%) of impressions. We default to
  `REWARD_MODE="per_slot_click"` (`wasProductClicked` at each slot),
  because it gives the Q-function real, position-resolved learning
  signal at every slot; the alternative `REWARD_MODE="banner_click_last_slot"`
  (banner-level reward only at the terminal slot) is implemented and
  configurable in `fqe.py::_reward_for_slot`, but would make all non-terminal
  slots reward-free, losing most of the slot-level structure that
  motivates this whole formulation. With the default, the 3 "ad clicked
  but no product clicked" impressions contribute zero reward at every
  slot (their click signal is simply absent from the slot-MDP) -- a minor,
  documented loss (0.7% of impressions).
- This is **an explicit modeling choice**, not a claim about the true
  data-generating process: click labels for products at different slots
  may depend on position (list-position bias), which the state captures
  only through the hashed `slot=k` / `horizon=H` tokens, not through any
  richer position model.

### Q-function input features

`fqe.py::build_state_action_features` reuses `features.featurize_candidate`
verbatim for the candidate's own context+product features, then adds,
via `features.hash_token` into *separate* hashed buckets (distinct token
prefixes, so they cannot collide with the plain candidate features used
for Plackett-Luce *scoring*):
- `slot={k+1}`, `horizon={nb_slots}` (position/horizon indicators)
- `n_placed` (log1p-scaled count of already-placed candidates)
- `placed:{idx}` for every hashed index active in the mean of the
  already-placed candidates' own (product-only) feature vectors --
  i.e. a coarse "profile of what's already been shown" signal.

### The bandit-only baseline is a true special case, not a reimplementation

`fqe.py::fit_fqe(..., horizon_cap=1, gamma=0.0, n_iters=1)` disables the
bootstrap term entirely (forces every slot-0 transition terminal), which
makes the iterative FQE code degenerate to exactly one weighted ridge
regression of reward on `(s_0, a_0)` features -- the direct method. This
is checked against an **independently written** `bandit_direct_method()`
in `tests/test_fqe.py::test_bandit_special_case_matches_direct_method`
(sanity check 3): the two should (and do) match to numerical precision.

## 3. Target policies (`run_experiment.py`)

1. **Uniform** -- `PlackettLucePolicy.uniform` (all-zero theta).
2. **Popularity/CTR** -- Laplace-smoothed per-hashed-token click rate on
   TRAIN displayed candidates, converted directly to a PL log-odds score
   (`build_popularity_policy`). Deliberately *not* a jointly fit model
   (that's policy 3), to give a genuinely different, simpler inductive
   bias ("show whatever historically got clicked most, feature by
   feature, independently").
3. **Reward-model** -- logistic regression (`train.py::fit_reward_model`,
   reused as-is) predicting product click from features + a hashed
   slot-position indicator; its coefficients become the PL theta.
   Temperature `T in {0.5, 1, 2}` is grid-searched on the VALIDATION slice
   by clipped SNIPS (`select_reward_model_temperature`); **T=2.0** was
   selected on the real sample.
4. **Logging-policy proxy** -- the logged behavior itself. It has no
   parametric theta (only the realized banner's propensity is logged, not
   the scores that produced it), so it is special-cased throughout:
   importance weight `w_i = 1` for the IS estimators (which reduces
   IPS = SNIPS = DR = the 10x-corrected CTR exactly -- see Section 5), and
   the **on-policy/SARSA FQE variant** (bootstraps on the logged next
   action) for the FQE column.

## 4. Results (test slice, n=83 impressions, seed=0, `dim=65536`)

```
policy          FQE (slot-MDP)            Bandit DM                 IPS                       SNIPS                     DR                        ESS/n     slot1_cov
uniform         0.0459 [0.0319, 0.0634]   0.0098 [0.0015, 0.0187]   0.0002 [0.0000, 0.0005]   0.0002 [0.0000, 0.0017]   0.0136 [-0.0155, 0.0336]  0.027     0.120
popularity      1.2562 [-0.7473, 4.5539]  -0.0192 [-0.0275, -0.0107]0.0000 [0.0000, 0.0000]   0.0000 [0.0000, 0.0000]   0.0334 [0.0224, 0.0485]   0.047     0.190
reward_model    0.0421 [0.0294, 0.0568]   0.0103 [0.0020, 0.0190]   0.0002 [0.0000, 0.0005]   0.0003 [0.0000, 0.0015]   0.0147 [-0.0163, 0.0378]  0.039     0.127
logging_proxy   0.0063 [0.0014, 0.0137]   n/a                       0.0050 [0.0012, 0.0106]   0.0050 [0.0012, 0.0106]   0.0050 [0.0012, 0.0106]   0.961     1.000
```
(reproduce with `python run_experiment.py`; full output also saved to
`results.json` and the convergence plot to `fqe_convergence.png`.)

Convergence curves (Bellman error and V_hat per iteration, all 4
policies overlaid) are in **`fqe_convergence.png`**.

### Reading these results honestly

- **Logging-proxy consistency (sanity check 1)**: on-policy FQE (0.0063)
  and SNIPS-for-the-logging-policy (0.0050, which is *exactly* the
  10x-corrected test CTR by construction -- `w==1` makes SNIPS reduce
  algebraically to `sum(c*r)/sum(c)`) are close, with the FQE estimate
  within its own bootstrap CI of the corrected CTR. The ~0.0013 gap is
  well within what 83 impressions can resolve.
- **The popularity policy's FQE estimate diverges** (0.0129 -> ... ->
  1.256 across iterations; see `fqe_convergence.png`). This is not a bug:
  the popularity policy's `slot1_cov = 0.19` means 81% of its
  probability mass sits on candidates the logging policy rarely or never
  displayed in this impression. The linear ridge Q-model (dim=65536,
  alpha=1, ~83 impressions / a few thousand transitions) is badly
  underdetermined in that region of feature space, so each Bellman
  backup amplifies extrapolation error on those out-of-support
  candidates -- the textbook "poor target-policy coverage breaks FQE"
  failure mode the task asked us to surface, reproduced concretely rather
  than smoothed over. The bandit direct method (a single regression, no
  iterative amplification) does **not** diverge for the same policy
  (-0.019), which is itself informative: the instability is specifically
  a multi-step bootstrapping artifact, not a property of the Q-features
  alone.
- **DR is the most stable estimator across policies** here (small,
  plausible CIs for every policy, including popularity), because it
  falls back on the raw observed reward whenever the reward model and IS
  weight roughly agree, and does not iterate a Q-function at all.
- **IPS/SNIPS are both pinned near 0 for `uniform`/`popularity`/`reward_model`**
  (ESS/n of 0.027-0.052): with only 83 test impressions and importance
  weights that can be extreme, almost all effective sample size collapses
  onto a handful of impressions. These numbers should be read as "we
  cannot resolve policy value with this estimator at this sample size,"
  not as "these policies have near-zero value."
- Overall: **do not rank policies from this table**. Every
  interval overlaps every other interval for at least one estimator, the
  sample is tiny, and the popularity policy's FQE instability shows the
  estimator itself is not trustworthy for ill-covered policies here. This
  is a pipeline correctness demonstration, not a policy recommendation
  (see Section 6).

## 5. Estimators reused as-is from `estimators.py` / `train.py`

IPS, SNIPS, clipped SNIPS (`clip_m=20.0`), doubly robust, effective
sample size, and bootstrap CIs are the existing, unmodified
implementations in `estimators.py`. The DR baseline `E_{a~pi_e}[r_hat]`
for policies 1-3 reuses `train.py::dr_baseline_mc` (Monte-Carlo banner
sampling + noisy-OR aggregation of the reward model's per-slot
predictions). For the **logging-policy proxy**, there is no theta to
sample banners from, so its DR baseline uses `r_hat_logged_banner` on the
*actually logged* banner; algebraically this makes
`DR_i = r_hat_logged_i + 1*(r_i - r_hat_logged_i) = r_i` exactly, i.e. DR
for the logging policy degenerates to the plain (10x-corrected) reward
average, which is the mathematically correct behavior when evaluating a
policy against itself.

### FQE confidence intervals: a documented approximation

`fqe_value_ci()` bootstraps the **final policy-value averaging step**
over impressions, holding the already-fitted Q-function fixed, rather
than refitting the entire `n_iters`-iteration FQE loop on every bootstrap
resample (2,000 resamples x 20 refits would be prohibitively slow at
scale and is unnecessary for a pipeline check at n=83). This
**understates** the true sampling uncertainty of FQE (it ignores
Q-model-fitting variance entirely) and should be read as an optimistic /
lower-bound CI. A fully rigorous CI would refit Q per resample; this is
left as a documented limitation / future-work item.

## 6. Limitations (read before trusting any number above)

- **~400 impressions total (83 in the test slice).** Every confidence
  interval above is wide; treat these results as a pipeline correctness
  check, not evidence about which policy is actually best. This is
  explicitly called out per the task's own caveat.
- **The slot-MDP is a modeling choice**, not a literal description of how
  clicks are generated. Position (list-position bias) is represented only
  through a hashed `slot=k` indicator; any slot-interaction effects beyond
  that are not modeled.
- **FQE is biased/unstable under poor target-policy coverage.** The
  `popularity` policy's divergence in Section 4 is a concrete,
  reproduced instance of this, not a hypothetical caveat. The
  `slot1_coverage` diagnostic (share of a policy's first-slot probability
  mass that falls on candidates *actually displayed somewhere in that
  same impression*) is reported for every policy precisely to flag this
  risk ahead of time.
  - **Coverage-diagnostic caveat**: the Criteo schema has no stable
    product identifier across impressions, so a literal "share of mass on
    products the logging policy ever placed in that slot" (pooled across
    the whole log, as worded in the task) is not computable. `slot1_coverage`
    is an explicitly documented **per-impression proxy** for that idea
    (displayed-in-this-impression vs. not), not the literal global
    diagnostic.
- **The 10x non-click subsampling correction (`c_i`) is applied
  everywhere** reward or Q-regression targets are computed or averaged
  (slot rewards, popularity-policy click rates, FQE's weighted ridge
  fits, the final `V_hat` averages, and every IS estimator) -- consistent
  with the existing `train.py`/`estimators.py` convention.
- **Inherited parsing limitation** (Section 1): repeated-key multi-valued
  features are silently collapsed to their last value by the existing
  (unmodified) `parser.py`/`features.py`. This affects ~36% of candidate
  lines and is a genuine, if modest, information loss baked into every
  featurization this addendum builds on.
- **Cost/benefit of the state representation**: the "placed-so-far"
  summary is a mean of hashed product-only feature vectors, not an
  explicit set -- it cannot distinguish "placed candidates A and B" from
  "placed candidates with the same average feature profile but different
  identities." This is a deliberate simplification to keep the
  Q-function's input dimensionality bounded (see spec: "a summary of the
  products already placed ... count, mean of their feature vectors").
- **No test-slice information was used for training or tuning anywhere**
  in this addendum: the reward model, the popularity policy, and the
  temperature grid search all use TRAIN/VALIDATION only; `test_imp` is
  touched only inside `evaluate_policy_full` / the final results table.
- Pre-existing files (`parser.py`, `features.py`, `policy.py`,
  `estimators.py`, `train.py`, `README.md`, `conftest.py`,
  `requirements.txt`, `tests/test_parser.py`, `tests/test_features.py`
  if present, `tests/test_policy.py`, `tests/test_estimators.py`,
  `tests/test_train.py`) were **not modified** by this work.

## 7. How to run

```powershell
pip install -r requirements.txt -r requirements-fqe.txt
python run_experiment.py --data "C:\Users\msk80743\Downloads\sample.txt"
python -m pytest tests/ -v   # includes the pre-existing suite + tests/test_fqe.py
```

Fixed random seed: `0` (printed at the start of every `run_experiment.py`
run, matching the existing `train.py` convention).

## 8. New files added by this task

| File | Purpose |
|---|---|
| `fqe.py` | Slot-MDP dataset construction, `QModel`, `fit_fqe` (off-policy + on-policy/SARSA), `bandit_direct_method`, `estimate_policy_value`, `slot1_coverage`. |
| `run_experiment.py` | Builds the 4 target policies, runs FQE + bandit DM + IPS/SNIPS/DR for each on the test slice, prints the results table, saves `fqe_convergence.png` and `results.json`. |
| `tests/test_fqe.py` | Sanity checks 1, 3, 4, 5, 8 (checks 2/6/7 already covered by the existing `tests/test_policy.py` / `tests/test_train.py`). |
| `requirements-fqe.txt` | Additive: lists `matplotlib`, the only new dependency (needed for convergence plots). |
| `README_FQE.md` | This document. |

## 9. What FQE is, and where it is used (general background)

Fitted Q Evaluation (FQE) is an off-policy evaluation method for
reinforcement learning. Given a fixed logged dataset of transitions
`(s, a, r, s')`, it estimates how good a target policy `pi` would be,
without running it. It repeatedly regresses Q-values toward the TD target
`r + gamma * Q(s', pi(s'))`, where the next action comes from the policy
being evaluated, not from a max as in Q-learning. This is exactly the
`fit_fqe` loop in `fqe.py`: at each iteration it computes
`y = r + gamma * E_{a' ~ pi_e(.|s')}[Q_prev(s', a')]` (an *expectation*
over `pi_e`'s next action, computed exactly over the remaining
candidates, rather than a single sampled or greedy-max action) and
refits `Q` by weighted regression onto `y`.

### Main applications

1. **Evaluating policies before deployment.** Checking whether a newly
   trained policy beats the current one before it touches real users,
   patients, or inventory. Safety gating: only ship a policy if its
   estimated value exceeds the logging policy by a margin.
2. **Model selection in offline RL.** Offline RL has no environment to
   test against, so FQE scores candidate policies (different CQL, BCQ,
   IQL or DQN checkpoints and hyperparameters) and picks the best. It is
   widely used as the validation step in offline RL benchmarks.
3. **Healthcare and clinical decision support.** Evaluating treatment
   policies (sepsis management, ventilation, dosing) from historical
   patient records, where online testing is unethical.
4. **Recommendation and advertising.** Evaluating sequential recommenders
   and ad-allocation policies that optimize long-term engagement, where
   simple IPS has exploding variance over long horizons -- this is
   exactly the ranking setting in this project (the slot-MDP formulation
   in Section 2 is a direct instance of this application).
5. **Supply chain and operations.** Estimating the value of a new
   ordering policy from logged inventory decisions, before changing real
   orders (see the companion DQN inventory-control project in this
   workspace).
6. **Robotics and autonomous systems.** Ranking policies learned from
   logged driving or manipulation data before risky real-world tests.
7. **Finance and pricing.** Evaluating dynamic pricing or execution
   policies from historical logs.
8. **Research and benchmarking.** A standard baseline in OPE benchmarks
   such as DOPE, and a building block for doubly robust and other hybrid
   estimators (as used directly alongside FQE in Section 5 here).

### Why use FQE instead of importance sampling

- **Long horizons**: trajectory-level IS multiplies weights across steps,
  so variance explodes. FQE avoids this by bootstrapping instead of
  reweighting whole trajectories.
- **Lower variance in practice**, and it works with function
  approximation (our Q-model is a ridge regression over hashed features;
  it could equally be an MLP).
- **No need for logged behavior propensities**, because it is a
  model-based-in-value-space method, not a ratio method -- unlike
  IPS/SNIPS/DR (Section 5), FQE's own estimate does not require
  `propensity` at all (though this project's slot-MDP *actions* are still
  only well-defined because the logging process is known to be
  sequential-without-replacement).

### Limitations (general, beyond the project-specific ones in Section 6)

- **Biased if the function class is wrong or the data doesn't cover the
  target policy's actions** (distribution shift) -- concretely
  demonstrated by the `popularity` policy's divergence in Section 4.
- **Can diverge or overestimate with poor features**, like other
  bootstrapped methods (the same mechanism as Q-learning/DQN divergence
  without a target network, discussed in the companion DQN
  inventory-control project).
- **Sensitive to hyperparameters** (ridge `alpha`, `n_iters`, hashed
  `dim`), so run several seeds and sanity-check against simple baselines
  (here: the bandit direct method and the logging-policy proxy).
- **No built-in confidence intervals**, so bootstrap over trajectories
  (`fqe_value_ci` in `run_experiment.py`) -- with the caveat, documented
  in Section 5, that our implementation bootstraps the final averaging
  step only, not the full iterative refit.
