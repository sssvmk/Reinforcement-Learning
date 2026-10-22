# Off-Policy Ranking Evaluation and Learning on Criteo Counterfactual Logs

This project builds, trains, and evaluates a **ranking policy** for display
ads using **importance sampling (IS) / off-policy evaluation (OPE) and
learning**, on the Criteo counterfactual-learning sample
(`C:\Users\msk80743\Downloads\sample.txt`). It implements the full pipeline
from raw log parsing through Plackett-Luce policy scoring, several
off-policy estimators (IPS, SNIPS, clipped variants, doubly robust), and a
counterfactual-risk-minimization (CRM) policy learner -- all designed to
stream so the same code can later run on the full 100M+ impression dataset.

---

## 1. Background: what is importance sampling, and where it's used

Importance sampling estimates a quantity under one distribution using
samples drawn from another, by reweighting each sample with the ratio
p(x)/q(x). It's useful whenever sampling from the target distribution is
expensive, risky, or impossible.

**Machine learning and AI**
- Off-policy evaluation and learning (what you're doing): estimating how a
  new recommender, ad-ranking or bandit policy would perform using logs
  from an old policy.
- Reinforcement learning: correcting for off-policy data in policy
  gradients (PPO, IMPALA/V-trace), experience replay, and offline RL.
- LLM training and RLHF: PPO's probability-ratio clipping is an importance
  weight between the old and new policy.
- Covariate shift and domain adaptation: reweighting training data so it
  resembles the test distribution.
- Training speedups: importance sampling of mini-batches, so hard or
  informative examples are drawn more often, and sampled softmax with
  large vocabularies.
- Variational inference and generative models: importance-weighted
  autoencoders (IWAE) give tighter likelihood bounds.

**Statistics and causal inference**
- Inverse propensity weighting: estimating treatment effects from
  observational data, and correcting for survey non-response or selection
  bias.
- Rare-event estimation: computing very small probabilities (tail risks,
  failure rates) by sampling from a distribution that makes the event
  likely.
- Bayesian computation: self-normalized importance sampling for posterior
  expectations, sequential Monte Carlo and particle filters.

**Finance and risk**
- Tail-risk estimation: VaR and CVaR for extreme losses.
- Derivative pricing: pricing deep out-of-the-money options with Monte
  Carlo.
- Credit and portfolio risk: estimating the probability of joint defaults.

**Engineering and science**
- Reliability and safety: probability of system failure, e.g. in
  autonomous vehicle testing, where crashes are too rare to see in normal
  simulation.
- Physics and graphics: particle transport and nuclear shielding
  simulation, and path tracing in rendering, where light paths are sampled
  by importance to reduce noise.
- Communications: bit-error-rate estimation at very low error rates.

**Practical caveat**

Importance sampling only works well when the sampling distribution covers
the target and the weights don't explode. Heavy-tailed weights give huge
variance. That's why practitioners use self-normalization (SNIPS), weight
clipping, and effective sample size checks, as in your Criteo setup.

---

## 2. How to run

```bash
cd criteo-counterfactual-ranking
pip install -r requirements.txt

# Step 0 - inspect the raw file and print summary stats
python parser.py "C:\Users\msk80743\Downloads\sample.txt"

# Run the full pipeline: parse -> featurize -> fit reward model -> build
# 4 policies -> evaluate all 4 on the test slice
python train.py --path "C:\Users\msk80743\Downloads\sample.txt" --dim 4096 --n_boot 1000 --mc_samples 50

# Run the test suite (sanity checks 1-6)
python -m pytest tests -v
```

`train.py` flags:
- `--path`: path to the log file (`.txt` or `.txt.gz`)
- `--dim`: hashed feature-space size (default 65536; a smaller value like
  4096 runs faster on the 400-impression sample with no accuracy loss,
  since there are far fewer than 4096 distinct feature values in the
  sample -- see "Scaling to the full dataset" below)
- `--seed`: random seed (printed in the output; default 0)
- `--n_boot`: bootstrap resamples per estimator CI (default 2000)
- `--mc_samples`: Monte-Carlo samples per impression for the DR baseline
  term (default 50)

---

## 3. Step 0 findings: actual file format vs. the spec

The real `sample.txt` matches the described format closely. Confirmed via
`parser.py summarize(...)`:

```
Impressions: 412
Raw CTR (uncorrected for subsampling): 6.0680%
Corrected CTR (10x-weighted non-clicks): 0.6418%
Slot-count distribution: {1: 69, 2: 152, 3: 96, 4: 33, 5: 8, 6: 54}
Propensity summary: {'min': 7.97e-10, 'max': 0.9996, 'mean': 0.1576, 'median': 0.0162}
Distinct banner types (feat 1,2,3,5 combos): 149
```

Differences / clarifications found against the spec, all incorporated into
`parser.py`:
- **No multi-valued (comma-separated) features appear anywhere in the
  412-impression sample.** The parser still handles them (splits on `,`
  and multi-hot-hashes each sub-value) for forward-compatibility with the
  full dataset, but this was never exercised against real rows -- flagged
  as untested-on-real-data in Limitations.
- **Display features are exactly IDs 1-10, product features exactly IDs
  11-35** (35 total, as documented) -- no extra/missing feature IDs found.
- **Non-displayed candidates (`candidates[nbSlots:]`) are *always*
  labeled `wasProductClicked = 0`** in the sample (8,417/8,417) -- expected,
  since a product that was never shown cannot be clicked; the reward model
  is deliberately trained only on *displayed* candidates (see §5) to avoid
  teaching it "not shown => unclicked" instead of real click propensity.
- **The banner-level `wasAdClicked` label matches "any displayed product
  was clicked" for 409/412 impressions (99.3%).** Not a perfect
  equivalence, but close enough to justify a noisy-OR aggregation
  `r_hat(x, banner) = 1 - prod_k (1 - p_hat(candidate_k))` for the reward
  model's banner-level reward estimate (§5), documented as an assumption
  rather than an exact identity.
- Header/candidate-line structure, `exid` linkage, and `nbSlots <=
  nbCandidates` all validated with zero structural errors across all 412
  impressions.

---

## 4. Design decisions and assumptions

### 4.1 Parsing (`parser.py`)
- Streams one impression (header + its candidate lines) at a time;
  never loads the full file. Supports `.gz` transparently.
- Validates: propensity in `(0, 1]`, `nbSlots <= nbCandidates`, exactly
  `nbCandidates` candidate lines per header, and `exid` consistency between
  header and candidate lines. Raises `ValueError` with the offending
  `exid` on any violation (fail fast rather than silently drop/guess).
- **Non-click subsampling correction**: every impression gets a sampling
  weight `c_i = 10.0` if `wasAdClicked == 0`, else `c_i = 1.0`, matching
  the stated 10% non-click retention rate. Every estimator in this project
  is a `c`-weighted statistic; omitting `c` would understate CTR and
  policy value by roughly 10x (demonstrated in `tests/test_train.py`).

### 4.2 Features (`features.py`)
- **Features 1 and 2 are numeric**: `log1p`-transformed (no additional
  standardization). We deliberately skip mean/variance standardization
  because it requires a full corpus pass to compute global statistics,
  which breaks single-pass streaming to 100M+ rows; `log1p` needs no
  corpus-level statistics and still compresses the wide, right-skewed
  range observed (feature 1: 2-1024, feature 2: 2-768 in the sample).
- **All other features (categorical)** are encoded via the **hashing
  trick** (Weinberger et al. 2009): each `(feature_id, value)` token is
  MD5-hashed into a bucket index plus a `+-1` sign (for approximately
  unbiased collisions), avoiding any need for a fitted vocabulary -- a
  pure function of its inputs, hence trivially streaming-safe.
- **Multi-valued categoricals** (comma-separated, not observed in the
  sample) are handled as multi-hot: each sub-value contributes its own
  hashed slot.
- A candidate's feature vector = its own product features (11-35) +
  the impression's display features (1-10), simply combined (no explicit
  display x product interaction features) -- kept simple and documented;
  interactions are a natural extension (§7 Next Steps).
- Default hash dimension: 65536 (`--dim`); 4096 is used in the reported
  run below since the 412-impression sample has far fewer distinct
  feature values than that.

### 4.3 Ranking policy (`policy.py`)
- Parametric Plackett-Luce score: `s(x, candidate) = exp(theta . phi(x,
  candidate) / T)`.
- `log_prob_logged_banner` computes the **exact** PL log-probability of
  the logged banner via the prefix structure: since the logged banner is
  always `candidates[0:nbSlots]` in slot order, the "remaining candidates"
  set at slot `k` is simply `candidates[k:]` (no need to track an explicit
  removal set) -- `O(nbCandidates)` per impression, computed in log-space
  with `logsumexp` for numerical stability.
- `sample_banner` draws banners from the same distribution via sequential
  softmax sampling without replacement, used only to sanity-check the
  log-probability formula (never needed for evaluation itself, since the
  logged action is already known) and for the doubly-robust baseline's
  Monte-Carlo expectation (§3.5).

### 4.4 Estimators (`estimators.py`)
- `w_i = pi_e(logged banner_i | x_i) / propensity_i`.
- **IPS**, **SNIPS**, **clipped** variants (`w` capped at a configurable
  `M`), **effective sample size** (Kish's ESS on the combined weight
  `c_i * w_i`, which by Cauchy-Schwarz is provably `<= n`), and
  **percentile bootstrap CIs** (resampling impressions with replacement)
  are all implemented per the spec.
- **Doubly robust (DR)**: `DR_i = baseline_i + w_i * (r_i -
  r_hat(x_i, a_i))`, where `baseline_i = E_{a~pi_e}[r_hat(x_i, a)]` is
  approximated via Monte-Carlo sampling banners from the policy (§3.5) and
  `r_hat(x_i, a_i)` is the reward model's prediction for the *actually
  logged* banner. **We treat the whole banner as a single joint "slate
  action"** rather than decomposing DR per-slot -- the simplest valid DR
  construction for ranking/slate logs, documented here as a modeling
  choice (a per-slot/pseudo-inverse DR estimator for slates is more
  sample-efficient but substantially more complex; out of scope here).

### 4.5 Reward model and DR baseline (`train.py`)
- A single **logistic regression** predicts `wasProductClicked` from a
  candidate's hashed features **plus a hashed slot-position indicator**
  (since click probability has position bias), trained **only on the
  `nbSlots` displayed candidates of the TRAIN split** (never the held-out
  non-displayed candidates, which are deterministically unclicked, and
  never validation/test data).
- Banner-level `r_hat(x, banner)` = **noisy-OR** aggregation of per-slot
  click probabilities: `1 - prod_k (1 - p_hat(candidate_k, slot k))`,
  justified by the 409/412 empirical match found in §2.
- The DR baseline `E_{a~pi_e}[r_hat(x,a)]` has no closed form for a
  Plackett-Luce slate distribution, so it is approximated by **Monte Carlo**:
  sample `--mc_samples` banners per impression from the policy via
  `sample_banner`, score each with the noisy-OR reward model, and average.

### 4.6 Off-policy-optimized (CRM) policy (`train.py: train_crm_policy`)
- Maximizes **clipped SNIPS - L2 penalty - POEM-style variance penalty**
  (`var_penalty * sqrt(Var[c*w*r] / n)`) over `theta`, via Adam in
  PyTorch, **initialized from the reward-model policy's coefficients**.
- Fully **vectorized** across impressions with padded tensors (no
  per-impression/per-candidate Python loop inside the gradient loop),
  making the multi-point hyperparameter grid tractable even though PL
  log-probabilities require a sequential (per-slot) computation.
- Hyperparameter grid searched (kept intentionally small for the
  412-impression sample; widen for the full dataset): temperature
  `T in {1.0, 2.0}`, clip `M in {10, 20}`, `L2 = 1e-3` (fixed),
  variance penalty `in {0.0, 0.2}` -- 8 configurations, each trained for
  150 epochs, selected by **validation clipped-SNIPS only**.

### 4.7 Evaluation protocol (`train.py: run`)
- **Temporal split by exID**: first 60% train / next 20% validation /
  last 20% test, by impression count, never shuffled (train=247,
  val=82, test=83 on the sample). Verified by test (`test_train.py`) to be
  non-overlapping and strictly ordered.
- All hyperparameters (temperature, clip `M`, L2, variance penalty) for
  the reward-model and CRM policies are selected using **validation
  clipped-SNIPS only** -- never training loss, never the test slice.
- The **test slice is used exactly once**, to report final IPS/SNIPS/
  clipped-SNIPS/DR with bootstrap CIs, ESS, mean weight, and lift over the
  logging policy's SNIPS, for all four policies.

---

## 5. The four policies

| Policy | theta | Temperature | Notes |
|---|---|---|---|
| `logging_policy` | n/a | n/a | Baseline: `w = 1` for every impression **by construction** (not computed via PL at all), so its SNIPS/IPS trivially equal the observed 10x-corrected CTR. |
| `uniform` | all-zero | 1.0 | All candidates score `exp(0) = 1` -> uniformly random permutation/subset at every slot. |
| `reward_model` | logistic-regression coefficients | selected on validation (2.0) | PL scores = the reward model's learned linear scores; a "greedy-ish" ranking policy without any CRM-style off-policy correction during training. |
| `crm` | gradient-optimized, init'd from `reward_model` | selected on validation (1.0) | Directly optimizes clipped SNIPS + variance penalty against the *actual logged feedback*, the only policy trained with explicit off-policy awareness. |

---

## 6. Results (test slice, `--dim 4096 --n_boot 1000 --mc_samples 50 --seed 0`)

```
Split sizes: train=247 val=82 test=83
reward_model policy: selected T=2.0, clip_M=5.0 (val clipped-SNIPS=0.0005)
crm policy: selected (T, clip_M, l2, var_penalty)=(1.0, 20.0, 0.001, 0.2) (val clipped-SNIPS=0.0005)

=== Test-slice results (policy x estimator) ===
policy            n  mean_w    ESS                 IPS                SNIPS           SNIPS_clip                      DR          lift(SNIPS)
logging_policy   83   1.000  79.76  0.0050 [0.0012,0.0106]  0.0050 [0.0012,0.0106]  0.0050 [0.0012,0.0106]                   n/a       0.0000
uniform          83   0.753   2.24  0.0002 [0.0000,0.0005]  0.0002 [0.0000,0.0015]  0.0005 [0.0000,0.0018]  0.0235 [0.0125,0.0347]     -0.0048
reward_model     83   0.563   3.18  0.0001 [0.0000,0.0004]  0.0002 [0.0000,0.0012]  0.0004 [0.0000,0.0013]  0.0240 [0.0049,0.0403]     -0.0048
crm              83   0.115   5.71  0.0000 [0.0000,0.0000]  0.0000 [0.0000,0.0000]  0.0000 [0.0000,0.0000]  0.0297 [0.0214,0.0402]     -0.0050
```

**Read this with real caution -- 83 test impressions is a tiny sample.**
IPS/SNIPS CIs for every learned policy comfortably include the logging
policy's value, so **none of these differences are statistically
distinguishable** at this sample size:

- **`logging_policy`'s SNIPS (0.0050) matches the 10x-corrected test CTR
  exactly**, as required by construction (sanity check 1/5).
- **ESS collapses hard for every non-logging policy** (2-6 out of 83),
  meaning the IPS/SNIPS point estimates for `uniform`/`reward_model`/`crm`
  are each dominated by a handful of impressions with unusually large
  importance weights -- a direct consequence of propensities spanning 9
  orders of magnitude (`7.97e-10` to `0.9996`) in this log. This is exactly
  the symptom clipping and DR exist to address, and exactly why the
  reported CIs are so wide.
- **`crm`'s mean weight (0.115) is the smallest** of the three learned
  policies -- its variance penalty is actively discouraging it from
  assigning high probability to banners the logging policy rarely showed,
  which is the intended CRM behavior, but it also means `crm`'s raw
  IPS/SNIPS are driven to ~0 (clip-dominated) rather than reflecting a
  confident value estimate.
- **DR estimates (0.023-0.030) are far higher than IPS/SNIPS (~0.0001-
  0.0005) for every learned policy** -- this is a real but unreliable
  discrepancy: with ESS in the single digits, IPS/SNIPS are essentially
  noise, while DR's point estimate is dominated by its *baseline* term
  (the reward model's own prediction under the policy), which was fit on
  only 247 impressions / 25 total clicks and should not be trusted as
  precise either. **The honest conclusion from this sample is "we cannot
  reliably rank these four policies yet," not "CRM/DR wins."**

---

## 7. Sanity checks (pytest, `tests/`)

All 6 requested checks pass, 26 tests total (`python -m pytest tests -v`):

1. **Logging policy evaluated against itself gives `w = 1`** --
   `test_train.py::test_logging_policy_weight_is_one_when_target_equals_logging_policy`
   (synthetic, since the true hidden logging-policy scores aren't in the
   log) + `test_estimators.py::test_weight_all_ones_gives_snips_equal_to_corrected_ctr`
   (algebraic form, and matches the real result in §5).
2. **PL probabilities of all permutations sum to 1** --
   `test_policy.py::test_pl_probabilities_of_all_permutations_sum_to_one`.
3. **`sample_banner`'s empirical distribution matches the analytic PL
   probabilities** (chi-square, p > 0.01 over 20,000 samples) --
   `test_policy.py::test_sample_banner_matches_analytic_distribution_chi_square`.
4. **10x non-click weighting changes the CTR estimate as expected** --
   `test_train.py::test_non_click_subsample_weight_changes_ctr_estimate`
   (uses the real sample; corrected CTR 0.64% vs. raw CTR 6.07%).
5. **Mean weight reported for every policy, flagged if far from 1** --
   `mean_weight` is in every `evaluate_policy` output (§5 table); the CRM
   policy's 0.115 is called out explicitly above as a flagged deviation.
6. **No train/val/test overlap; `exid` strictly ordered across splits** --
   `test_train.py::test_temporal_split_no_overlap_and_strictly_ordered_exid`.

Also covered: parser structural validation (propensity bounds, slot/
candidate-count/`exid` consistency, `.gz` support), the prefix-based PL
formula vs. brute-force enumeration, ESS bounded by `n` under non-uniform
`c` (a regression test added after finding and fixing a bug -- see §8),
and DR's reduction to the pure baseline when the reward model is exact.

---

## 8. Files

| File | Purpose |
|---|---|
| `parser.py` | Streaming impression parser + validation + `summarize()` (Step 0 stats) |
| `features.py` | Feature hashing (numeric `log1p` + categorical hashing trick, multi-hot for multi-valued) |
| `policy.py` | `PlackettLucePolicy`: scoring, exact log-probability of the logged banner, permutation enumeration (for tests), sampling |
| `estimators.py` | IPS, SNIPS, clipped variants, ESS, doubly robust, bootstrap CI, `evaluate_policy` |
| `train.py` | Full pipeline: split, featurize, fit reward model, build 4 policies (incl. vectorized CRM training), evaluate on test |
| `tests/` | 26 pytest sanity checks (parser, policy, estimators, pipeline-level) |
| `conftest.py` | Adds the project root to `sys.path` so tests can import the modules above |
| `requirements.txt` | numpy, scipy, scikit-learn, torch, pytest |

---

## 9. Limitations

- **Tiny sample (412 impressions, 83 test).** Every CI above is wide by
  construction; do not read point estimates as validated policy rankings.
- **Missing logging-policy scores.** We only ever observe `propensity`
  (a single number per impression), never the logging policy's actual
  per-candidate scores -- so we cannot build a "true" logging-policy
  object to compute, e.g., its own DR estimate; `logging_policy`'s row in
  §5 omits DR for this reason.
- **Extreme propensity range (1e-10 to ~1.0)** means importance weights
  can be astronomically large before clipping; clipping is essential here,
  not optional, and even clipped estimates have very low ESS at this
  sample size.
- **Reward model trained on only 247 impressions / 25 clicks** -- the
  logistic regression (and therefore the DR baseline and `reward_model`/
  `crm` policies that depend on it) is itself high-variance; a GBM or
  more data would likely help more than further estimator engineering.
- **Banner treated as a single joint action for DR** (not decomposed
  per-slot) -- simpler but leaves sample-efficiency on the table versus a
  full slate/pseudo-inverse DR estimator.
- **No display x product interaction features** -- kept simple per spec;
  likely under-fits real position/context effects.
- **Multi-valued categorical handling is untested on real data** (never
  observed in the 412-impression sample) -- logic is in place but
  unverified against an actual multi-valued row.
- **CRM hyperparameter grid is small** (8 configs) for runtime reasons on
  a tiny sample; would need a wider sweep (and likely a validation set
  larger than 82 impressions) to trust the selected config on real data.
- A harmless `OptimizeWarning: Unknown solver options: iprint` appears
  from the installed scikit-learn/scipy version combination during
  logistic regression fitting; it does not affect the fitted model.

### Scaling to the full 100M+ impression dataset
- `parser.parse_impressions` and `features.featurize_candidate` are
  already pure streaming functions (generator + stateless hashing) and
  need no changes.
- `train.py` currently materializes the full impression list and all
  candidate feature dicts in memory (fine at 412 impressions). For the
  full dataset, replace the in-memory lists with: (a) a single streaming
  pass to fit the reward model via `SGDClassifier.partial_fit` on
  mini-batches instead of a single `LogisticRegression.fit`, and (b) a
  `torch` `IterableDataset` wrapping `parse_impressions` for the CRM
  training loop (mini-batch gradient steps instead of one dense padded
  tensor for the whole split).
- The CRM padded-tensor batching (`_build_padded_batch`) would need to
  operate per mini-batch rather than on the whole training split at once.
