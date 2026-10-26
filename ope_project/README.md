# Off-policy evaluation with GTD2, TDC and ETD (research project)

Research code only; nothing here is trading or inventory advice. CPU-only, seeded (seed 20240607).

## Status
| Part | Status |
|---|---|
| Trading domain (TECHM, one instrument) | **Run** (`--full`: 19 conditions, 115 repetitions, 36.8 min) |
| Inventory domain (M5) | **DONE (extension to the locked plan)** – M5 path supplied after the trading run, so `analysis_plan.yaml` (hash-locked, `domains: [trading]`) was not changed; inventory is reported as a recorded deviation/extension. Run: `python run_experiment.py --full --yes --domain inventory` (192.3 min for the final run, 10 reps primary + 2 walk-forward folds + 2 replication series + 12 ablations incl. lead time 1/5), then `python make_report.py full_inv`. Results: `outputs/results_full_inv.md`, `outputs/figures/*_inv.png`. Caveats: all inventory costs, the 8 order levels, the base-stock behavior and the 24-agent pool are my assumptions; plain DQN agents still collapse to (near) never_order even after the fix (600 train windows, up to 20k steps), while the two CQL agents learn useful policies (test profit about 4,000-4,160, on par with the best base-stock rules); GTD2 diverged in 17% of primary-fold reps; ranking quality on the primary series is weak (Spearman: TD 0.37, TDC 0.30, ETD 0.33, GTD2 -0.05, OracleProj 0.99) while replication series rank better (about 0.82-0.86 for TD/GTD2/TDC/ETD on intermittent). The first run (collapsed DQN pool) is archived in `outputs/archive_inv_run1_dqn_collapsed/`. Inventory-specific split override in `config.yaml` (`inventory.splits`, `inventory.walk_forward`). |
| Tests | 43 pass (`python -m pytest tests -q`): estimators (tests 6-12, 15), trading (2-5, 13, 14), inventory env and domain (1, `vec_step` equals the scalar env for L=0,1,2,4, conservation, splits, determinism) |

**Inventory config:** `config.yaml: paths.m5_dir` holds the real folder (no braces); `check_m5` reports READY. Nothing inventory-related remains unimplemented except the items under Caveats.

## Results index (click to open)
| What | Trading | Inventory (M5) |
|---|---|---|
| Result tables, ablations, paired tests | [results_full.md](outputs/results_full.md) | [results_full_inv.md](outputs/results_full_inv.md) |
| Business metrics per agent (profit, log return, continuous rate, Sharpe, drawdown, benchmarks) | [business_metrics_trading.md](outputs/business_metrics_trading.md), [csv](outputs/business_metrics_trading.csv) [business_metrics_inventory.md](outputs/business_metrics_inventory.md), [csv](outputs/business_metrics_inventory.csv) |
| Estimated vs true J | [j_scatter.png](outputs/figures/j_scatter.png) | [j_scatter_inv.png](outputs/figures/j_scatter_inv.png) |
| MSVE / MSPBE learning curves | [learning_curves.png](outputs/figures/learning_curves.png) | [learning_curves_inv.png](outputs/figures/learning_curves_inv.png) |
| Step-size sensitivity | [stepsize_sensitivity.png](outputs/figures/stepsize_sensitivity.png) | [stepsize_sensitivity_inv.png](outputs/figures/stepsize_sensitivity_inv.png) |
| Ranking vs coverage | [ranking_vs_coverage.png](outputs/figures/ranking_vs_coverage.png) | [ranking_vs_coverage_inv.png](outputs/figures/ranking_vs_coverage_inv.png) |
| Ranking vs log size | [ranking_vs_logsize.png](outputs/figures/ranking_vs_logsize.png) | [ranking_vs_logsize_inv.png](outputs/figures/ranking_vs_logsize_inv.png) |
| Performance of each method's picks | [picks_performance.png](outputs/figures/picks_performance.png) | [picks_performance_inv.png](outputs/figures/picks_performance_inv.png) |
| ETD emphasis variance | [etd_emphasis_variance.png](outputs/figures/etd_emphasis_variance.png) | [etd_emphasis_variance_inv.png](outputs/figures/etd_emphasis_variance_inv.png) |
| Raw per-rep metrics (CSV) | [full_rep_metrics.csv](outputs/results/full_rep_metrics.csv) | [full_inv_rep_metrics.csv](outputs/results/full_inv_rep_metrics.csv) |
| Per-agent estimates (CSV) | [full_agent_rows.csv](outputs/results/full_agent_rows.csv) | [full_inv_agent_rows.csv](outputs/results/full_inv_agent_rows.csv) |
| Paired statistics, tuning, ETD diagnostics | [stats](outputs/results/full_stats.json), [tuning](outputs/results/full_tuning.json), [diagnostics](outputs/results/full_diagnostics.csv) | [stats](outputs/results/full_inv_stats.json), [tuning](outputs/results/full_inv_tuning.json), [diagnostics](outputs/results/full_inv_diagnostics.csv) |
| Data audit | [step0_report.txt](outputs/step0_report.txt), [step0_extra.txt](outputs/step0_extra.txt) | [step0_m5_report.txt](outputs/step0_m5_report.txt) |
| Run logs | [full_run.log](outputs/full_run.log) | [full_inv_run.log](outputs/full_inv_run.log) |

Other: [analysis plan](analysis_plan.yaml) (hash in [analysis_plan.sha256](outputs/analysis_plan.sha256)), [sanity results](outputs/sanity_results.json), [DATA_NOTES.md](DATA_NOTES.md), [PROGRESS.md](PROGRESS.md), [update-rule verification](../gtd_etd_verification.md).

## Run
```
pip install -r requirements.txt          # already satisfied in the dev environment
python -m pytest tests -q
python run_experiment.py --quick         # ~1 min
python run_experiment.py --full --yes    # ~37 min on 16 cores; prints an estimate first
python make_report.py full               # outputs/results_full.md + outputs/figures/*.png
```
`analysis_plan.yaml` is hashed into `outputs/analysis_plan.sha256` before the first run; the runner refuses to run if it changes.
Raw per-agent rows, per-rep metrics, diagnostics, tuning tables and stats are in `outputs/results/`.

## Update rules implemented (verified against Sutton & Barto 2018 Ch. 11 and the 2009/2016 papers; see `../gtd_etd_verification.md`)
x = x(S_t), x' = x(S_t+1), rho = pi/b, delta = R + gamma w.x' - w.x, w0 = v0 = 0, beta = kappa*alpha.
- GTD2: v += beta*rho*(delta - v.x)*x ; w += alpha*rho*(x - gamma x')(x.v)
- TDC: same v; w += alpha*rho*(delta x - gamma x'(x.v))
- ETD(lambda), interest 1: F_t = gamma rho_{t-1} F_{t-1} + 1; M_t = lambda + (1-lambda) F_t; e_t = rho_t(gamma lambda e_{t-1} + M_t x); w += alpha delta e_t. F=1 and e=0 at each window start ("reset"); `ETD_noreset` carries them across windows; `ETD_clip` clips F at 50 (a deliberate, non-standard variance fix).
- Window-end bootstrapping: truncated windows bootstrap at the last step (no terminal zeroing).

## Setup (key decisions)
- State (day, position), action = target position {flat, long}; reward = a*next-day return - cost*|a - pos|, 5 bps cost (assumed). Adjusted `close` only for returns; SAR flag uses unadjusted high/low; prices never repaired.
- Behavior: SAR-up rule w.p. 1-eps, else uniform (eps 0.3 base); target eps 0.1; gamma 0.95; window H = 100.
- Ground truth: exact finite-horizon DP on (day, position) (stronger than rollouts; mid-window values carry a gamma^H truncation error).
- Splits chronological: train 60% / validation 20% (70% OPE-log, 30% tuning-log) / test 20%; 3 extra walk-forward folds.
- 29 agents: 25 rule agents + DQN x2 + CQL x2 (offline, trained on the train period only).
- Linear features: 62-dim sparse (257-dim dense ablation), norm 1.
- Hyperparameters (alpha, kappa, lambda) chosen on the tuning-log with data-only objectives (MSPBE; emphatic PBE for ETD), never ground truth.

## Results (medians over 10 reps, primary fold; full tables in [results_full.md](outputs/results_full.md))
| Method | nRMSE of J | Spearman | nRegret@1 | diverged |
|---|---|---|---|---|
| TD(0) | 0.19 | 0.93 | 0.047 | 0 |
| GTD2 | 0.23 | 0.92 | 0.022 | 0 |
| TDC | 0.35 | 0.92 | 0.012 | 0 |
| ETD (lambda tuned, reset) | 14.5 | 0.57 | 0.39 | 7% |
| ETD_clip (F<=50) | 0.33 | 0.80 | 0.043 | 0 |
| LSTD | 0.23 | 0.91 | 0.047 | 0 |
| FQE | 0.16 | 0.80 | 0.043 | 0 |
| WIS (clipped) | 1.25 | 0.22 | 0.45 | 0 |
| Oracle projection (upper bound) | 0.16 | 0.94 | 0.029 | 0 |

Reading (this instrument, these settings only):
- GTD2, TDC and TD/LSTD rank agents about equally well (Spearman 0.9). No reliable winner among them: the pairwise Wilcoxon tests have n = 10, whose smallest possible p is 0.002, so Holm correction over 66 pairs can never reach 0.05. Differences are reported with Cliff's delta and median paired differences instead; treat "significant" claims with caution.
- Plain ETD is unstable here: gamma^2 E[rho^2] is 1-5 for most agents (max rho about 6.3), so the emphasis variance is very large (see `etd_emphasis_variance.png`). This matches the book's variance warning; it is a finding, not a bug. Resetting F per window does not cure it; clipping F helps but biases the estimate.
- Importance-sampling WIS is poor at this overlap; FQE and LSTD are strong baselines.
- Tuning picks the smallest alphas (3e-4 to 1e-3) with objective ratios near 0.93-1.0 of the w=0 value, so the data-only tuning signal is weak. Estimates are therefore close to the initial weights and partly reflect the small step size, not full convergence.
- Ablations (5 reps each; `outputs/results_full.md`): wider behavior coverage (eps 0.6) and larger logs help the gradient methods; narrow coverage (eps 0.1) hurts all; protocol B (evaluation on the test period, i.e. distribution shift) degrades every method (Spearman about 0.3-0.5); dense features inflate ETD errors; kappa = 0.1 stress does not break GTD2/TDC at the tuned alphas.
- Picked-agent test performance ([picks_performance.png](outputs/figures/picks_performance.png)) is noisy and agent picks are not significantly different; do not read it as a profitability claim.

## Trading performance (business metrics) and what is and is not claimed
Produced by `python business_metrics.py` -> [business_metrics_trading.md](outputs/business_metrics_trading.md) / [csv](outputs/business_metrics_trading.csv) (primary fold, test period 923 days, TECHM, 5 bps cost, 252-day annualisation; greedy agents replayed from flat). It reports cumulative (net and gross), annualised return, volatility, Sharpe, Sortino, max drawdown, Calmar, win rate, profit factor and turnover for all 29 agents, plus benchmarks and what each estimator's pick earned.
- **Continuously compounded view:** `log_return` = ln(final equity), `cont_rate_pa` = log_return x 252 / days (annualised continuous rate), `profit_per_100k` = 100,000 x cumulative return. Buy-and-hold: log return 0.705, continuous rate 19.3% p.a. (simple annualised 21.2%); best rule agent `rsi_mr_L40_U60`: 14.6% p.a.; `dqn_6000`: 12.6%; pool median: 6.6%; worst (`random_p0.2`): -3.8%. These are the same results as the table above expressed as rates, not extra evidence.
- **Benchmarks (test):** buy-and-hold +102% (Sharpe 0.86, max drawdown 29%); always-flat 0; pool median +27% (Sharpe 0.56); rule-agent median +29%; learned-agent (DQN/CQL) median +35% but with turnover about 4x higher (0.41 vs 0.11 per day), so costs take roughly 15 points of cumulative return (gross 0.52 vs net 0.35); random-position agents +10% (Sharpe 0.12).
- **No agent beats buy-and-hold on cumulative return (0 of 28)**; 2 of 28 beat it on Sharpe (`rsi_mr_L40_U60` 0.93, `rsi_sar_L50_U80` 0.89), both with lower drawdown. The test period was a strong rally, so any long-biased policy is judged against a very high bar.
- **Does OPE truth predict test performance?** Spearman across the pool between the exact OPE-period value J and test cumulative return is 0.33, with Sharpe 0.14 and Calmar 0.11. The best agent by validation truth (`mom20`) earned +56% (Sharpe 0.70), below buy-and-hold.
- **Estimator picks (median over 10 reps):** TD, LSTD, FQE, ETD_clip pick `buy_hold` in most reps (+102%); GTD2 and TDC pick `rsi_gt50` (+21%, Sharpe 0.37); unclipped ETD and OracleProj usually pick `random_p0.8` (+46%); WIS +40%; a random pick +26%. Picking `buy_hold` mostly reflects its high J on a drifting stock, so the estimators' good picks here are not evidence of skill beyond that. Earlier README text already warns about this; the numbers are noisy and are not significantly different across methods.
- **Claims I make:** (1) the update rules match Sutton & Barto Ch. 11 and the 2009/2016 papers, and the unit tests reproduce the hand-computed and fixed-point numbers; (2) under this exact-DP setting, TD/GTD2/TDC/LSTD/FQE value estimates are closer to truth than WIS, and unclipped ETD has very high variance (emphasis variance diagnostics); (3) ranking quality is moderate (Spearman about 0.9 for the TD family on the OPE period) and degrades when evaluated on a later period (protocol B, about 0.3-0.5).
- **Claims I do NOT make:** that any agent or estimator is profitable or deployable; that results generalise beyond one stock and one 923-day test period; that GTD2/TDC beat TD (they do not on value error, only partly on regret); that tuned step sizes are converged; that the transaction-cost, price-scale and SAR assumptions are right. The business-metric table is a descriptive evaluation on one realised path, with no confidence intervals and no multiple-testing correction, and the test period was visible to nothing in training or tuning but is a single market regime.
- **Inventory business metrics** ([table](outputs/business_metrics_inventory.md)): test period 388 days, one item (FOODS_2_197_CA_4), USD with assumed costs. Best base-stock by test profit (`base_stock_6d`) 4,006 (fill rate 1.0); best by validation truth (`base_stock_3d`) 3,161; oracle best by test profit (`sS_5_7d`) 4,213; CQL agents 4,158 / 4,004; never_order -1,920; order_max -60,831; pool median 2,532; plain DQN median -1,856 (still collapsed). Validation-period truth vs test profit Spearman 0.45 (vs fill rate -0.11): the OPE-period truth is only a moderate guide to test profit because the truth penalises stockouts and holding with discounting over 100-day windows. Window-bootstrap 95% CIs (resampling truth windows only, estimators fixed) are in the table; e.g. Spearman TD 0.36 [0.30, 0.45], TDC 0.30 [0.17, 0.50], ETD 0.32 [-0.13, 0.53], GTD2 -0.03 [-0.33, 0.26]. Estimator picks: TDC picks `cql_a0.1` (test profit 4,158); TD/LSTD/FQE pick the lowest-stock rule `base_stock_0.5d` (-946). Descriptive only, no claim of real-world profit.

## Convergence conditions (from the sources; recorded for the test-bench)
- GTD2/TDC (Sutton et al. 2009): converge to the TD fixed point under on-policy and off-policy sampling for linear features with the standard stochastic-approximation step-size conditions, linearly independent features and nonsingular A, C (Thm 1, p. 4; Thm 2, p. 5). Convergence is for i.i.d./stationary sampling, not for truncated windows of 100 steps.
- ETD (Sutton, Mahmood & White 2016): expected/in-expectation convergence to the emphatic fixed point under the stated conditions (Thm 1, pp. 18-19); finite variance is NOT guaranteed, which is what the diagnostics above show.
- Truncated windows with reset do not satisfy the continuing-process assumptions; the unit tests show a small effect on the tiny MDP (about 0.4%) but no guarantee on real data.

## Caveats
- ONE instrument (TECHM): every trading conclusion rests on a single stock and a single price history.
- Costs, the SAR/price-scale handling and the behavior rule are assumptions; the SAR price-scale treatment is UNVERIFIED against the data vendor.
- Ground truth is exact only for the DP MDP defined here, not for real trading.
- Intervals: a full bootstrap over the log is only feasible for LSTD/WIS; gradient-method uncertainty reflects evaluation windows only. The `tie_matrix` (moving-block bootstrap, 20-day blocks) is implemented and unit-tested but is not reported in the tables.
- Per-condition re-tuning is done once per fold/gamma/cost/eps pair, not for every ablation.
- Inventory (M5): costs (unit cost 0.6 x mean price, holding 0.5% of unit cost, stockout penalty 0.2 x price, fixed order cost 0.5 x unit cost), the 8 order levels, base-stock behavior policy and the 24-agent pool are assumptions. Ground truth is Monte Carlo (24 rollouts per start day, 12 for state values, common random numbers) on the real demand path, so it carries sampling noise. Tuning again picks small step sizes. Plain DQN agents still collapse to near never_order (more training did not fix it; CQL agents work), which lowers effective pool diversity. Window-bootstrap CIs resample the truth windows only; estimator re-estimation noise is the spread across the 10 repetitions. Inventory and trading results use different split fractions (see `config.yaml: inventory.splits`).
- Inventory results are in `outputs/results_full_inv.md`; they were produced after the analysis plan was locked and are an extension, not pre-registered.
