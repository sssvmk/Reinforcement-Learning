# Energy procurement – README

Day-ahead electricity procurement for a 10 MW Spanish customer (data 2015–2018, train 2015–16, validation 2017, test 2018). Classical agents (forecast, newsvendor, CVaR stochastic programming, mean-variance) are compared with policy-gradient methods (FDPG, REINFORCE, A2C, DPG, TRPO, PPO). Results: see `BUSINESS_OUTCOMES.md`; data: `DATA_NOTES.md`, `EDA_REPORT.md`; frozen analysis design: `analysis_plan.yaml`. The next section explains everything in plain language; the technical material follows it.

# Plain-language guide

## The problem
A company uses about 10 MW of electricity on average. Every day it must decide, **the day before**, how much power to buy for each of the next 24 hours on the Spanish *day-ahead* market. Two things go wrong:
- **Buy too little:** the shortfall is bought at the *adjustment (real-time) price*, which is usually higher.
- **Buy too much:** the surplus is wasted (or sold back cheaply).

The "normal" approach is to buy exactly what the grid operator's (TSO) load forecast says. The question: **can a smarter buying rule do better, and by how much?** Savings are always measured against buying the TSO forecast. We train on 2015–2016, tune on 2017, and report on a year the methods never saw (2018).

## What we learned about the data
- Real-time prices are higher than day-ahead prices most of the time, so buying slightly *less* than forecast is rarely punished much, and in about 11 % of hours buying nothing day-ahead is the best choice.
- The forecast of load is already good (error about 1.5 %). Most of the money comes from predicting the **price gap** between the two markets, not from predicting load better.
- Buying nothing day-ahead costs 10 % more than the TSO forecast; buying the maximum costs 49 % more.

## The "classical" methods (the ones that work)
Think of each as a smarter way to decide the daily purchase.
1. **Forecast model.** A gradient-boosted decision-tree model (GBM) predicts not just one number but a *range* (quantiles) for tomorrow's load error and price gap.
2. **Newsvendor.** The classic "how many newspapers to stock" rule: buy the amount where the cost of buying too little balances the cost of buying too much, using the forecast range.
3. **CVaR stochastic programming.** Like the newsvendor, but it also tries to protect against the *worst* few days (the worst 5 %), trading a little average saving for safer outcomes.
4. **Mean-variance.** Penalises day-to-day swings in cost, so costs become steadier.

## The "policy-gradient" methods (learning by trial and error)
These are reinforcement-learning methods: a small model proposes the hourly purchase, sees the resulting cost, and nudges itself to do better, over thousands of simulated days. We tried six families (the names matter only to specialists): **FDPG** (nudges using small test changes to its settings), **REINFORCE**, **A2C**, **TRPO**, **PPO** (all variants of "repeat what worked, with safeguards against big jumps") and **DPG** (learns an internal estimate of cost, then improves against it). We also tested simple linear models and small neural networks (PyTorch) as the "brain".

## How we judged fairness and success
- **Savings** in euros per day, percent, and per year, with a confidence range (we resample weeks of data to see how uncertain the number is).
- **Skill score:** 0 = no better than buying the TSO forecast, 1 = as good as the classical method on the same goal. It lets different goals be compared.
- Every method got the same data, the same training budget (30,000 simulated days), and several random restarts (seeds); settings such as learning speed were chosen on 2017 only, never on 2018.

## The results

**1. Smart buying saves money, modestly.** For a 10 MW customer in 2018 (about EUR 14,200/day of cost):

| Method | Saves per day | Percent | Per year |
|---|---|---|---|
| Newsvendor with price-gap forecast | EUR 184 | 1.3 % | ~EUR 67k |
| Mean-variance | EUR 140 | 1.0 % | ~EUR 51k |
| CVaR (risk-averse) | EUR 136 | 1.0 % | ~EUR 50k |
| Newsvendor without price-gap forecast | EUR 42 | 0.3 % | ~EUR 15k |
| Just correcting the forecast's bias | EUR 5 | 0.03 % | ~EUR 2k |

The ranges are wide (for example 62–352 EUR/day for the best method), so the saving is real but imprecise. The "oracle" that knows the exact load saves only EUR 73/day: knowing the load is **not** the main prize; predicting the price gap is.

**2. Risk protection is small.** The risk-averse versions reduce the bad-day cost by only about 1.5 %; asking for much more safety eliminates the saving.

**3. It holds up over time.** Re-training every quarter (2017–2018) keeps the savings positive in 7–8 of 8 quarters, averaging about EUR 145–180/day. One quarter (2018 Q3) was bad for the plain newsvendor (it lost money), while the risk-averse versions stayed positive. Quarter-to-quarter swings are large (about EUR 10 to 380/day).

**4. Neural network vs tree model for the forecast.** A small PyTorch neural network gives the same savings as the tree model (within uncertainty), trains in 40 seconds instead of ~22 minutes, and its ranges are slightly better calibrated, but its point predictions are a bit less accurate. It is a cheaper substitute, not an upgrade.

**5. Learning-by-trial-and-error did NOT beat the classical methods.**
- Learning from scratch, the best method (FDPG) recovers only about 26 % (plain goal) to 40 % (risk-averse goal) of what the classical method achieves; the others recover 0–25 %. Several often "blew up" during training.
- Starting from the classical answer, all methods simply *keep* about 88 % of it and never improve.
- Giving them a cleaner, less noisy objective ("conditional risk"), or using neural networks instead of linear models, helps somewhat (best 0.2–0.4 of the classical benefit) but does not close the gap.
- Why: the daily gain at stake is only 1–2 % of the load, buried in market noise, so trial-and-error learning has very little signal. The classical methods use known cost formulas directly and are also far cheaper to run.

**Bottom line:** use a forecast of the price gap plus a newsvendor-style rule (CVaR or mean-variance if you want safer costs). Expect about 1 % savings, re-train regularly, and do not expect reinforcement learning to improve on it for this problem.

## Things to keep in mind
- One customer size and one test year (plus rolling checks); numbers are 2015–2018 market conditions.
- Some mean-variance policy-gradient "skills" above 1 look like wins but are an artefact of mismatched objectives; ignore them (details below).
- The EUR savings use the classical, conditional-risk results; policy-gradient results are research findings, not business advice.

# Detailed walkthrough for non-specialists

This section expands the short guide above. No maths or programming knowledge is needed.

## A. The business problem in detail

**Who is buying?** A factory-sized customer using about 10 MW of power on average (roughly EUR 14,200 of electricity per day in 2018).

**How the market works.** Spain has two relevant prices for each hour:
1. **Day-ahead price (pDA):** you commit the day before how many MWh you want for each of the 24 hours of tomorrow, and pay this price.
2. **Adjustment / real-time price (pA):** whatever you did not pre-buy (shortfall) is settled afterwards at this price.

**The dilemma.** Tomorrow's real consumption is not known today. If you buy too little, the missing energy is bought at the adjustment price, which is *usually higher*. If you buy too much, the extra is wasted or resold cheaply. So each hour has two costs:
- *cost of being short* = adjustment price minus day-ahead price (usually positive);
- *cost of being long* = what you paid for energy you did not use.

**Worked example.** Tomorrow at 3 pm you expect 10 MWh. Day-ahead costs EUR 50, adjustment EUR 56. Buying 10 and using 10 costs 500. If you use 11, the extra MWh costs 56, so 556. If you use only 9 you wasted EUR 50. Because the shortfall penalty is EUR 6 but being long wastes the full EUR 50, the best buy is slightly *below* what you expect. In a few hours the adjustment price is *below* day-ahead, and then the best choice is to buy nothing ahead at all (about 11 % of hours).

**The benchmark.** The grid operator (TSO) publishes a load forecast; buying exactly that is the "default". Every result below is the money saved compared with the default. Two sanity limits: buying nothing costs 10 % more than the default, buying the maximum costs 49 % more.

**Two sources of uncertainty:** (1) how much power you will really use (the TSO forecast is already good, error about 1.5 %), and (2) how far apart the two prices will be (hard to predict, and this is where most money is).

**How we test honestly.** Methods learn from 2015-16, settings are tuned on 2017, and the final score is on 2018, a year they never saw.

## B. The classical methods and how each solves the problem

They build on each other like steps of a ladder.

### B1. Forecast model (the "eyes")
*What it does:* predicts tomorrow's load error and the price gap for each hour, not as one number but as a **range** (for example "the gap is probably between 2 and 9 EUR, most likely 5"). It is a gradient-boosted decision-tree model, which is a large collection of simple yes/no rules (time of day, weekday, season, recent prices, wind and so on) that vote.
*How it helps:* it turns "I don't know what will happen" into "here is how likely each outcome is". On its own, simply correcting the TSO forecast for its average bias saves only about EUR 5/day, so a forecast alone is not enough; it must feed a decision rule.

### B2. Newsvendor (the "decision rule")
*The idea:* a newspaper seller must choose how many papers to stock. Too few means lost sales; too many means waste. The best amount is not the average demand: it is set by *how costly each mistake is*. The same logic applies to each hour of electricity: buy the amount at which the chance of being short, weighed by the shortage cost, balances the chance of being long, weighed by the waste cost.
*How it helps:* because being short is cheap and being long is expensive, it buys slightly less than forecast (and sometimes nothing). We tested two versions: one that uses only past load errors (saves about EUR 42/day), and one that also uses the forecast price gap (saves about **EUR 184/day, 1.3 %, roughly EUR 67k/year**). The second version is the best average performer in the study.
*Weakness:* it only looks at the average cost, so a bad day can still be costly.

### B3. CVaR stochastic programming (the "cautious planner")
*The idea:* instead of only the average, also look at the **worst 5 % of days** (CVaR = average cost on those days). The model generates hundreds of possible tomorrows from the forecast ranges and chooses the purchase that keeps average cost low *and* the bad-day cost under control, with a dial (lambda) for how cautious to be.
*How it helps:* it is insurance. In our tests it saves about EUR 136/day (1.0 %, ~EUR 50k/year), less than the newsvendor, and lowers the bad-day cost by about 1.5 %. Asking for much more protection wipes out the savings, so protection is cheap only in small doses.

### B4. Mean-variance (the "steady-cost planner")
*The idea:* the same scenarios, but the penalty is on how much daily cost *swings* (variance). Budgeting is easier when costs are predictable.
*How it helps:* saves about EUR 140/day (1.0 %, ~EUR 51k/year) with smoother costs. It performs about the same as CVaR; which to choose depends on whether the company worries more about rare bad days (CVaR) or general volatility (mean-variance).

## C. The reinforcement-learning methods

### C1. What reinforcement learning means here
Instead of being told the cost formula, a program (the **policy**) acts like a trainee buyer. Each simulated day it sees the situation (hour, forecasts, recent prices), proposes how much to buy, then is shown the actual cost. Over tens of thousands of simulated days it adjusts itself so that actions leading to lower cost become more likely. Our budget was 30,000 simulated days per run, with several random restarts (seeds) for fairness.

**Why try it?** The classical methods need a forecast model, a scenario generator and an optimizer, each with assumptions. A learning agent could in principle discover a better rule directly from experience, adapt without hand-built formulas, and cope with any objective (average cost, worst-case, steadiness) by changing the reward. The question was: does it?

The "brain" of the trainee was either a **linear model** (a weighted sum of the inputs) or a small **neural network** (PyTorch), which can capture more complex patterns.

### C2. The six families, in plain words
- **FDPG (finite-difference policy gradient):** wiggle each setting slightly up and down, see which wiggle reduces cost, and move that way. Simple and steady.
- **REINFORCE:** try actions with some randomness; make actions followed by good outcomes more likely and bad ones less likely. Simple but noisy.
- **A2C (advantage actor-critic):** REINFORCE plus a second model (the critic) that estimates "what cost was expected here"; the actor only learns from whether the result was better or worse than expected, which reduces noise.
- **TRPO:** like A2C but never allows the policy to change too much in one step, to avoid sudden collapses.
- **PPO:** a simpler, widely used version of the same safety idea (clips how much the policy may change per update).
- **DPG (deterministic policy gradient):** the critic learns the cost of any purchase, and the policy is nudged straight towards the purchase the critic thinks is cheapest.
(SPSA, a random-perturbation relative of FDPG, was also evaluated.)

### C3. Three experiments with them
1. **Cold start:** the agent starts knowing nothing.
2. **Warm start:** the agent begins with the classical answer copied in, and is then allowed to improve it.
3. **Cleaner objective and neural policies:** a less noisy "conditional risk" reward (risk judged per day, as the classical methods do), re-tuned learning speeds per objective, and neural-network policies.

## D. Results explained

**Scoreboard unit:** the **skill score**. 0 means no better than the default TSO purchase; 1 means matching the classical method trained for the same goal. Skill of 0.26 means the agent captured about a quarter of the classical saving.

### D1. Classical methods (2018, EUR saved per day vs default)
| Method | EUR/day | % | EUR/year | Plain reading |
|---|---|---|---|---|
| Newsvendor with price-gap forecast | 184 | 1.3 | ~67k | Best on average; uncertainty range 62-352 |
| Mean-variance | 140 | 1.0 | ~51k | Steadier costs |
| CVaR | 136 | 1.0 | ~50k | Protects bad days |
| Newsvendor, no price-gap forecast | 42 | 0.3 | ~15k | Shows price-gap forecast matters |
| Bias correction only | 5 | 0.03 | ~2k | Forecast alone barely helps |
| Oracle knowing exact load | 73 | | | Less than the forecast-based methods: perfect load knowledge is not the main prize |

### D2. Is it stable over time?
Retraining every quarter for 2017-2018 gives positive savings in 7 of 8 quarters for the newsvendor and 8 of 8 for CVaR and mean-variance (average about EUR 145-180/day). The one bad quarter, 2018 Q3, made the plain newsvendor lose money (about EUR 144/day with trees), while the cautious versions did not. Lesson: the newsvendor wins on average, but CVaR/mean-variance are safer.

### D3. Tree forecaster vs neural-network forecaster
| Policy | Trees (EUR/day) | Neural net (EUR/day) |
|---|---|---|
| Newsvendor | 182 | 174 |
| CVaR | 136 | 172 |
| Mean-variance | 140 | 157 |

The uncertainty ranges overlap, so they are statistically the same. The neural network is slightly less accurate on single-number predictions, slightly better calibrated on ranges, and trains in about 40 seconds instead of 22 minutes. Verdict: a cheaper substitute, not an improvement.

### D4. Reinforcement learning
| Setup | Best skill | Plain reading |
|---|---|---|
| Cold start, average-cost goal | FDPG 0.26 (SPSA 0.18, A2C 0.17, TRPO 0.09, REINFORCE 0.06, DPG ~0, PPO 0) | Best learner gets about a quarter of the classical saving; several collapse |
| Cold start, CVaR goal | FDPG 0.40 | Still well below 1 |
| Warm start (classical answer given) | about 0.9 to 1.1 | It only keeps what it was given; no improvement |
| Cleaner conditional-risk goals | FDPG 0.27 (average), 0.18 (CVaR), 0.44 (mean-variance) | Fairer comparison, same conclusion |
| Neural-network policies | 0.1 to 0.18 | Not better than the simple linear policy |

*Note:* some mean-variance results show skill above 1 (even 3). These are an artefact: the agent was optimizing a different (cruder) definition of "variance" than the classical method, and the reference gain is tiny. They are not real wins.

**Why RL struggled:** the gain at stake each day is only 1-2 % of the load, and market noise is much larger than that, so there is almost no learning signal. The classical methods use the cost formula directly and know exactly which direction is better, whereas RL has to discover it from noisy trial and error. RL also costs far more computing (minutes to hours per run versus seconds to minutes).

**What RL is still good for:** it flexibly handles any objective without formulas, and it is a useful check. Here it confirmed that the classical solution is hard to beat, because warm-started agents never found anything better.

## E. The final winner

- **Overall winner: the newsvendor rule driven by the price-gap forecast** (about EUR 184/day, 1.3 %, ~EUR 67k/year for a 10 MW customer), with the forecast built by either trees or the faster neural network.
- **If the company cares about risk:** choose **CVaR** (protects worst days) or **mean-variance** (smoother costs); they give up about EUR 45/day for steadier outcomes and held up in every quarter.
- **Reinforcement learning:** did **not** win. Best case, FDPG learnt up to about 0.2-0.4 of the classical benefit from scratch, and none improved on it from a warm start.
- **Key business insights:** (1) predict the price gap, not just the load; (2) buy slightly below forecast; (3) retrain regularly; (4) expect roughly 1 % savings, not more.

**Caveats:** one customer size, one test year (plus quarterly checks), 2015-2018 market conditions; wide uncertainty ranges (the best-method estimate is EUR 62-352/day); not financial advice.

# Technical guide

## Layout
- `config.yaml` data paths/parameters; `data.py`, `features.py`, `costs.py`, `risk.py`, `problem.py`, `metrics.py` core modules.
- `agents/` classical agents (incl. `forecast_nn.py`, PyTorch quantile MLP); `pg/` policy-gradient environment, policies and algorithms.
- `run_eda.py`, `run_agents.py`, `run_pg.py`, `run_extensions.py` experiment drivers; `evaluate_algorithms.py`, `evaluate_business.py` Stage 7 evaluation.
- `tests/` pytest suites (data/costs, agents, PG incl. known-optimum recovery, business).
- `outputs/` all generated results (`agents`, `pg`, `pg_warm`, `eval`).

Run from this directory; `pip install -r requirements.txt`. CPU only; the data folder is read-only.

## Reproduce
Runtimes are wall-clock on a 16-core CPU, measured unless marked "~" (rough).

| Step | Command | Runtime |
|---|---|---|
| EDA | `python run_eda.py` | ~minutes |
| Classical agents | `python run_agents.py` (`--quick` for a smoke run) | ~tens of minutes (GBM forecast bundle alone: ~22 min, cached) |
| PG smoke | `python run_pg.py --quick` | ~0.3 min per run |
| PG lr grid (validation only) | `python run_pg.py --grid --objective <obj>` | ~10–20 min per objective with `--jobs 10` |
| PG full | `python run_pg.py --full --jobs 10 [--objective <obj>] [--seeds 5] [--warm --out outputs/pg_warm]` | a few hours for everything; PPO is ~8–14 min per run, others ~1–6 min |
| PG conditional-risk objectives | `--objective neutral_cond\|cvar_cond\|meanvar_cond` (or `all_cond`) | as above |
| PG neural policy | `--policy mlp` (not FDPG) | as above |
| Neural vs GBM forecaster; rolling-retrained forecasters | `python run_extensions.py --static --rolling --jobs 6` (`--quick` smoke) | static ~50 min, rolling ~2–3 h (GBM refits dominate; NN refit ~1 min) |
| Algorithm evaluation | `python evaluate_algorithms.py --full` (`--quick` for 3 seeds) | a few minutes |
| Business evaluation | `python evaluate_business.py --full --jobs 6` (`--quick` skips rolling origin) | ~50 min |
| Tests | `python -m pytest tests -q` | ~3 min |

## Deviations / caveats (please read)
1. **The oracle is not a lower bound**: it knows the load but not the price gap; GBM policies beat it.
2. **Cold-start PG underperforms** the classical agents (low gradient signal-to-noise); warm start only preserves the distilled skill (~0.88). To get stability we added regime features (`g_mean<=0`, `g_mean/pDA`), Polyak averaging of the evaluated policy and best-validation checkpointing; these are applied equally to all algorithms.
3. **Mean-variance/CVaR skill > 1** for the unconditional objectives reflects an objective mismatch (unconditional cross-day variance vs. the classical conditional per-day risk) and a tiny skill denominator; not a business result. The `*_cond` objectives remove the mismatch (PG optimises the same scenario-based conditional risk the classical agents do); skill stays well below 1 there.
4. Learning-rate grids were run for every objective (validation only, seeds 100–101, 12k rollouts) in `outputs/pg_v2`. The original cold/warm runs in `outputs/pg`, `outputs/pg_warm` reused the neutral grid; re-tuned unconditional runs (`outputs/pg_v2`, 5 seeds) give the same ranking.
5. Rolling-origin retraining: bias-scenario agents (`evaluate_business.py`), and the GBM and neural forecasters themselves (`run_extensions.py`). Rolling hyper-parameters are fixed; scenario residuals come from earlier out-of-sample forecasts, not re-cross-fitted per quarter.
6. Learning budget is equal across algorithms in rollouts (days) – 30,000 for the full run.
7. Neural components: `agents/forecast_nn.py` (PyTorch MLP quantile forecaster) and the MLP policy / Q-network / value-network in `pg/`. The main PG experiments use linear policies (see 2); MLP policies were run on the conditional objectives only (5 seeds). The MLP forecaster matches but does not beat the GBM.
8. Data caveats: system-wide load rescaled to a 10 MW customer; 2015–2018 only; see `DATA_NOTES.md`.
9. Fixed bug during development: `saa_quantity` returned the max scenario load instead of the cap (fixed, covered by tests).

