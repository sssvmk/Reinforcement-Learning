# PROGRESS

- Stage 0 – done: paths, `DATA_NOTES.md`.
- Stage 1 – done: EDA (`EDA_REPORT.md`).
- Stage 2–4 – done: cost/risk modules, forecasting, newsvendor, CVaR SP, mean-variance, benchmarks (`outputs/agents`).
- Stage 5–6 – done: PG algorithms, tests, lr grids, full cold (210 runs) and warm (105 runs) experiments.
- Stage 7 – done: `evaluate_algorithms.py --full`, `evaluate_business.py --full` (outputs in `outputs/eval`).
- Stage 8 – done: `BUSINESS_OUTCOMES.md`, `README.md`.

## Open questions
None. See README "Deviations / caveats".

## Extensions (done)
- Neural quantile forecaster (agents/forecast_nn.py) vs GBM; rolling quarterly retraining of both (run_extensions.py).
- Per-objective lr grids, conditional-risk PG objectives (*_cond), MLP policies (outputs/pg_v2).
- Docs updated; tests: 46 passed, 2 skipped.
