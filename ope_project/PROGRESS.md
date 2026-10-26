# PROGRESS

- [x] Stage 0 data audit (trading and M5): DATA_NOTES.md, outputs/step0_report.txt, outputs/step0_m5_report.txt
- [x] Stage 1 environments: trading and inventory (InvDomain tested against the scalar env)
- [x] Stage 2 agents (29 trading, 24 inventory); truth: exact DP (trading), Monte Carlo (inventory)
- [x] Stage 3 estimators + tests 6-12, 15
- [x] Stage 4 metrics.py, stats.py, run_experiment.py (analysis_plan.yaml locked)
- [x] Stage 5 trading --full (36.8 min), inventory --full (192.3 min, rerun after pool fix; first run archived), make_report.py, README.md
- [x] 43 tests pass
- [x] Done: inventory business-metric table, benchmarks, window bootstrap (outputs/business_metrics_inventory.md). Remaining known limitation: plain DQN agents still collapse (CQL works).
