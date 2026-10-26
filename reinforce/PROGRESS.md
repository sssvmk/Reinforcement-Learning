# PROGRESS

| Stage | Status |
|---|---|
| 0 data audit (`audit_data.py`, `DATA_NOTES.md`) | done |
| 0b FDPG audit + hashes (`audit_fdpg.py`, `FDPG_INVENTORY.md`) | done; hashes unchanged after all runs |
| 1 environments, problems, benchmarks, metrics | done |
| 2 policies, `reinforce/` package, stats, tests | done (24 tests pass; machinery tests passed before `--full`) |
| 3 `run_reinforce.py` quick + full (+sensitivity) | done (full: 10 seeds, 40k episodes, ~50 min) |
| 4 `analysis_plan.yaml` (frozen before full), `evaluate_reinforce.py`, `evaluate_business.py`, `compare_with_fdpg.py` | done (quick + full) |
| 5 README, requirements | done |

DONE (exploratory, 30 tests pass): neural R-I4/R-T3 (10 seeds, cm-spsa/random, sensitivity), neural vs FDPG, FDPG-fold walk-forward (`outputs/wf_full/WALKFORWARD.md`), Q_max-free (`outputs/qfree/eval`), price-scale audit (`outputs/ropen/eval`), `FINAL_REPORT.md`. Nothing pending except confirming the TECHM price series with the data owner.
NOT REPRODUCED: none (11/11 FDPG test numbers reproduced).
