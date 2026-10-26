# DATA_NOTES

Sources: outputs/step0_report.txt, outputs/step0_extra.txt (scripts step0_audit.py, step0_extra.py). Data read-only.

## TECHM (TECHM_extend.csv.gzip)
- 4677 rows x 119 cols, 2007-09-17 .. 2026-10-23, monotonic dates, no duplicates, required columns present.
- NaN in core columns: `sar` 1, `RSI` 1 (first row). One zero-volume day (2025-03-18).
- `nifty_volume` is not all zero in this file; it is dropped anyway.

## Price-scale finding (important)
- `low <= close <= high` holds in only 1.09% of rows (0% in 2007-2024). `close/mid` drifts smoothly 0.65 (2007) -> 0.97 (2026):
  `close` is corporate-action adjusted; `open/high/low/sar` are on a different (unadjusted) scale.
- Daily open-to-open returns differ strongly from close-to-close returns (std of difference 3.1%; |diff|>5% on 7.7% of days),
  and corr(close ret, nifty ret)=0.47 vs corr(open ret, nifty ret)=0.14. `open` is therefore NOT a trustworthy return series.
- Decision: rewards and return features use adjusted `close` only. Prices are never repaired. `close` is never compared with `sar`/`high`/`low`.
  Open/high/low/sar are used only for the within-scale SAR features (flag sar<low vs sar>high, distance (open-sar)/open... see features.py),
  flagged as lower-confidence because of the unadjusted/unaligned scale.

## RSI
Verified: Wilder smoothing, period 14, on `close`, trailing (shift 0), max abs diff 8e-9 after row ~300. Early rows differ
(seeding differs), so the first 60 rows are excluded from every sample. Use stored RSI after warm-up (equals own recompute).

## SAR
Best match: step 0.02, max 0.2 (TA-Lib style), shift 0. Only ~69% of rows match within 1e-3; flag (sar<low) agrees with the recompute 96%.
Mismatch share by year is erratic (13%-59%), cause not diagnosed (probably vendor initialisation/scale). Recompute is causal (truncation test).
Stored sar[t] contains day t's high/low, so it is known at close of t and is used only for the position held t -> t+1.

## Feature whitelist (look-ahead audit)
Used: 1/5/20-day log returns of close, RSI, SAR flag, SAR distance, trailing 60-day volume z-score, current position.
Excluded as unverified (could embed future data or unknown definition): rank, beta/alpha, d_*a* columns, all other indicator columns, nifty_volume, turnover.
Caveat: because of the unadjusted scale, SAR features may jump on ex-dates.

## Only one ticker
All trading conclusions rest on a single instrument.

## M5 inventory: READY and run
Path supplied by the user (`C:\Users\msk80743\OneDrive - GSK\m5-forecasting-accuracy`); audit in `outputs/step0_m5_report.txt`. Uses `sales_train_evaluation.csv`. Series: primary FOODS_2_197_CA_4, intermittent HOUSEHOLD_2_246_CA_1 (65% zeros), seasonal HOBBIES_1_371_WI_1. 0 invalid-price days in all three. Prices forward-filled at most 2 weeks. Results in `outputs/results_full_inv.md`.
The folder itself exists with calendar.csv, sales_train_{validation,evaluation}.csv, sell_prices.csv, sample_submission.csv.
`inventory.enabled: auto` turns it on when env var `M5_DIR` is set to the real path (or the braces are removed in config.yaml). No code change needed.
