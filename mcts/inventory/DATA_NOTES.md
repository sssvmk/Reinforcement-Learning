# M5 Inventory Data — Step 0 Notes

All findings below come from directly reading the local files in `M5_DIR`
(`C:\Users\msk80743\OneDrive - GSK\m5-forecasting-accuracy`), read-only.
Nothing was written back into `M5_DIR`.

## File listing (flat, no subfolders, no zips)
| File | Size (bytes) |
|---|---|
| calendar.csv | 103,469 |
| sales_train_evaluation.csv | 121,736,518 |
| sales_train_validation.csv | 120,007,726 |
| sample_submission.csv (ignored) | 5,228,786 |
| sell_prices.csv | 203,395,785 |

**Correction to the task description**: *both* `sales_train_validation.csv`
and `sales_train_evaluation.csv` are present (the task said not to assume
the evaluation file exists). Since both exist, we use
**`sales_train_evaluation.csv`** because it is a strict superset in time
(`d_1..d_1941` vs. `d_1..d_1913`), giving more history for the
train/val/test split. This choice is implemented in `inventory/data.py`.

## `calendar.csv`
Shape **(1969, 14)**. Columns match the documented schema exactly:
`date, wm_yr_wk, weekday, wday, month, year, d, event_name_1, event_type_1,
event_name_2, event_type_2, snap_CA, snap_TX, snap_WI`.
- Date range: **2011-01-29 to 2016-06-19** (matches "Jan 2011 to mid 2016").
- Null counts: `event_name_1`/`event_type_1` null in 1807/1969 rows,
  `event_name_2`/`event_type_2` null in 1964/1969 rows (events are rare, as
  documented); `snap_*` columns have no nulls (always 0/1).

## `sales_train_evaluation.csv`
Shape **(30490, 1947)**: `id, item_id, dept_id, cat_id, store_id, state_id`
+ `d_1 .. d_1941` (1941 day columns). Confirms the documented wide format
and ~30,490 item-store series.
(`sales_train_validation.csv`: same 30,490 rows, `d_1..d_1913`, not used —
see correction above.)

## `sell_prices.csv`
Shape **(6,841,121, 4)**: `store_id, item_id, wm_yr_wk, sell_price`
(`sell_price` is `float64`). Confirms a missing `(store_id, item_id,
wm_yr_wk)` row means the item was not sold/stocked that week — verified
directly below for one chosen series with a known pre-launch gap.

## Missing-price handling (decision, implemented in `inventory_env.py`)
A missing weekly price row is treated as "item unavailable that week":
prices are **forward-filled for gaps of up to 2 weeks** (brief stockroom/
listing gaps), and **longer gaps are excluded from episode windows** (the
series is trimmed to the date range with continuous or near-continuous
price coverage). This matches the task's documented default.

## Chosen series (5), profiles computed on full `sales_train_evaluation.csv` history

| id | role | mean demand/day | zero-fraction | CV | weekday CV | notes |
|---|---|---|---|---|---|---|
| `FOODS_3_477_CA_3_evaluation` | **mid-volume FOODS (primary/starter)** | 4.82 | 0.022 | 0.53 | 0.070 | available since day 1, flat weekday profile, full price coverage (282/282 weeks) |
| `FOODS_3_090_CA_3_evaluation` | **high-volume, strongly seasonal** | 130.79 | 0.185 | 0.82 | 0.198 | weekend-peaked (Fri/Sat ~1.3-1.5x midweek), full price coverage |
| `FOODS_3_586_TX_2_evaluation` | **high-volume, stable** | 100.53 | 0.004 | 0.31 | 0.183 | lowest zero-fraction of all candidates, full price coverage, weekend-peaked |
| `HOUSEHOLD_2_062_TX_1_evaluation` | **low-volume, highly intermittent** | 0.59 | 0.800 | 23.19 | 1.382 | extreme CV, one weekday (Thu, wday=5) carries most of the demand mass; full price coverage |
| `HOBBIES_1_234_CA_3_evaluation` | **bursty, late-launch, missing-price test case** | 15.61 | 0.419 | 1.76 | 0.063 | first sale at day index 260 (pre-launch zeros must be trimmed); **37/282 weeks have no price row** — exercises the missing-price-gap logic above |

Each row's "mean/zero-fraction/CV" is computed on the item's full observed
`d_*` history (no pre-launch trimming applied yet at profiling time); CV is
`std/mean` of daily units; weekday CV is the coefficient of variation of the
7 weekday-mean demands (a seasonality signal independent of overall
volatility). `HOBBIES_1_234_CA_3` was deliberately chosen because it both
needs leading-zero trimming and has genuine price gaps, letting the data
pipeline exercise both edge cases from the task.

## Leading-zero trimming
`HOBBIES_1_234_CA_3_evaluation` has its first nonzero sale at day index 260
(`d_260`); the pipeline trims all days before the first nonzero sale for
every series (treating them as "not yet launched"), per the task's
documented policy. The other 4 chosen series have a nonzero sale on day 1
(no trimming needed).

## SNAP flag mapping
`store_id` prefix maps to `state_id` (`CA_3 -> CA`, `TX_1 -> TX`,
`TX_2 -> TX`), confirmed directly from the `state_id` column in the sales
file; `data.py` selects `snap_CA`/`snap_TX`/`snap_WI` from `calendar.csv`
accordingly per series.
