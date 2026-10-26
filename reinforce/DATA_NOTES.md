# DATA_NOTES

Generated from `audit_data.py` (outputs in `outputs/audit/`). Nothing was downloaded; data are read only.

## Trading: TECHM (`TECHM_extend.csv.gzip`, read with `compression="gzip"`)
- 4,677 rows, 2007-09-17 .. 2026-10-23 (dates monotonic, none duplicated). The end date lies after the date this analysis was run; it is taken as given and not corrected.
- RSI(14) matches Wilder's definition exactly (MAE 7e-10, timing lag 0). Parabolic SAR best matches step 0.02 / max 0.3 (84% within 0.5%); SAR trend uses prior data only (99.8%).
- **Price-scale audit FAILED**: `close` is inside `[low, high]` on only 1.1% of rows (close/mid-price ratio about 0.65 in early years), i.e. `close` is split/dividend-adjusted while OHLC are not.
- **Deviation (same as FDPG, config `allow_adjusted_close: true`)**: returns use adjusted `close`; SAR and RSI features stay as supplied. Restricting to consistent ranges leaves about 50 rows, unusable. Return levels are therefore approximate for long horizons.
- Features at close t: RSI, SAR trend flag, SAR-distance, 5-day return, volume z-score; standardised with TRAIN statistics only. `R[t]` = next-day return (never visible to the policy at t). Largest absolute daily returns (2009-06 +25.6%, +17.7%) are kept; they look like data-scale artefacts and are flagged, not removed.

## Inventory: M5
- Files found: calendar, sales_train_validation, sales_train_evaluation, sell_prices, sample_submission. `sell_prices.csv` loads; all selected series have price coverage 100%.
- Series (chosen by the same profiling rule as FDPG, `outputs/audit/m5_profile.csv`): `FOODS_3_124_CA_3_validation` (primary), `FOODS_2_128_WI_2_validation` (intermittent), `HOBBIES_1_371_WI_1_validation` (seasonal). Series are cached under `outputs/series/`.
- Splits 60/20/20 by time (train/validation/test). Windows are 90 days (inventory) or 250 days (trading) and never cross a split.

## MDP and cost assumptions (identical to FDPG so the projects are comparable)
- Inventory: lost sales, lead time L=2, daily order a_t in [0, Q_max], Q_max = 7*mu (REINFORCE and evidence B; FDPG had none). unit cost 0.6*mean price, holding 0.005*unit cost/day, stock-out 0.2*mean price per lost unit, fixed order cost 0.5*unit cost, initial stock mu*(L+1). Demand and prices are replayed (exogenous).
- Trading: long fraction in [0,1], reward = pos*R - 5bps*|dpos|, windows start flat, summed returns inside windows; test replay is compounded.
- Rewards are divided by a normaliser so J is in "norm units": std of train-window profit under the (L+1)mu cover policy (inventory), std of train-window buy-and-hold returns (trading). Discount gamma (0.99 / 0.995) is used only inside the gradient; reported J is the undiscounted window return.
