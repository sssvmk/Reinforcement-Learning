# DATA_NOTES

**CURRENT STATUS (supersedes the Stage-0 text below):** `sell_prices.csv` was supplied later, so inventory runs on 3 M5 series
(FOODS_3_124_CA_3, FOODS_2_128_WI_2, HOBBIES_1_371_WI_1). TECHM trading runs on adjusted `close` (documented deviation).
RSI = Wilder 14 on adjusted close (exact); SAR step 0.02 / max 0.3 matches 84% of rows (partly verified). See README.md.

## (Stage 0, historical) Inventory: was PENDING
M5 folder `C:\Users\username\OneDrive - XXYY\m5-forecasting-accuracy` had calendar.csv, sales_train_validation.csv, sales_train_evaluation.csv; `sell_prices.csv` was missing at that time.

## Trading: TECHM_extend.csv.gzip
- 4677 rows x 119 cols, 2007-09-17 to 2026-10-23, ascending, no duplicate dates, all required columns present.
- RSI in [12.6, 100]; sar > 0 (1 NaN on row 1).
- **Price-scale audit FAILS**: share of rows with low <= close <= high is 0% for 2007-2024, 0.4% in 2025, 25.5% in 2026.
  Median close/mid(high,low) rises smoothly from 0.65 (2007) to 0.97 (2026): `close` is adjusted for corporate actions
  (likely dividends/splits), `open/high/low/sar` are not. Consistent date range: essentially none (only part of 2026, ~50 rows).
- Per the rules prices are not repaired, so the file is excluded for return-based trading features.
- Largest |close returns|: 2009-06-09 (25.6%), 2009-06-03, 2020-03-23, 2008-10-10, 2008-01-22 (market events, plausibly real).
- Indicator audit (RSI/SAR recompute) not run: no usable data remains.

---
UPDATE: superseded by the audit findings and the adjusted-close deviation documented in README.md (Data section). Trading proceeds with adjusted close; inventory stays PENDING (sell_prices.csv missing).
