# Value Iteration Inventory Control with Dynamic SAP APO Corridors

`inventory_corridor_vi.py` recommends order quantities for every
**CYCLE_DATE × MAT_ID × LOC_ID × YEARMONTH × YEARWEEK** using **Value Iteration**, constrained by the dynamic
min/max inventory corridor defined in `Test_3.1.6_LOCATION_MATERIAL_MD_HIST.r` and `corridor.pptx`.

> The script is Python and runs **outside** SAP. SAP APO / ECC provide the data and receive the recommended
> orders (see section 6).

---

## 1. Functionality

1. Reads all file paths/options from `config.json` (command-line flags override).
2. Streams the 3-year weekly supply/demand plan (`0011_apo_calc_demand_receipt.csv`) in chunks.
3. Joins master data: unit price / GOV_UNIT (`c_md.csv`), APO transit days (`tlane.csv`), and optional per-material-location order parameters (`params` CSV).
4. Builds the **dynamic corridor** (min, safety stock, cycle stock, tolerance, max) in days of cover and quantity.
5. Solves an inventory MDP with **Value Iteration** for each distinct normalised corridor (cached).
6. Applies the resulting policy week by week, rounds to MOQ / lot / rounding value, and writes order recommendations.

## 2. Inputs

| File | Fields used |
|---|---|
| `0011_apo_calc_demand_receipt.csv` | `CYCLE_DATE` (plan version), `MAT_ID`, `LOC_ID`, `YEARMONTH`, `YEARWEEK`, `TOTAL_DEMAND`, `TOTAL_RECEIPTS`, `ON_HAND_STK_QTY` (projected stock), `SAFETY_STK_QTY` (dynamic SS), `AVG_DMD_R6` (daily demand), `CD_DMD_R6`, `WD_DMD_R6` |
| `c_md.csv` | `UNIT_PRICE`, `GOV_UNIT`, `LOT_SIZE_CALC_METHOD` |
| `tlane.csv` | `APO_Transit_param` (transit days), `VALFR` / `VALTO` |
| params CSV (optional but recommended) | `MAT_ID, LOC_ID, PROC_TYPE, MOQ_QTY, BSTFE, LOT_SIZE_CALC_METHOD, PERIOD_TYPE, PERIOD_AMOUNT, INV_MIN_RV, GROUPING_CALENDAR_DAYS, SAFETY_STOCK_CALC_METHOD, SAFETY_STK_DAYS, OBLIGATED_STOCK_FLAG, STATIC_SS_QTY, IN_HOUSE_LT_DAYS, GOV_UNIT` |

Defaults when missing: PROC_TYPE `F`, lot method `E`, MOQ 0, rounding 1, lead time 14 days, price 1.0.
The demand file must be grouped by CYCLE_DATE, MAT_ID, LOC_ID.

`config.json`:
```json
{
  "paths":   { "demand": "...", "master": "...", "tlane": "...", "params": null, "out": "..." },
  "options": { "target": "max", "default_lt_days": 14, "default_price": 1.0, "chunk": 2000000,
               "cv": 0.3, "hold_rate": 0.25, "backlog_rate": 0.10, "ss_rate": 0.03, "min_rate": 0.03,
               "over_rate": 2.0, "fixed_cost_days": 0, "gamma": 0.98 }
}
```

## 3. Algorithm

### 3.1 Dynamic corridor (from the R script and pptx)
- **Safety stock**: dynamic (`SAFETY_STK_QTY` > 0 anywhere in the horizon) else static (`STATIC_SS_QTY`). Methods `SB/MB` (qty), `SZ/MZ` (days), `SM/MM` (max, or sum if obligated stock = `X`).
- **Cycle stock days** by lot-size method: `P` = max(grouping-calendar days, MOQ days, period days); `E`/`F` = max(grouping-calendar days, MOQ days) or max(MOQ days, 7); rounded up to the rounding value unless MOQ-driven.
- **PROC_TYPE**: `F`/`X` lead time = APO transit days; `E` = in-house time; for `F` the fixed lot `BSTFE` is the MOQ.
- **Tolerance**: Cx 20% of cycle days; Rx/Vx min(50%, 45 days).
- **MIN** = ceil(0.5 × SS days) × daily demand. **MAX** = min(ceil(SS + cycle + tolerance days), 410) × daily demand. No demand → no corridor.

### 3.2 Value Iteration MDP
| Element | Definition |
|---|---|
| State | Net inventory position, in days of cover (grid step *u*); backlog allowed |
| Action | No order, or order up to a level inside the corridor [MIN, MAX] |
| Demand | Weekly Normal(forecast, `cv` × forecast); lead time L weeks handled by costing the position at week t+L |
| Cost | Holding + backlog + below-safety-stock + below-MIN + above-MAX + fixed order cost |
| Discount | `gamma` |

Bellman update, iterated until the change is below `vi_tol`:

```
V(i) = min_j [ F·(j > i) + c(j) + γ · Σ_k p_k · V(j − k) ]
```

The solution is a table: for each inventory level, the order-up-to level (or no order).

### 3.3 Scaling to ~1M material-locations
The MDP is expressed in days of cover, so rows with the same normalised corridor (max, min, SS days, weekly demand
days, lead time) share one solved policy, which is cached. A few thousand solves cover millions of rows, and the weekly
simulation is vectorised across all material-locations.

### 3.4 Executing the policy
Each week, from the lead time onward: look up the target for the projected stock → quantity = target (days × daily
demand, capped at MAX) − stock → round to fixed lot (PROC_TYPE F, `BSTFE`) or MOQ, then to the rounding value → receipt
in that week, order released one lead time earlier. Weeks inside the lead time cannot be corrected and are flagged
`UNAVOIDABLE_SHORTAGE`.

## 4. Outcome (output CSV)

One row per CYCLE_DATE / MAT_ID / LOC_ID / YEARMONTH / YEARWEEK:

| Group | Columns |
|---|---|
| Corridor | `CORRIDOR_MIN_QTY`, `CORRIDOR_MAX_QTY`, `CORRIDOR_SS_QTY`, `CORRIDOR_CYCLE_STOCK_QTY`, matching `*_DAYS`, `SS_SOURCE` |
| Decision | **`ORDER_QTY`** (released that week), `VI_ORDER_UP_TO_QTY`, `PLANNED_RECEIPT_NEW` (arrives that week), `LEAD_TIME_WEEKS` |
| Result | `PROJ_STOCK_BEFORE/AFTER`, `DAYS_COVER_AFTER`, `ORDER_VALUE`, `UNAVOIDABLE_SHORTAGE` |

Sample (material `60000000000105`, `PL50`): ordering 17,706 in week 202735 clears the backlog by 202737, followed by
about 340 per week, with targets inside the 336–1,600 corridor. In a synthetic test of 780k rows (5,000
material-locations × 156 weeks), 99% of projected stock after ordering was at or above the corridor minimum.
A week split across two YEARMONTHs appears twice with the same order, so aggregate on unique YEARWEEK.

## 5. How to run

```
python inventory_corridor_vi.py                       # uses config.json
python inventory_corridor_vi.py --config other.json
python inventory_corridor_vi.py --demand x.csv --out y.csv
```
Requires Python 3 with `numpy`, `pandas`, `scipy`. Speed: about 23 s for 780k rows including solving, so roughly
1–1.5 hours for 156M rows (1M material-locations × 3 years) on one process. Split by MAT_ID range for parallel runs.
The deterministic version without Value Iteration is `inventory_corridor.py`.

## 6. Using it with SAP APO and SAP ECC

The script cannot execute inside APO or ECC. Typical flow:

```
SAP APO / ECC  --extract-->  CSV / Parquet  --python-->  orders CSV  --load-->  SAP APO / ECC
```

### 6.1 Extract (SAP → files)
| Need | Source |
|---|---|
| Demand/supply plan per CYCLE_DATE | APO planning-book / DRP snapshot (BW or Databricks extract = `0011_apo_calc_demand_receipt`) |
| Safety stock, lot-size key, MOQ, rounding value, fixed lot, procurement type | ECC material master MRP views (MARC: `EISBE`, `SHZET`, `SHFLG`, `DISLS`, `BSTMI`, `BSTRF`, `BSTFE`, `BESKZ`); APO product-location master and lot-size table (`SAFTY`, `SVTTY`, `SL_LSZ_RANGE`) – the sources read by the R script |
| Transit time | APO transportation lanes (`APO_Transit_param`) |
| Price | Material master / `c_md` |

Typical checks: ECC `MM03` (MRP views), `MD04` (stock/requirements list), `SE16N` (table extracts); APO `/SAPAPO/MAT1`
(product master). Build the params CSV from these tables, one row per MAT_ID/LOC_ID, and set `paths.params`.
(Transaction names are typical; confirm with your SAP team.)

### 6.2 Run
Schedule the script (Databricks job, Airflow or cron) after each APO planning cycle, with `config.json` pointing to the
fresh extracts.

### 6.3 Load back (files → SAP)
Rows with `ORDER_QTY > 0` are **recommendations**:
- **APO**: upload as planned orders or stock transfer requisitions through your CIF/BAPI or flat-file interface (release = `YEARWEEK`, receipt = `YEARWEEK` + `LEAD_TIME_WEEKS`).
- **ECC**: for PROC_TYPE `F` create purchase requisitions (`ME51N` or BAPI upload); for `E` create planned orders (`MD11`) or production orders.
- Alternatively feed the corridor min/max into APO/ECC as inventory targets and let normal MRP/DRP run.
- Validate in `MD04` / the APO product view that projected stock stays inside the corridor, and review `UNAVOIDABLE_SHORTAGE` rows before releasing anything to the live system.

## 7. Limitations
- Policies are solved as stationary per week's corridor (not a full 156-week backward induction); the grid and key rounding make the solution approximate.
- Cost rates and demand CV are defaults; calibrate them.
- Grouping-calendar days must be supplied in the params CSV (text-to-days conversion is not implemented).
- CH71 / `LOC_ID_USED` redirection from the R script is not ported.
- Demand is assumed independent across weeks.

## 8. Files
| File | Purpose |
|---|---|
| `inventory_corridor_vi.py` | Value Iteration ordering within the corridor (this document) |
| `inventory_corridor.py` | Corridor engine and shared driver; also the deterministic min/max variant |
| `config.json` | Paths and options |
| `README_inventory_corridor.md` | Longer notes on the deterministic variant |
