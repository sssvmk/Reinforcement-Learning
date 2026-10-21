# Policy Evaluation (DP) for Corridor-Based Inventory Management

Evaluates, with dynamic programming, the cost of the **inventory corridor policy**
(min / max levels from `corridor.pptx`) for every MAT_ID / LOC_ID / YEARWEEK / YEARMONTH
of the internal supply chain (SAP ECC + SAP APO data).

| File | Role |
|---|---|
| `policy_eval_inventory_v3.py` | Current DP (on-hand trigger, state = on-hand + in-transit) |
| `policy_eval_inventory_v2.py` | Data loading, weekly shaping, CLI (v3 reuses it); older inventory-position DP |
| `corridor_logic.py` | Min / max corridor per the slides and the R corridor script |
| `policy_eval_inventory.py` | v1 single-key prototype |

---

## 1. Policy Evaluation (DP) in brief

Reinforcement learning has two related problems:

- **Prediction (policy evaluation)** - given a *fixed* policy π, how good is it?
- **Control** - find the best policy.

Policy evaluation computes the **value function** V^π(s): the expected total
(discounted) cost when starting in state *s* and following π. It requires a known model of
the environment (transition probabilities and costs) - hence "DP".

Bellman expectation equation:

```
V^π(s) = E[ cost(s, a, D) + γ · V^π(s') ]      with a = π(s), s' = next state
```

Finite horizon (weeks of the projection) is solved by **backward induction**:

```
V_T(s) = 0
V_t(s) = E_D[ cost_t(s, π_t(s), D) + γ · V_{t+1}(s') ]      t = T-1 ... 0
```

No policy is improved here; the corridor policy is fixed and *evaluated*. The result tells
how much holding and lost-sales cost the corridor is expected to generate from the
current inventory state, and how likely stockouts are.

Terms: **state** = situation (stock on hand, in transit); **action** = order quantity;
**policy** = rule mapping state to action; **γ** = discount factor (0.99 default);
**V** = expected discounted future cost ("value to go").

---

## 2. Functional explanation (inventory)

### 2.1 Policy being evaluated (corridor, in-market only)
For week *t* the corridor gives on-hand levels **MIN_t** and **MAX_t** (section 3).

- **Trigger:** current *on-hand* ≤ MIN_t (in-transit is excluded, per the slides)
- **Order qty:** `MAX_t + expected lead-time demand - (on-hand + in-transit)`,
  only if positive, at least MOQ, rounded up to the rounding value (`INV_MIN_RV`)
- Otherwise no order. Ordering cost = 0 (internal supply chain).

### 2.2 MDP model
| Element | Definition |
|---|---|
| State | (x = on-hand, z = in-transit) |
| Action | order quantity q from the rule above |
| Demand | D ~ Normal(TOTAL_DEMAND_t, CV × TOTAL_DEMAND_t), 25-point quantile quadrature |
| Lead time | `APO_Transit_param` days ÷ 7, rounded up to weeks (L) |
| Pipeline | each week 1/L of in-transit lands (mean-lag approximation, see 5) |
| Transition | arrivals a = z/L; x' = max(x + a − D, 0) (lost sales); z' = z − a + q |
| Cost | holding = carry × UNIT_PRICE × x'; lost sales = shortage × UNIT_PRICE × max(D − x − a, 0) |
| Parameters | carry 0.01 %/week, lost-sales 0.1 % of price per unit, γ = 0.99, ordering cost 0 |

CV is estimated from how the same future week's demand changes across CYCLE_DATE
versions; if only one version exists, `--cv` (default 0.3) is used.

### 2.3 Data shaping
- Primary key of projections: MAT_ID, LOC_ID, YEARMONTH, YEARWEEK, CYCLE_DATE.
  CYCLE_DATE is a version (new on each Sunday / working day 8).
- One record per YEARWEEK is built (flows summed, stock columns from the last row), so
  Sunday and month-end rows do not double count.
- `--cycle latest` (default) evaluates the newest version; `--cycle all` every version.
- The DP runs per key over the weeks of the version; results are joined back to YEARMONTH.

---

## 3. Corridor logic (`corridor_logic.py`, from `corridor.pptx`)

**Safety stock selection**
1. Cycle stock driver = "NO DEMAND" → no corridor (min 0, max 0)
2. Planning-Book safety stock populated (SS > 0 at any point) → **dynamic SS**:
   Min = ½ SS; Max = SS + cycle stock + tolerance
3. Else static SS in master data → Min = ½ static SS; Max = static SS days + cycle stock + tolerance
4. Else Min = 0; Max = cycle stock + tolerance

Max days are capped at 410 (`--literal-410` uses the slide's literal `max(…, 410)`).
`STATIC_MIN_DAYS` / `STATIC_MAX_DAYS` are **ignored**.

**Static SS in days** by `SAFETY_STOCK_CALC_METHOD`:
SZ/MZ = `SAFETY_STK_DAYS`; SB/MB = `SAFETY_STK_QTY` ÷ daily demand;
SM/MM = max of both, or days + qty-days (Min: ½ days + qty-days) when the obligated
stock flag is `X`.

**Cycle stock days** by lot-size method (7 days/week, 30 days/month):
| Method | Cycle stock |
|---|---|
| P periodic | max(grouping-calendar days, MOQ days, period days); without calendar max(MOQ days, period days) |
| E lot-for-lot | max(grouping-calendar days, MOQ days); without calendar max(MOQ days, 7) |
| F fixed | static fixed lot quantity (`MOQ_QTY_USED`, = BSTFE for procurement type F) |

MOQ days = MOQ ÷ daily demand. Except MOQ-driven results, cycle days are rounded up to the
average rounding value in days (max(1, `INV_MIN_RV` ÷ daily demand)).

**Tolerance** on cycle days by GOV_UNIT: CX 20 %; RX/VX 50 % (max 45 days); others 0.

Daily demand = weekly `TOTAL_DEMAND` ÷ 7. Quantity = days × daily demand.

---

## 4. Inputs and SAP source columns

### 4.1 Projections (`--proj`, "B projections")
| Column | Use |
|---|---|
| MAT_ID, LOC_ID, YEARMONTH, YEARWEEK, CYCLE_DATE | key / version |
| TOTAL_DEMAND | weekly demand → mean of D |
| TOTAL_RECEIPTS | weekly receipts (informational) |
| ON_HAND_STK_QTY | state x (negative values floored at 0) |
| IN_TRANS_STK_QTY | state z |
| SAFETY_STK_QTY | Planning-Book (dynamic) safety stock |
| MIN_STK_QTY, MAX_STK_QTY | fallback corridor when `--matloc` is not given |

### 4.2 Lead time (`--tlane`)
`APO_Transit_param` (days); newest valid lane per MAT_ID/LOC_ID
(VALFR/VALTO, SPRKZ blank, NOTALLOWFLG ≠ T).

### 4.3 Price (`--cmd`)
`UNIT_PRICE` per MAT_ID/LOC_ID; `GOV_UNIT` used for the tolerance.

### 4.4 Corridor master (`--matloc`) - output of the corridor R script
Built in `Test_3.1.6_LOCATION_MATERIAL_MD_HIST.r` from these SAP sources:

| Master column | SAP ECC source | SAP APO source |
|---|---|---|
| MAT_ID, LOC_ID | MARC `MATNR`, `WERKS` | MATLOC / MATKEY |
| PROC_TYPE (not used in the DP) | MARC `BESKZ` | MATLOC `BESKZ` |
| LOT_SIZE_CALC_METHOD_USED (P/E/F) | MARC `DISLS` (FX→F, EX→E, period codes→P) | MATLOTSZ `SL_LSZ_RANGE` |
| PERIOD_TYPE_USED, PERIOD_AMOUNT_USED | DISLS first letter; T439A `PERAZ` | MATLOTSZ `PERVR`, `PER_AMOUNT` |
| MOQ_QTY_USED | MARC `BSTMI` | MATLOTSZ `BSTMI` (`BSTFE` when BESKZ = F) |
| INV_MIN_RV_USED (rounding) | MARC `BSTRF` | MATLOTSZ `BSTRF` |
| SAFETY_STK_QTY (static) | MARC `EISBE` | MATLOC `SAFTY` |
| SAFETY_STK_DAYS | MARC `SHZET` | MATLOTSZ `SVTTY`/240000 |
| SAFETY_STOCK_CALC_METHOD | derived from MARC `SHFLG`, SHZET, EISBE (SB / SZ / MM) | MATLOC `MSDP_SB_METHOD` |
| OBLIGATED_STOCK_FLAG | - | MATLOC `ZZ_OBSTK` |
| GROUPING_CALENDAR_USED / _DAYS_USED | - | ZGCAL, ZGCAL_ASSIGN (text converted to days) |
| MOQ/rounding unit conversion | MARM `UMREZ`, `UMREN` | MATLOTSZ `LSUOM` |
| LOC_ID_USED, SUPP_LOC_ID (not used in the DP) | - | TRPROD (supplier location), MATLOC `AT102` push type |
| STATIC_MIN_DAYS / STATIC_MAX_DAYS (ignored) | - | MATLOC `AT104`, `AT105` |

SAP non-source inputs: Projections, B_LOCATION_MD, matloc_corridor_1, work_day_calendar.

---

## 5. Output (`--out`)

One row per MAT_ID, LOC_ID, YEARMONTH, YEARWEEK, CYCLE_DATE:

| Column | Meaning |
|---|---|
| DEMAND_MEAN, CV, LEAD_WEEKS, UNIT_PRICE | model inputs |
| CORRIDOR_MIN, CORRIDOR_MAX, CYCLE_DRIVER, CORRIDOR_SOURCE | corridor result and driver (e.g. `P PERIOD_IN_DAYS`, `F FIXED_QTY`) |
| STATE_ONHAND, STATE_IN_TRANS | state at that week |
| TRIGGER_ONHAND_LE_MIN | on-hand ≤ MIN (reorder trigger) |
| ORDER_QTY | quantity the policy orders in that state |
| EXP_HOLDING_COST, EXP_LOST_SALES_COST | expected cost of the week |
| STOCKOUT_PROB | probability demand exceeds available stock that week |
| VALUE_TO_GO | V^π(state): expected discounted cost from this week to the end of the projection |

## 6. Usage

```
python policy_eval_inventory_v3.py --proj proj.csv --tlane tlane.csv --cmd c_md.csv ^
       --matloc matloc.csv --out out.csv --cycle latest --validate 4000
```
Options: `--carry` `--shortage` `--gamma` `--cv` `--moq` `--rounding` `--cycle latest|all`
`--literal-410` `--price-by-material` `--max-keys` `--validate N`.
Without `--matloc`, MIN/MAX come from the projection columns. Requires Python with numpy,
pandas, scipy.

## 7. Assumptions and limits
- Pipeline is approximated (1/L lands per week). On a synthetic test the DP matched a
  Monte Carlo of the same model (11.53 vs 11.45) but was ~25 % below a simulation with an
  exact lead-time pipeline (15.30), so values are optimistic when the trigger is on-hand.
- The slides describe the corridor as a monitoring band; the reorder rule
  (on-hand ≤ MIN, order up to MAX + lead-time demand) is a modelling choice.
- Normal demand, lost sales, ordering cost 0, carry 0.01 %/week (low), γ = 0.99.
- Procurement type, push deployment / supplier-location parameters and working-day
  calendar demand are not modelled yet.
- F cycle stock (fixed lot) is marked "to be determined" in the deck; the static lot
  quantity is used.
- The DP covers the weeks of one projection version (finite horizon, terminal value 0).
