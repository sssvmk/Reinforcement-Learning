# DATA_NOTES (step 0, read-only inspection)

m5_dir: `C:\Users\username\OneDrive - XXYY\m5-forecasting-accuracy`

## File listing
```
calendar.csv  103,469 bytes
sales_train_evaluation.csv  121,736,518 bytes
sales_train_validation.csv  120,007,726 bytes
sample_submission.csv  5,228,786 bytes
sell_prices.csv  203,395,785 bytes
```
- found `sales_train_validation.csv` -> `C:\Users\username\OneDrive - XXYY\m5-forecasting-accuracy\sales_train_validation.csv`
- found `sales_train_evaluation.csv` -> `C:\Users\username\OneDrive - XXYY\m5-forecasting-accuracy\sales_train_evaluation.csv`
- found `calendar.csv` -> `C:\Users\username\OneDrive - XXYY\m5-forecasting-accuracy\calendar.csv`
- found `sell_prices.csv` -> `C:\Users\username\OneDrive - XXYY\m5-forecasting-accuracy\sell_prices.csv`
- `sample_submission.csv` ignored by design.

### calendar.csv
shape: (1969, 14)

dtypes:
```
date              str
wm_yr_wk        int64
weekday           str
wday            int64
month           int64
year            int64
d                 str
event_name_1      str
event_type_1      str
event_name_2      str
event_type_2      str
snap_CA         int64
snap_TX         int64
snap_WI         int64
```
null counts (non-zero only):
```
event_name_1    1807
event_type_1    1807
event_name_2    1964
event_type_2    1964
```
head (first 12 cols):
```
         date  wm_yr_wk   weekday  wday  month  year    d event_name_1 event_type_1 event_name_2 event_type_2  snap_CA
0  2011-01-29     11101  Saturday     1      1  2011  d_1          NaN          NaN          NaN          NaN        0
1  2011-01-30     11101    Sunday     2      1  2011  d_2          NaN          NaN          NaN          NaN        0
2  2011-01-31     11101    Monday     3      1  2011  d_3          NaN          NaN          NaN          NaN        0
```
date range: 2011-01-29 .. 2016-06-19; rows=1969

### sales_train_validation.csv (id columns only)
shape: (30490, 6)

dtypes:
```
id          str
item_id     str
dept_id     str
cat_id      str
store_id    str
state_id    str
```
null counts (non-zero only):
```
none
```
head (first 12 cols):
```
                              id        item_id    dept_id   cat_id store_id state_id
0  HOBBIES_1_001_CA_1_validation  HOBBIES_1_001  HOBBIES_1  HOBBIES     CA_1       CA
1  HOBBIES_1_002_CA_1_validation  HOBBIES_1_002  HOBBIES_1  HOBBIES     CA_1       CA
2  HOBBIES_1_003_CA_1_validation  HOBBIES_1_003  HOBBIES_1  HOBBIES     CA_1       CA
```
number of d_* columns: 1913; unique series ids: 30490
states=3 stores=10 cats=3 depts=7 items=3049
sales min=0 max=763; all non-negative: True

### sell_prices.csv
shape: (6841121, 4)

dtypes:
```
store_id      category
item_id       category
wm_yr_wk         int16
sell_price     float32
```
null counts (non-zero only):
```
none
```
head (first 12 cols):
```
  store_id        item_id  wm_yr_wk  sell_price
0     CA_1  HOBBIES_1_001     11325        9.58
1     CA_1  HOBBIES_1_001     11326        9.58
2     CA_1  HOBBIES_1_001     11327        8.26
```
Missing prices appear as ABSENT rows (not NaN): sell_price nulls = 0

## Series profiling (leading zeros trimmed before stats)

series whose first sale is after d_100: 15737 of 30490; median zero fraction 0.634; FOODS candidates for primary: 63
```
                                       id    cat state  first_sale_idx   mean  zero_frac     cv  wk_ratio
primary       FOODS_3_228_CA_1_validation  FOODS    CA               0  9.016      0.006  0.505     2.005
intermittent  FOODS_1_141_CA_1_validation  FOODS    CA               0  0.681      0.600  1.532     1.656
seasonal      FOODS_3_697_CA_2_validation  FOODS    CA               0  5.949      0.091  0.701     2.859
```

Selection rationale:
- **primary**: FOODS, post-launch mean 8-15 units/day, zero fraction < 5%, first sale within 30 days, median CV
  among such items -> mid-volume dense demand, so lost-sales MDP is informative and censoring is minor.
- **intermittent** (replication 1): zero fraction closest to 0.6 with mean > 0.3 -> sparse demand.
- **seasonal** (replication 2): largest max/min weekday-mean ratio among dense mid-volume series -> strong weekday effect.

selected ids: {'primary': 'FOODS_3_228_CA_1_validation', 'intermittent': 'FOODS_1_141_CA_1_validation', 'seasonal': 'FOODS_3_697_CA_2_validation'}

## Observed vs. documented
- observed 30490 series, 1913 d_* columns (sales_train_validation.csv), 1969 calendar rows. Documented: ~30,490 / 1,913 (val) or 1,941 (eval) / ~1,969.

## PyTorch
importable, version 2.14.1+cpu