# TECHM price-scale robustness

## Interpretation

The close/mid ratio drifts smoothly from 0.652 (2007) to 0.974 (2026) with year-to-year changes of 0.2%-5.4% and no single-year jump: this pattern is what a cumulative DIVIDEND adjustment (total-return series) looks like, not a split (a split would give a one-off step of 2x or more). So `close` is a dividend-adjusted series and its returns are the appropriate total-return series, while the 49% cumulative gap to the unadjusted OHLC scale explains why the level audit fails. Returns from adjusted close are therefore kept as the primary choice; the 'open' variant below is a sensitivity check (it excludes dividends, so it understates total return, and open prints are noisy: 101 days flagged).

Caveat: this is an inference from the data pattern; the data provider's adjustment method is not documented, so please confirm with the data owner.

`close` is split/dividend-adjusted while open/high/low/sar are not (DATA_NOTES.md). Adjusted close gives the economically correct return series (corporate actions removed); the 'open' variant takes returns from next-open to following-open on the UNADJUSTED scale (execution at the next open), replacing days whose open-to-open and close-to-close ratios differ by more than 8% in log terms (splits / large corporate actions) by the close-to-close return. It is a robustness check, not a replacement of the primary results.

## close / mid(high, low) by year (median)

| year | close_over_mid_median | ratio_change_vs_prev_year |
|---|---|---|
| 2007.0000 | 0.6522 | nan |
| 2008.0000 | 0.6564 | 0.0064 |
| 2009.0000 | 0.6682 | 0.0180 |
| 2010.0000 | 0.6695 | 0.0019 |
| 2011.0000 | 0.6736 | 0.0062 |
| 2012.0000 | 0.6778 | 0.0063 |
| 2013.0000 | 0.6800 | 0.0032 |
| 2014.0000 | 0.6860 | 0.0088 |
| 2015.0000 | 0.6924 | 0.0094 |
| 2016.0000 | 0.7032 | 0.0155 |
| 2017.0000 | 0.7200 | 0.0240 |
| 2018.0000 | 0.7381 | 0.0251 |
| 2019.0000 | 0.7532 | 0.0204 |
| 2020.0000 | 0.7768 | 0.0314 |
| 2021.0000 | 0.8188 | 0.0540 |
| 2022.0000 | 0.8512 | 0.0396 |
| 2023.0000 | 0.8884 | 0.0437 |
| 2024.0000 | 0.9220 | 0.0377 |
| 2025.0000 | 0.9500 | 0.0304 |
| 2026.0000 | 0.9741 | 0.0254 |

## Agreement of the two return series

| corr_open_vs_close_daily_returns | share_days_flagged_corporate_action | n_flagged | median_abs_log_gap_unflagged | ann_vol_close | ann_vol_open |
|---|---|---|---|---|---|
| 0.3027 | 0.0216 | 101.0000 | 0.0155 | 0.3586 | 0.3495 |

## Flagged days

| date | ret_close | ret_open |
|---|---|---|
| 2007-10-09 | 0.0926 | 0.0077 |
| 2007-10-11 | -0.0622 | 0.0308 |
| 2007-10-16 | -0.0235 | -0.1291 |
| 2007-10-17 | 0.0198 | 0.1094 |
| 2007-11-29 | 0.0411 | -0.0406 |
| 2007-12-07 | -0.0437 | 0.0885 |
| 2007-12-17 | 0.0483 | -0.0614 |
| 2007-12-19 | 0.0129 | -0.1410 |
| 2007-12-20 | 0.0042 | 0.2082 |
| 2008-01-23 | -0.0420 | 0.0411 |
| 2008-01-24 | 0.0590 | -0.0594 |
| 2008-01-31 | 0.0972 | -0.0486 |
| 2008-03-28 | -0.0258 | 0.0570 |
| 2008-04-08 | -0.0283 | 0.1126 |
| 2008-04-28 | 0.1090 | -0.0330 |
| 2008-04-29 | 0.0121 | 0.1145 |
| 2008-05-05 | 0.0254 | -0.1126 |
| 2008-05-06 | 0.0114 | 0.1140 |
| 2008-06-04 | 0.0748 | -0.0318 |
| 2008-06-05 | -0.0278 | 0.0653 |
| 2008-06-10 | 0.0341 | -0.0608 |
| 2008-07-10 | -0.0701 | 0.0222 |
| 2008-07-22 | 0.0126 | 0.1014 |
| 2008-10-07 | -0.0922 | -0.1748 |
| 2008-10-08 | -0.1475 | -0.0651 |
| 2008-10-10 | 0.0697 | -0.1107 |
| 2008-10-13 | 0.0274 | 0.1557 |
| 2008-10-16 | -0.0518 | 0.0643 |
| 2008-10-17 | 0.0699 | -0.1214 |
| 2008-10-22 | -0.1033 | -0.0049 |
| 2008-10-29 | 0.0633 | -0.0883 |
| 2008-10-31 | 0.0150 | 0.1069 |
| 2008-11-21 | -0.0556 | 0.0251 |
| 2008-12-08 | 0.0434 | -0.0557 |
| 2008-12-10 | -0.0311 | 0.0560 |
| 2008-12-11 | 0.0272 | -0.0530 |
| 2008-12-16 | -0.0342 | 0.0878 |
| 2009-01-06 | -0.1437 | 0.0714 |
| 2009-01-07 | -0.0085 | -0.1667 |
| 2009-01-27 | 0.0389 | -0.0518 |
| 2009-02-18 | 0.1166 | 0.0133 |
| 2009-02-19 | 0.0120 | -0.0879 |
| 2009-02-20 | 0.0059 | 0.2311 |
| 2009-04-09 | 0.1186 | 0.0002 |
| 2009-04-15 | -0.0478 | 0.0394 |
| 2009-04-22 | 0.0864 | -0.0286 |
| 2009-05-18 | -0.0688 | 0.0729 |
| 2009-05-19 | 0.0546 | -0.0903 |
| 2009-05-22 | 0.0159 | 0.1548 |
| 2009-05-25 | -0.0886 | -0.0066 |
| 2009-05-26 | 0.0942 | -0.0523 |
| 2009-05-27 | -0.0359 | 0.0491 |
| 2009-05-29 | 0.1039 | 0.0174 |
| 2009-06-02 | 0.1765 | 0.0781 |
| 2009-06-03 | 0.0157 | 0.1422 |
| 2009-06-04 | -0.0460 | 0.0642 |
| 2009-06-08 | 0.2561 | -0.0857 |
| 2009-06-09 | 0.0522 | 0.2803 |
| 2009-06-15 | 0.1288 | -0.0144 |
| 2009-06-16 | -0.0429 | 0.1450 |
| 2009-06-17 | 0.0374 | -0.0703 |
| 2009-06-18 | -0.0291 | 0.0617 |
| 2009-07-06 | 0.0157 | -0.0808 |
| 2009-07-09 | -0.0599 | 0.0235 |
| 2009-07-22 | 0.0183 | -0.0936 |
| 2009-07-23 | 0.0141 | 0.1167 |
| 2009-08-17 | 0.0432 | -0.0445 |
| 2009-11-09 | -0.0290 | 0.1085 |
| 2009-11-10 | 0.0547 | -0.0403 |
| 2009-11-11 | -0.0053 | 0.0790 |
| 2009-12-17 | -0.0139 | 0.1532 |
| 2009-12-18 | 0.0042 | -0.1207 |
| 2010-01-22 | -0.0778 | 0.0357 |
| 2010-01-27 | 0.0555 | -0.0784 |
| 2011-02-14 | 0.0017 | 0.0926 |
| 2011-03-07 | 0.0950 | 0.0005 |
| 2011-03-08 | -0.0122 | 0.1013 |
| 2011-06-09 | -0.0033 | 0.1434 |
| 2011-06-10 | 0.0024 | -0.1220 |
| 2011-09-02 | 0.0676 | -0.0670 |
| 2011-09-05 | -0.0250 | 0.0855 |
| 2011-10-05 | 0.0585 | -0.0261 |
| 2012-06-15 | -0.0224 | 0.1820 |
| 2012-06-18 | 0.0001 | -0.1416 |
| 2012-06-25 | -0.0028 | 0.1246 |
| 2012-06-26 | 0.0108 | -0.1108 |
| 2014-01-17 | 0.0301 | -0.0526 |
| 2015-12-14 | -0.0436 | 0.0398 |
| 2018-03-19 | 0.0385 | -0.0470 |
| 2018-06-27 | -0.0700 | 0.0348 |
| 2018-10-31 | -0.0312 | 0.0748 |
| 2020-03-12 | 0.0094 | -0.1076 |
| 2020-03-13 | -0.0801 | 0.0523 |
| 2020-03-19 | 0.0875 | -0.0447 |
| 2020-03-20 | -0.1547 | 0.0284 |
| 2020-03-23 | 0.0335 | -0.0981 |
| 2020-03-27 | 0.0450 | -0.0590 |
| 2020-03-31 | -0.0960 | 0.0428 |
| 2020-04-01 | 0.0190 | -0.0826 |
| 2020-10-28 | 0.0387 | -0.0482 |
| 2020-11-10 | 0.0197 | -0.0596 |

## Test-period cumulative return (contiguous replay, 5 bps cost, mean and sd over seeds)

| env_returns | policy_trained_on | strategy | n_seeds | test_cum_return | sd |
|---|---|---|---|---|---|
| close | - | buy_and_hold | 1 | 1.0000 | 0.0000 |
| close | train of env | RSI_rule(train-tuned) | 1 | 1.0000 | 0.0000 |
| open | - | buy_and_hold | 1 | 0.8039 | 0.0000 |
| open | train of env | RSI_rule(train-tuned) | 1 | 0.8039 | 0.0000 |
| close | close | R-T1(reinforce-loo_time) | 10 | 0.8131 | 0.2037 |
| open | close | R-T1(reinforce-loo_time) | 10 | 0.6900 | 0.2011 |
| close | close | R-T1(reinforce-none) | 10 | 0.7856 | 0.3273 |
| open | close | R-T1(reinforce-none) | 10 | 0.6609 | 0.3004 |
| close | close | R-T1(cm-central) | 10 | 0.7881 | 0.1540 |
| open | close | R-T1(cm-central) | 10 | 0.6500 | 0.1128 |
| close | close | R-T1(cm-spsa) | 10 | 0.6172 | 0.1996 |
| open | close | R-T1(cm-spsa) | 10 | 0.4991 | 0.1674 |
| close | close | R-T2(reinforce-loo_time) | 10 | 0.7559 | 0.1551 |
| open | close | R-T2(reinforce-loo_time) | 10 | 0.6307 | 0.1282 |
| close | close | R-T2(reinforce-none) | 10 | 0.6158 | 0.1538 |
| open | close | R-T2(reinforce-none) | 10 | 0.5154 | 0.1671 |
| close | open | R-T1(reinforce-loo_time) | 10 | 0.6166 | 0.3079 |
| open | open | R-T1(reinforce-loo_time) | 10 | 0.4441 | 0.2383 |
| close | open | R-T1(reinforce-none) | 10 | 0.6904 | 0.2526 |
| open | open | R-T1(reinforce-none) | 10 | 0.5058 | 0.2168 |
| close | open | R-T1(cm-central) | 10 | 0.5399 | 0.1173 |
| open | open | R-T1(cm-central) | 10 | 0.4368 | 0.0902 |
| close | open | R-T1(cm-spsa) | 10 | 0.5604 | 0.2130 |
| open | open | R-T1(cm-spsa) | 10 | 0.4441 | 0.1834 |
| close | open | R-T2(reinforce-loo_time) | 10 | 0.5070 | 0.1738 |
| open | open | R-T2(reinforce-loo_time) | 10 | 0.4037 | 0.1627 |
| close | open | R-T2(reinforce-none) | 10 | 0.4978 | 0.1806 |
| open | open | R-T2(reinforce-none) | 10 | 0.3791 | 0.1522 |

## Same policy family, trained and evaluated within its own return source (Mann-Whitney)

| policy | method | close_trained_on_close_env | open_trained_on_open_env | mannwhitney_p |
|---|---|---|---|---|
| T1 | reinforce-loo_time | 0.8131 | 0.4441 | 0.0058 |
| T1 | reinforce-none | 0.7856 | 0.5058 | 0.0539 |
| T1 | cm-central | 0.7881 | 0.4368 | 0.0004 |
| T1 | cm-spsa | 0.6172 | 0.4441 | 0.0640 |
| T2 | reinforce-loo_time | 0.7559 | 0.4037 | 0.0013 |
| T2 | reinforce-none | 0.6158 | 0.3791 | 0.0173 |
