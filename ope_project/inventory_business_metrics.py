"""Inventory business metrics, benchmarks, validation->test correlation and window-bootstrap CIs (primary series).
Run AFTER `run_experiment.py --full --domain inventory`.  -> outputs/business_metrics_inventory.md|csv"""
import copy, numpy as np, pandas as pd
from scipy.stats import spearmanr
from common import load_config, make_folds
import data_m5
from domains_inventory import InvDomain

cfg = load_config()
seed0 = cfg["experiment"]["seed"]
c2 = copy.deepcopy(cfg)
c2["splits"] = cfg["inventory"]["splits"]
c2["walk_forward"] = cfg["inventory"]["walk_forward"]
s = data_m5.load_inventory_series(cfg, "outputs")["primary"]
dom = InvDomain(s, make_folds(s.N, c2)["primary"], c2)
pool = dom.build_pool(1)
names = [a.name for a in pool]
lo_t, hi_t = dom.rng_of("test")


def deploy(act):
    """Replay a greedy policy over the whole TEST period from the initial stock (no exploration)."""
    t, on, pipe = np.array([lo_t]), np.array([dom.init_on]), np.zeros((1, dom.npipe))
    prof = sold = dem = 0.0
    stock_days = orders = stockout_days = 0
    inv = []
    for _ in range(hi_t - lo_t):
        a = int(act(dom.raw(t, on, pipe))[0])
        q = dom.Q[a]
        r, t, on_next, pipe, sales, lost = dom.vec_step(t, on, pipe, np.array([q]))
        prof += float(r[0]); sold += float(sales[0]); dem += float(sales[0] + lost[0])
        orders += int(q > 0); stockout_days += int(lost[0] > 1e-9)
        inv.append(float(on[0] + (q if dom.L == 0 else 0.0) - sales[0]))
        on = on_next
    n = hi_t - lo_t
    return dict(total_profit=prof, avg_daily_profit=prof / n, fill_rate=sold / max(dem, 1e-9),
                stockout_rate=stockout_days / n, service_level=1 - stockout_days / n, avg_inventory=float(np.mean(inv)),
                n_orders=orders)


rows = []
for a in pool:
    m = deploy(a.act)
    m["J_true_ope"] = float(np.mean(list(dom._day_values(a.act, 0.1, "ope").values())))
    m["agent"] = a.name
    rows.append(m)
df = pd.DataFrame(rows).set_index("agent")
df.to_csv("outputs/business_metrics_inventory.csv")

bs = df[df.index.str.startswith("base_stock")]
beh = deploy(dom._ensure_behavior().act)
rand = df[~df.index.isin(["never_order", "order_max"])]
learned = df.index.str.startswith(("dqn", "cql"))
bench = pd.DataFrame({
    "best_base_stock_by_validation_truth": df.loc[bs.J_true_ope.idxmax()],
    "best_base_stock_by_test_profit": df.loc[bs.total_profit.idxmax()],
    "oracle_best_by_test_profit": df.loc[df.total_profit.idxmax()],
    "best_by_validation_truth": df.loc[df.J_true_ope.idxmax()],
    "behavior_rule (eps=0)": pd.Series({**beh, "J_true_ope": np.nan}),
    "never_order": df.loc["never_order"], "order_max": df.loc["order_max"],
    "pool_median": df.median(), "learned_agents_median": df[learned].median()}).T
bench.insert(0, "agent", [bs.J_true_ope.idxmax(), bs.total_profit.idxmax(), df.total_profit.idxmax(), df.J_true_ope.idxmax(),
                          "base-stock tuned on train", "-", "-", "-", "-"])

cor = {m: float(spearmanr(df.J_true_ope, df[m])[0]) for m in ("total_profit", "fill_rate", "service_level", "avg_inventory")}

# ---- window bootstrap of the ground truth for the estimator rankings (base condition, primary series)
ar = pd.read_csv("outputs/results/full_inv_agent_rows.csv")
ar = ar[(ar.cond == "base") & (ar.fold == "primary")]
B = 500
rng = np.random.default_rng(0)
draws = {}
for r in sorted(ar.rep.unique()):
    rr = np.random.default_rng(seed0 + 1000 * int(r) + 17)
    log = dom.sample_log("ope", cfg["experiment"]["ope_windows"], 0.3, rr)
    Jw, _ = dom.truth(pool, log, 0.1, "ope", np.array([[0, 0]]))
    for m, g in ar[ar.rep == r].groupby("method"):
        est = g.set_index("agent").loc[names, "J_final"].to_numpy(float)
        ok = np.isfinite(est)
        if ok.sum() < 3:
            continue
        for _ in range(B):
            idx = rng.integers(0, Jw.shape[1], Jw.shape[1])
            Jb = Jw[:, idx].mean(axis=1)
            sp = spearmanr(est[ok], Jb[ok])[0]
            reg = (Jb.max() - Jb[ok][np.argmax(est[ok])]) / (Jb.max() - Jb.min())
            draws.setdefault(m, []).append((sp, reg))
ci = []
for m, v in draws.items():
    v = np.array(v)
    ci.append(dict(method=m, spearman_median=np.nanmedian(v[:, 0]), spearman_lo95=np.nanpercentile(v[:, 0], 2.5),
                   spearman_hi95=np.nanpercentile(v[:, 0], 97.5), nregret1_median=np.nanmedian(v[:, 1]),
                   nregret1_lo95=np.nanpercentile(v[:, 1], 2.5), nregret1_hi95=np.nanpercentile(v[:, 1], 97.5)))
ci = pd.DataFrame(ci).set_index("method")

pk = pd.read_csv("outputs/results/full_inv_rep_metrics.csv")
pk = pk[(pk.cond == "base") & (pk.fold == "primary")]
picks = pk.groupby("method")[["pick_J_true", "pick_test_cum"]].median().round(3)
picks["most_frequent_pick"] = pk.groupby("method").pick.agg(lambda x: x.value_counts().index[0])
# test profit of each estimator's modal pick, in profit units
picks["pick_test_total_profit"] = [df.total_profit.get(p, np.nan) for p in picks.most_frequent_pick]

out = ["# Inventory business-metric evaluation (auto-generated by inventory_business_metrics.py)\n",
       f"Primary series {s.sid}, TEST period {lo_t}-{hi_t} ({hi_t - lo_t} days), lead time {dom.L}, greedy policies from the initial stock "
       "(on-hand = ceil(2u), empty pipeline). Profit is in the M5 price currency (USD) for this ONE item; costs are my assumptions "
       "(unit cost 0.6 x price, holding 0.5% of unit cost, stockout penalty 0.2 x price, fixed order cost 0.5 x unit cost). "
       "fill_rate = units sold / units demanded; stockout_rate = share of days with lost sales; service_level = 1 - stockout_rate; "
       "avg_inventory = mean end-of-day on-hand units. J_true_ope = Monte Carlo value (eps_t=0.1, 24 rollouts per start day).\n",
       "## Per-agent test performance\n", df.round(3).to_markdown(), "\n",
       "## Benchmarks\n", bench.round(3).to_markdown(), "\n",
       "## Does the OPE-period truth predict test performance? (Spearman across the pool)\n",
       "\n".join(f"- J_true_ope vs test {k}: {v:.2f}" for k, v in cor.items()), "\n",
       f"## Window-bootstrap 95% CIs (resampling the {cfg['experiment']['ope_windows']} OPE windows of the ground truth; "
       f"{B} draws per repetition, pooled over {ar.rep.nunique()} repetitions)\n",
       "Each estimator's final value estimates are held fixed; only the ground truth is resampled over windows, so these CIs "
       "capture uncertainty in the truth and ranking metric, NOT the estimators' own re-estimation noise (that is the spread across repetitions).\n",
       ci.round(3).to_markdown(), "\n",
       "## What each estimator's most frequent pick earned on the test period (primary, base)\n", picks.to_markdown(), "\n"]
open("outputs/business_metrics_inventory.md", "w", encoding="utf8").write("\n".join(out))
print("\n".join(out[3:8]))
print(cor)
print(ci.round(3).to_string())
