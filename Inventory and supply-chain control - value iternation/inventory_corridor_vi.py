"""Value Iteration inventory control constrained by the dynamic SAP APO corridor (min/max).

Per CYCLE_DATE x MAT_ID x LOC_ID x YEARMONTH x YEARWEEK.
Corridor (SS + cycle stock + tolerance, by PROC_TYPE / lot-size method / days of cover) comes from
inventory_corridor.corridor (port of Test_3.1.6_LOCATION_MATERIAL_MD_HIST.r and corridor.pptx).

MDP (solved by Value Iteration, in DAYS-OF-COVER units so material-locations with the same normalised
corridor share one solved policy -> millions of material-locations need only a few thousand solves)
  state   : net inventory position (grid step u days of demand; backlog allowed)
  action  : do not order, or order up to a level inside the corridor [MIN, MAX]
  demand  : weekly Normal(r days, cv*r); lead time L weeks handled by costing the position at t+L
  cost    : holding + backlog + below-safety-stock + below-MIN + above-MAX + fixed order cost
  solve   : V(i) = min_j [ F*(j>i) + c(j) + gamma * sum_k p_k V(j-k) ]  iterated to convergence
Each week the plan applies the policy of that week's corridor; quantities are converted back to units with the
daily demand and rounded to MOQ / fixed lot / rounding value (PROC_TYPE F uses BSTFE).
Options (config.json "options"): cv, hold_rate, backlog_rate, ss_rate, min_rate, over_rate, fixed_cost_days,
gamma, vi_tol, vi_max_iter, max_states.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

import inventory_corridor as ic

GH_X, GH_W = np.polynomial.hermite_e.hermegauss(21)
GH_W = GH_W / GH_W.sum()
CACHE = {}  # key -> (lo_n, N, policy array)
OPT = dict(cv=0.3, hold_rate=0.25, backlog_rate=0.10, ss_rate=0.03, min_rate=0.03, over_rate=2.0,
           fixed_cost_days=0.0, gamma=0.98, vi_tol=1e-4, vi_max_iter=600, max_states=80)


def pack(M, mn, ss, r2, L, f10):
    k = (((M.astype(np.int64) * 1024 + mn) * 1024 + ss) * 1024 + r2) * 64 + L
    return k * 1024 + f10


def unpack(k):
    k, f10 = divmod(int(k), 1024)
    k, L = divmod(k, 64)
    k, r2 = divmod(k, 1024)
    k, ss = divmod(k, 1024)
    M, mn = divmod(k, 1024)
    return M, mn, ss, r2, L, f10


def solve_key(key, o):
    """Value Iteration for one normalised corridor. Returns (lo_n, N, target index per state)."""
    M, mn, ss, r2, L, f10 = unpack(key)
    r = max(r2 / 2.0, 0.5)
    lo_n = int(min(40, np.ceil(1.5 * r * (L + 1)) + 2))
    N = lo_n + M + int(np.ceil(r)) + 2
    xs = np.arange(N) - lo_n
    h, b = o["hold_rate"] / 52, o["backlog_rate"]
    # cost of the position at arrival (demand over L+1 weeks)
    m, sd = (L + 1) * r, max(o["cv"] * r * np.sqrt(L + 1), 1e-6)
    ni = xs[:, None] - np.maximum(m + sd * GH_X, 0)[None, :]
    c = (h * np.maximum(ni, 0) + b * np.maximum(-ni, 0) + o["ss_rate"] * np.maximum(ss - ni, 0)
         + o["min_rate"] * np.maximum(mn - ni, 0) + o["over_rate"] * h * np.maximum(ni - M, 0)) @ GH_W
    # one-week demand transition (integer steps of u)
    K = int(np.ceil(r + 4 * max(o["cv"] * r, 0.3))) + 2
    sd1 = max(o["cv"] * r, 0.3)
    pk = np.diff(norm.cdf((np.arange(K + 1) - 0.5 - r) / sd1))
    pk[0] += norm.cdf((-0.5 - r) / sd1)
    pk /= pk.sum()
    idx = np.arange(N)
    P = np.zeros((N, N))
    for k, p in enumerate(pk):
        P[idx, np.clip(idx - k, 0, N - 1)] += p
    jj, ii = np.meshgrid(idx, idx)  # [i, j]
    allowed = (jj == ii) | ((jj > ii) & (jj >= lo_n + min(mn, M)) & (jj <= lo_n + M))
    F = np.where(jj > ii, f10 / 10.0, 0.0)
    V = np.zeros(N)
    for _ in range(o["vi_max_iter"]):
        W = c + o["gamma"] * (P @ V)
        Q = np.where(allowed, F + W[None, :], np.inf)
        Vn = Q.min(axis=1)
        done = np.abs(Vn - V).max() < o["vi_tol"]
        V = Vn
        if done:
            break
    return lo_n, N, Q.argmin(axis=1).astype(np.int16)


def simulate_vi(df, c, a):
    o = {**OPT, **{k: v for k, v in a.opts.items() if k in OPT}}
    gid = df.groupby(ic.KEY, sort=False).ngroup().to_numpy()
    pos = df.groupby(ic.KEY, sort=False).cumcount().to_numpy()
    G, T = gid.max() + 1, pos.max() + 1

    def mat(x, fill=np.nan):
        m = np.full((G, T), fill)
        m[gid, pos] = x
        return m

    row0 = df.groupby(ic.KEY, sort=False).head(1)
    days = np.where(row0.PROC_TYPE.to_numpy() == "E", row0.IN_HOUSE_LT_DAYS.to_numpy(float),
                    row0.LEAD_TIME_DAYS.fillna(a.default_lt_days).to_numpy(float))
    L = np.ceil(days / 7).astype(int)
    moq = c._MOQ.groupby(gid).first().to_numpy()
    fixed = np.where(row0.PROC_TYPE.to_numpy() == "F", row0.BSTFE.to_numpy(float), 0.0)
    rv = np.maximum(row0.INV_MIN_RV.to_numpy(float), 1)

    daily = c.DAILY_DEMAND.to_numpy()
    ok = (daily > 0) & (c.CORRIDOR_MAX_DAYS.to_numpy() > 0)
    safe = np.where(ok, daily, 1.0)
    rdays = np.clip(df.TOTAL_DEMAND.fillna(0).to_numpy(float) / safe, 0.5, 14)
    Md, mnd, ssd = c.CORRIDOR_MAX_DAYS.to_numpy(), c.CORRIDOR_MIN_DAYS.to_numpy(), c.CORRIDOR_SS_DAYS.to_numpy()
    u = np.maximum(np.maximum(rdays / 2, Md / o["max_states"]), 0.25)
    u = np.round(u * 4) / 4
    Mu = np.clip(np.round(Md / u), 1, 1000)
    mnu = np.clip(np.round(np.minimum(mnd, Md) / u), 0, Mu)
    ssu = np.clip(np.round(ssd / u), 0, 1000)
    r2 = np.clip(np.round(2 * rdays / u), 1, 1000)
    f10 = int(round(10 * o["fixed_cost_days"]))
    keys = pack(Mu, mnu.astype(np.int64), ssu.astype(np.int64), r2.astype(np.int64), np.minimum(L[gid], 63), f10)
    keys = np.where(ok, keys, -1)

    uk, inv = np.unique(keys[ok], return_inverse=True)
    for k in uk:
        if k not in CACHE:
            CACHE[k] = solve_key(k, o)
    Nmax = max((CACHE[k][1] for k in uk), default=1)
    tab = np.zeros((len(uk), Nmax), np.int16)
    lo = np.zeros(len(uk), int)
    nn = np.ones(len(uk), int)
    for n, k in enumerate(uk):
        lo[n], nn[n] = CACHE[k][0], CACHE[k][1]
        tab[n, :nn[n]] = CACHE[k][2]
    kid_row = np.full(len(df), -1)
    kid_row[ok] = inv
    kid, um = mat(kid_row, -1).astype(int), mat(u)
    base, mn, daily_m = mat(df.ON_HAND_STK_QTY.to_numpy(float)), mat(c.CORRIDOR_MIN_QTY.to_numpy()), mat(daily)
    mxq = np.nan_to_num(mat(c.CORRIDOR_MAX_QTY.to_numpy()))
    valid = ~np.isnan(base) & (kid >= 0)

    order, receipt, after = np.zeros((G, T)), np.zeros((G, T)), np.full((G, T), np.nan)
    short, tgt_out, cum = np.zeros((G, T), bool), np.full((G, T), np.nan), np.zeros(G)
    for s in range(T):
        stock = np.nan_to_num(base[:, s]) + cum
        v = valid[:, s]
        low = v & (stock < mn[:, s])
        short[:, s] = low & (s < L)
        kk = np.where(v, kid[:, s], 0)
        step = np.where(v, um[:, s] * daily_m[:, s], 1.0)
        i = np.clip(np.round(stock / step).astype(int) + lo[kk], 0, nn[kk] - 1)
        j = tab[kk, i].astype(int)
        tgt = np.minimum((j - lo[kk]) * step, mxq[:, s])
        need = v & (s >= L) & (j > i)
        q = np.maximum(tgt - stock, 0)
        q = np.where(fixed > 0, np.ceil(q / np.where(fixed > 0, fixed, 1)) * fixed, np.maximum(q, moq))
        q = np.ceil(q / rv) * rv
        q = np.where(need & (q > 0), q, 0.0)
        receipt[:, s] = q
        tgt_out[:, s] = np.where(v, tgt, np.nan)
        idx = np.nonzero(q)[0]
        order[idx, s - L[idx]] += q[idx]
        cum += q
        after[:, s] = np.nan_to_num(base[:, s]) + cum
    out = pd.DataFrame(dict(
        PROJ_STOCK_BEFORE=df.ON_HAND_STK_QTY.to_numpy(float),
        ORDER_QTY=order[gid, pos], PLANNED_RECEIPT_NEW=receipt[gid, pos],
        PROJ_STOCK_AFTER=after[gid, pos], VI_ORDER_UP_TO_QTY=tgt_out[gid, pos],
        UNAVOIDABLE_SHORTAGE=short[gid, pos], LEAD_TIME_WEEKS=L[gid]), index=df.index)
    out["ORDER_VALUE"] = out.ORDER_QTY * df.UNIT_PRICE.to_numpy(float)
    out["DAYS_COVER_AFTER"] = np.where(c.DAILY_DEMAND > 0, out.PROJ_STOCK_AFTER / c.DAILY_DEMAND.replace(0, np.nan), np.nan)
    return out


if __name__ == "__main__":
    ic.main(sim=simulate_vi)
    print(f"Value Iteration policies solved: {len(CACHE)}")
