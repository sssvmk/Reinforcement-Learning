"""Policy Evaluation (DP) of the corridor policy - on-hand trigger version.

Same CLI / inputs as policy_eval_inventory_v2.py (+ --matloc corridor master data).

Policy being evaluated (corridor.pptx logic, on-hand based, in-transit excluded)
  MIN_t, MAX_t : corridor on-hand band of week t (corridor_logic.py)
  trigger      : current ON-HAND x_t <= MIN_t                 (not the inventory position)
  order        : q = MAX_t + E[demand over lead time] - (x_t + z_t)   (z = in-transit)
                 q > 0 only; at least MOQ, rounded up to INV_MIN_RV
State (x, z)   : on-hand and total in-transit.  Pipeline approximation: each week 1/L of
                 the in-transit quantity lands (geometric lag, mean = L weeks).
Weekly step    : arrivals a = z / L ; D ~ Normal(mu_t, cv*mu_t), lost sales
                 x' = max(x + a - D, 0), z' = z - a + q
Cost           : holding h * x' + lost-sales p * max(D - x - a, 0); ordering cost 0
Backward induction (policy evaluation, finite horizon, V_T = 0):
                 V_t(x,z) = E[ cost + gamma * V_{t+1}(x', z') ]
--validate N   : Monte Carlo of the same policy with an exact deterministic-lead pipeline.
"""
import math
import sys
import numpy as np
import pandas as pd
from scipy.stats import norm

import policy_eval_inventory_v2 as v2
from corridor_logic import corridor_levels

K_DEMAND = 25


def _interp2(F, dx, dz, x, z):
    """Bilinear interpolation on a uniform grid; queries are clamped to the grid."""
    nx, nz = F.shape
    fx = np.clip(x / dx, 0, nx - 1 - 1e-9)
    fz = np.clip(z / dz, 0, nz - 1 - 1e-9)
    i, j = fx.astype(int), fz.astype(int)
    u, v = fx - i, fz - j
    return ((1 - u) * (1 - v) * F[i, j] + u * (1 - v) * F[i + 1, j]
            + (1 - u) * v * F[i, j + 1] + u * v * F[i + 1, j + 1])


def order_rule(x, z, mn, mx, lt_mu, moq, rounding):
    need = mx + lt_mu - (x + z)
    q = np.where((x <= mn) & (need > 0), np.maximum(need, moq), 0.0)
    if rounding > 1:
        q = np.ceil(q / rounding - 1e-9) * rounding
    return q


def policy_evaluation_oh(mu, cv, L, mn, mx, h, p, gamma, moq, rounding,
                         x_obs=0.0, z_obs=0.0, nx=100, nz=60):
    T, L = len(mu), max(int(L), 1)
    mu_x = np.concatenate([mu, np.full(L + 1, mu[-1])])
    cs = np.concatenate([[0], np.cumsum(mu_x)])
    lt_mu = np.array([cs[t + L] - cs[t] for t in range(T)])
    sd = cv * mu
    zmax = max((mx + lt_mu).max() + moq + rounding, z_obs, 1.0) * 1.05
    xmax = max(zmax + 4 * sd.max(), x_obs * 1.05, 1.0)
    dx, dz = xmax / (nx - 1), zmax / (nz - 1)
    X, Z = np.meshgrid(np.arange(nx) * dx, np.arange(nz) * dz, indexing="ij")
    q_pts = norm.ppf((np.arange(K_DEMAND) + 0.5) / K_DEMAND)

    V = np.zeros((nx, nz))
    out = [None] * T
    for t in range(T - 1, -1, -1):
        q = order_rule(X, Z, mn[t], mx[t], lt_mu[t], moq, rounding)
        a = Z / L
        D = np.maximum(mu[t] + sd[t] * q_pts, 0.0)[None, None, :]
        avail = (X + a)[..., None]
        x_next = np.maximum(avail - D, 0.0)
        z_next = np.broadcast_to((Z - a + q)[..., None], x_next.shape)
        h_cost = h * x_next
        p_cost = p * np.maximum(D - avail, 0.0)
        Vn = _interp2(V, dx, dz, x_next, z_next)
        stockout = (D > avail).mean(axis=2)
        Hh, Pp = h_cost.mean(axis=2), p_cost.mean(axis=2)
        V = Hh + Pp + gamma * Vn.mean(axis=2)
        out[t] = dict(V=V.copy(), H=Hh, P=Pp, SO=stockout, Q=q)
    return out, (dx, dz)


def simulate(mu, cv, L, mn, mx, h, p, gamma, moq, rounding, x0, z0, n, seed=0):
    """Exact pipeline (order placed at t lands at t+L; initial in-transit lands evenly)."""
    rng = np.random.default_rng(seed)
    T, L = len(mu), max(int(L), 1)
    mu_x = np.concatenate([mu, np.full(L + 1, mu[-1])])
    cs = np.concatenate([[0], np.cumsum(mu_x)])
    x = np.full(n, float(x0))
    pipe = np.zeros((n, L + T + 1))
    pipe[:, :L] = z0 / L
    total = np.zeros(n)
    for t in range(T):
        z = pipe[:, t:t + L].sum(axis=1)
        q = order_rule(x, z, mn[t], mx[t], cs[t + L] - cs[t], moq, rounding)
        pipe[:, t + L] += q
        avail = x + pipe[:, t]
        d = np.maximum(rng.normal(mu[t], cv * mu[t], n), 0)
        x = np.maximum(avail - d, 0)
        total += gamma ** t * (h * x + p * np.maximum(d - avail, 0))
    return total.mean(), total.std() / math.sqrt(n)


def simulate_dp_model(mu, cv, L, mn, mx, h, p, gamma, moq, rounding, x0, z0, n, seed=0):
    """Monte Carlo of exactly the DP's model (1/L of in-transit lands per week)."""
    rng = np.random.default_rng(seed)
    T, L = len(mu), max(int(L), 1)
    mu_x = np.concatenate([mu, np.full(L + 1, mu[-1])])
    cs = np.concatenate([[0], np.cumsum(mu_x)])
    x, z, total = np.full(n, float(x0)), np.full(n, float(z0)), np.zeros(n)
    for t in range(T):
        q = order_rule(x, z, mn[t], mx[t], cs[t + L] - cs[t], moq, rounding)
        arr = z / L
        avail = x + arr
        d = np.maximum(rng.normal(mu[t], cv * mu[t], n), 0)
        x, z = np.maximum(avail - d, 0), z - arr + q
        total += gamma ** t * (h * x + p * np.maximum(d - avail, 0))
    return total.mean(), total.std() / math.sqrt(n)


def run_key(cycle_df, cv, L, price, a, master=None, gov_unit=None):
    w = v2.weekly_series(cycle_df)
    mu = w["TOTAL_DEMAND"].clip(lower=0).to_numpy(float)
    if len(mu) == 0:
        return None
    moq, rounding = a.moq, a.rounding
    if master is not None:
        mn, mx, drivers, src = corridor_levels(
            master, mu, w["SAFETY_STK_QTY"], gov_unit=gov_unit or master.get("GOV_UNIT"),
            literal_410=a.literal_410)
        mq = pd.to_numeric(master.get("MOQ_QTY_USED"), errors="coerce")
        rv = pd.to_numeric(master.get("INV_MIN_RV_USED"), errors="coerce")
        moq = 0.0 if pd.isna(mq) else float(mq)
        rounding = float(rv) if pd.notna(rv) and rv > 1 else 1.0
        src = "slide logic: " + src
    else:
        mn = w["MIN_STK_QTY"].to_numpy(float)
        mx = np.where(w["MAX_STK_QTY"] > 0, w["MAX_STK_QTY"], mn + a.cover_weeks * mu)
        drivers, src = [""] * len(mu), "projection columns"
    mx = np.maximum(mx, mn)
    h, p = a.carry * price, a.shortage * price
    x = w["ON_HAND_STK_QTY"].clip(lower=0).to_numpy(float)
    z = w["IN_TRANS_STK_QTY"].to_numpy(float)
    res, (dx, dz) = policy_evaluation_oh(mu, cv, L, mn, mx, h, p, a.gamma, moq, rounding,
                                         x_obs=x.max(), z_obs=z.max())
    rows = []
    for t in range(len(w)):
        f = lambda k: float(_interp2(res[t][k], dx, dz, x[t:t + 1], z[t:t + 1])[0])
        q_now = float(order_rule(x[t], z[t], mn[t], mx[t],
                                 float(np.sum(np.r_[mu, np.full(L + 1, mu[-1])][t:t + max(L, 1)])),
                                 moq, rounding))
        rows.append(dict(
            YEARWEEK=w["YEARWEEK"].iloc[t], DEMAND_MEAN=mu[t], CV=cv, LEAD_WEEKS=L,
            CORRIDOR_MIN=mn[t], CORRIDOR_MAX=mx[t], CYCLE_DRIVER=drivers[t], CORRIDOR_SOURCE=src,
            UNIT_PRICE=price, STATE_ONHAND=x[t], STATE_IN_TRANS=z[t],
            TRIGGER_ONHAND_LE_MIN=bool(x[t] <= mn[t]), ORDER_QTY=q_now,
            EXP_HOLDING_COST=f("H"), EXP_LOST_SALES_COST=f("P"), STOCKOUT_PROB=f("SO"),
            VALUE_TO_GO=f("V")))
    out = pd.DataFrame(rows)
    if getattr(a, "validate", 0):
        args = (mu, cv, L, mn, mx, h, p, a.gamma, moq, rounding, x[0], z[0], a.validate)
        g, gse = simulate_dp_model(*args)
        e, ese = simulate(*args)
        print(f"validate {cycle_df['MAT_ID'].iloc[0]}/{cycle_df['LOC_ID'].iloc[0]}: "
              f"DP V0={out['VALUE_TO_GO'].iloc[0]:.4f} | MC same model={g:.4f}+/-{gse:.4f} "
              f"(numerics check) | MC exact pipeline={e:.4f}+/-{ese:.4f} (approximation gap)")
    return out


if __name__ == "__main__":
    if "--validate" in sys.argv:                 # --validate N -> handled by v2 parser below
        pass
    v2_parse = v2.argparse.ArgumentParser.parse_args

    def parse_args(self, *args, **kw):
        self.add_argument("--validate", type=int, default=0,
                          help="Monte Carlo paths to cross-check V0 (0 = off)")
        return v2_parse(self, *args, **kw)

    v2.argparse.ArgumentParser.parse_args = parse_args
    v2.run_key = run_key
    v2.main()
