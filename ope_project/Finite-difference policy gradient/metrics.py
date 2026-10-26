"""Business metrics (pure functions)."""
import numpy as np

ANN = 252


def max_drawdown(net):
    eq = np.cumprod(1.0 + np.asarray(net, float))
    peak = np.maximum.accumulate(np.r_[1.0, eq])[1:]
    return float((eq / peak - 1.0).min())


def trading_metrics(net, pos, gross=None, cost_rate=5e-4, benchmark=None):
    """net: daily net returns, pos: daily position (starts flat), gross: daily gross returns."""
    net, pos = np.asarray(net, float), np.asarray(pos, float)
    n = len(net)
    cum = float(np.prod(1 + net) - 1)
    ann = float((1 + cum) ** (ANN / n) - 1) if cum > -1 else -1.0
    sd = net.std(ddof=1) if n > 1 else np.nan
    dd = np.sqrt(np.mean(np.minimum(net, 0) ** 2))
    mdd = max_drawdown(net)
    wins, losses = net[net > 0], net[net < 0]
    turn = float(np.abs(np.diff(np.r_[0.0, pos])).sum())
    g = net + cost_rate * np.abs(np.diff(np.r_[0.0, pos])) if gross is None else np.asarray(gross, float)
    out = {"cum_return": cum, "ann_return": ann, "ann_vol": float(sd * np.sqrt(ANN)),
           "sharpe": float(net.mean() / sd * np.sqrt(ANN)) if sd > 0 else np.nan,
           "sortino": float(net.mean() * ANN / (dd * np.sqrt(ANN))) if dd > 0 else np.nan,
           "max_drawdown": mdd, "calmar": float(ann / abs(mdd)) if mdd < 0 else np.nan,
           "hit_rate": float((net > 0).mean()),
           "profit_factor": float(wins.sum() / -losses.sum()) if len(losses) else np.nan,
           "avg_win": float(wins.mean()) if len(wins) else 0.0, "avg_loss": float(losses.mean()) if len(losses) else 0.0,
           "turnover": turn, "ann_turnover": turn * ANN / n, "cost_paid": float(cost_rate * turn),
           "cost_drag": float(g.sum() - net.sum()), "avg_position": float(pos.mean()), "exposure": float((pos > 0.1).mean())}
    if benchmark is not None:
        b = np.asarray(benchmark, float)
        out["excess_cum_vs_benchmark"] = cum - float(np.prod(1 + b) - 1)
        out["vol_ratio_vs_benchmark"] = float(sd / b.std(ddof=1))
        out["mdd_diff_vs_benchmark"] = mdd - max_drawdown(b)
    return out


def inventory_metrics(D, price_mean, days_per_year=365):
    """D: detail dict of (N, H) arrays from inventory_env.simulate(detail=True). Averages over windows."""
    N, H = D["sales"].shape
    rev, pur, hold, so, fx = (D[k].sum(axis=1) for k in ("revenue", "purchase", "holding", "stockout", "fixed"))
    profit = rev - pur - hold - so - fx
    sales, dem = D["sales"].sum(axis=1), D["demand"].sum(axis=1)
    avg_oh = D["on_hand"].mean(axis=1)
    safe = lambda a, b: np.where(b > 0, a / np.maximum(b, 1e-12), np.nan)
    stock_days = (D["lost"] > 0).mean(axis=1)
    return {"profit": float(profit.mean()), "revenue": float(rev.mean()),
            "purchase_cost": float(pur.mean()), "holding_cost": float(hold.mean()), "stockout_cost": float(so.mean()),
            "fixed_cost": float(fx.mean()), "purchase_share": float(np.nanmean(safe(pur, rev))),
            "holding_share": float(np.nanmean(safe(hold, rev))), "stockout_share": float(np.nanmean(safe(so, rev))),
            "fixed_share": float(np.nanmean(safe(fx, rev))), "fill_rate": float(sales.sum() / max(dem.sum(), 1e-12)),
            "stockout_rate": float(stock_days.mean()), "service_level": float(1 - stock_days.mean()),
            "lost_units": float(D["lost"].sum(axis=1).mean()), "lost_value": float(D["lost"].sum(axis=1).mean() * price_mean),
            "avg_on_hand": float(avg_oh.mean()), "inventory_turns": float(np.nanmean(safe(sales, avg_oh)) * days_per_year / H),
            "days_of_supply": float(avg_oh.mean() / max(dem.mean() / H, 1e-12)),
            "n_orders": float((D["order"] > 0).sum(axis=1).mean()), "order_freq": float((D["order"] > 0).mean())}
