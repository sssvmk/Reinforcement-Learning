"""Monte Carlo risk assessment from OHLCV data.

Usage: python mc_risk.py data.csv [--horizon 10] [--sims 20000] [--demo]
CSV columns: Date, Open, High, Low, Close, Volume (case-insensitive).
"""
import argparse
import numpy as np
import pandas as pd


def load(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    return df.sort_values("date").reset_index(drop=True)


def demo_data(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    r = rng.standard_t(4, n) * 0.01 + 0.0004
    close = 100 * np.exp(np.cumsum(r))
    open_ = np.r_[100, close[:-1]] * np.exp(rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, n)))
    return pd.DataFrame({"date": pd.date_range("2022-01-01", periods=n, freq="B"),
                         "open": open_, "high": high, "low": low,
                         "close": close, "volume": rng.integers(1e5, 1e6, n)})


def garman_klass_vol(df):
    # Daily volatility estimator using OHLC
    hl = np.log(df.high / df.low) ** 2
    co = np.log(df.close / df.open) ** 2
    return float(np.sqrt((0.5 * hl - (2 * np.log(2) - 1) * co).mean()))


def simulate(df, horizon, sims, method, seed=42):
    rng = np.random.default_rng(seed)
    ret = np.log(df.close / df.close.shift()).dropna().to_numpy()
    s0 = df.close.iloc[-1]
    if method == "bootstrap":  # historical resampling keeps fat tails
        r = rng.choice(ret, size=(sims, horizon))
    else:  # GBM with Garman-Klass sigma
        mu, sigma = ret.mean(), garman_klass_vol(df)
        r = (mu - 0.5 * sigma**2) + sigma * rng.standard_normal((sims, horizon))
    paths = s0 * np.exp(np.cumsum(r, axis=1))
    return s0, paths


def metrics(s0, paths, alpha=0.05):
    final_ret = paths[:, -1] / s0 - 1
    var = -np.quantile(final_ret, alpha)
    cvar = -final_ret[final_ret <= -var].mean()
    peak = np.maximum.accumulate(np.c_[np.full(len(paths), s0), paths], axis=1)
    mdd = ((np.c_[np.full(len(paths), s0), paths] / peak) - 1).min(axis=1)
    return {
        f"VaR {int((1-alpha)*100)}%": var,
        f"CVaR {int((1-alpha)*100)}%": cvar,
        "P(loss)": float((final_ret < 0).mean()),
        "P(loss > 10%)": float((final_ret < -0.10).mean()),
        "Expected return": float(final_ret.mean()),
        "Mean max drawdown": float(mdd.mean()),
        "Worst max drawdown (5th pct)": float(np.quantile(mdd, 0.05)),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?")
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--sims", type=int, default=20000)
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    df = demo_data() if a.demo or not a.csv else load(a.csv)
    for m in ("gbm", "bootstrap"):
        s0, paths = simulate(df, a.horizon, a.sims, m)
        print(f"\n[{m}] last close={s0:.2f}, horizon={a.horizon}d, sims={a.sims}")
        for k, v in metrics(s0, paths).items():
            print(f"  {k:30s} {v:8.2%}")
