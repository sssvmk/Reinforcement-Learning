"""TD(0) value estimation of trading states from OHLCV using Parabolic SAR and RSI.

State  s = (RSI bucket, PSAR trend, PSAR distance bucket, position)
Policy = fixed rule being evaluated: long when PSAR is bullish and RSI < rsi_exit, else flat.
Reward = position * next-bar log return - cost * |position change|
TD(0)  : V(s) <- V(s) + alpha * [ r + gamma * V(s') - V(s) ]

Usage:
  python td0_trading.py --demo
  python td0_trading.py --file prices.csv.gz [--epochs 20 --alpha 0.05 --gamma 0.97]
  python td0_trading.py --file prices.csv.gz --lam 0.9 [--view backward|forward]   # TD(lambda)
Input columns (case-insensitive): open, high, low, close, volume (optional); optional timestamp/date column.
"""
import argparse
import numpy as np
import pandas as pd


def load(path):
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    for c in ("timestamp", "date", "datetime", "time"):
        if c in df.columns:
            df = df.sort_values(c).reset_index(drop=True)
            break
    need = {"open", "high", "low", "close"}
    if not need <= set(df.columns):
        raise SystemExit(f"missing columns: {need - set(df.columns)}")
    return df


def demo(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0003, 0.01, n)
    r[1:] += 0.1 * r[:-1]  # mild momentum so the indicators carry signal
    c = 100 * np.exp(np.cumsum(r))
    o = np.r_[c[0], c[:-1]]
    spread = np.abs(rng.normal(0, 0.005, n)) * c
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) + spread,
                         "low": np.minimum(o, c) - spread, "close": c})


def rsi(close, n=14):
    d = np.diff(close, prepend=close[0])
    up, dn = np.clip(d, 0, None), np.clip(-d, 0, None)
    out = np.full(len(close), 50.0)
    if len(close) <= n:
        return out
    au, ad = up[1:n + 1].mean(), dn[1:n + 1].mean()
    for i in range(n, len(close)):
        if i > n:
            au = (au * (n - 1) + up[i]) / n
            ad = (ad * (n - 1) + dn[i]) / n
        out[i] = 100.0 if ad == 0 else 100 - 100 / (1 + au / ad)
    return out


def psar(high, low, step=0.02, max_step=0.2):
    """Wilder Parabolic SAR. Returns (sar, trend) with trend = +1 bullish, -1 bearish."""
    n = len(high)
    sar, trend = np.zeros(n), np.ones(n)
    bull, af = True, step
    ep = high[0]
    sar[0] = low[0]
    for i in range(1, n):
        s = sar[i - 1] + af * (ep - sar[i - 1])
        if bull:
            s = min(s, low[i - 1], low[i - 2] if i > 1 else low[i - 1])
            if low[i] < s:
                bull, s, ep, af = False, ep, low[i], step
            elif high[i] > ep:
                ep, af = high[i], min(af + step, max_step)
        else:
            s = max(s, high[i - 1], high[i - 2] if i > 1 else high[i - 1])
            if high[i] > s:
                bull, s, ep, af = True, ep, high[i], step
            elif low[i] < ep:
                ep, af = low[i], min(af + step, max_step)
        sar[i], trend[i] = s, 1 if bull else -1
    return sar, trend


def build(df, a):
    h, l, c = (df[k].to_numpy(float) for k in ("high", "low", "close"))
    r = rsi(c, a.rsi_n)
    sar, tr = psar(h, l, a.psar_step, a.psar_max)
    rb = np.digitize(r, [30, 40, 50, 60, 70])            # 6 RSI buckets
    dist = np.abs(c - sar) / c
    db = np.digitize(dist, np.quantile(dist, [1 / 3, 2 / 3]))  # 3 distance buckets
    pos = ((tr > 0) & (r < a.rsi_exit)).astype(int)    # policy under evaluation
    state = ((rb * 2 + (tr > 0)) * 3 + db) * 2 + pos
    ret = np.diff(np.log(c), append=np.nan)               # return of bar t -> t+1
    return state, pos, ret, r, tr


def td0(state, pos, ret, a, lo, hi):
    V = np.zeros(6 * 2 * 3 * 2)
    visits = np.zeros(len(V), int)
    rew = np.array([pos[t] * ret[t] - a.cost * abs(pos[t] - pos[t - 1] if t else 0)
                    for t in range(hi)])
    for ep in range(a.epochs):
        err = 0.0
        if a.view == "forward":
            # offline forward view: G_t = r_t + g[(1-lam) V(s_t+1) + lam G_t+1], then move V toward G_t
            G = np.zeros(hi)
            nxt = V[state[hi - 1]]
            for t in range(hi - 2, lo - 1, -1):
                nxt = rew[t] + a.gamma * ((1 - a.lam) * V[state[t + 1]] + a.lam * nxt)
                G[t] = nxt
            for t in range(lo, hi - 1):
                s = state[t]
                err += abs(G[t] - V[s])
                V[s] += a.alpha * (G[t] - V[s])
                visits[s] += 1
        else:
            # backward view: accumulating eligibility traces; lam=0 is exactly TD(0)
            e = np.zeros(len(V))
            for t in range(lo, hi - 1):
                s, s2 = state[t], state[t + 1]
                delta = rew[t] + a.gamma * V[s2] - V[s]
                e *= a.gamma * a.lam
                e[s] += 1.0
                V += a.alpha * delta * e
                visits[s] += 1
                err += abs(delta)
        print(f"epoch {ep + 1:>2}  mean |error| = {err / (hi - 1 - lo):.6f}")
    return V, visits


def realised(pos, ret, a, lo, hi):
    """Discounted realised return from each bar (Monte Carlo target) for validation."""
    G = np.zeros(hi)
    acc = 0.0
    for t in range(hi - 2, lo - 1, -1):
        rew = pos[t] * ret[t] - a.cost * abs(pos[t] - pos[t - 1] if t else 0)
        acc = rew + a.gamma * acc
        G[t] = acc
    return G


def label(s):
    pos, s = s % 2, s // 2
    db, s = s % 3, s // 3
    up, rb = s % 2, s // 2
    rs = ["<30", "30-40", "40-50", "50-60", "60-70", ">70"][rb]
    return f"RSI {rs:>5} | SAR {'bull' if up else 'bear'} | dist {['near', 'mid', 'far'][db]:<4} | {'LONG' if pos else 'FLAT'}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--gamma", type=float, default=0.97)
    p.add_argument("--lam", type=float, default=0.0, help="TD(lambda); 0 = TD(0), 1 = Monte Carlo-like")
    p.add_argument("--view", choices=["backward", "forward"], default="backward",
                   help="backward = eligibility traces (online); forward = lambda-return (offline)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--cost", type=float, default=0.0005, help="cost per unit position change")
    p.add_argument("--rsi-n", type=int, default=14)
    p.add_argument("--rsi-exit", type=float, default=70, help="go flat when RSI >= this")
    p.add_argument("--psar-step", type=float, default=0.02)
    p.add_argument("--psar-max", type=float, default=0.2)
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--out", default="td0_state_values.csv")
    a = p.parse_args()
    if not a.demo and not a.file:
        p.error("give --file or --demo")
    df = demo() if a.demo else load(a.file)

    state, pos, ret, r, tr = build(df, a)
    warm = max(a.rsi_n, 30)
    split = int(len(df) * a.train_frac)
    V, visits = td0(state, pos, ret, a, warm, split)

    # out-of-sample check: does V(s_t) track the realised discounted return?
    G = realised(pos, ret, a, split, len(df))
    idx = np.arange(split, len(df) - 2)
    pred, real = V[state[idx]], G[idx]
    corr = np.corrcoef(pred, real)[0, 1] if pred.std() > 0 and real.std() > 0 else float("nan")
    print(f"\nout-of-sample corr(V(s), realised discounted return) = {corr:.3f}")
    print(f"mean V (long states) = {V[state[idx][pos[idx] == 1]].mean():.5f}"
          if (pos[idx] == 1).any() else "no long states out-of-sample")

    tab = pd.DataFrame({"state": [label(s) for s in range(len(V))], "V": V, "visits": visits})
    tab = tab[tab.visits > 0].sort_values("V", ascending=False)
    tab.to_csv(a.out, index=False)
    print("\nTop / bottom states by learned value:")
    print(pd.concat([tab.head(5), tab.tail(5)]).to_string(index=False))

    cur = state[-1]
    print(f"\nCurrent state: {label(cur)}  RSI={r[-1]:.1f}  V={V[cur]:.5f}")
    print(f"Values saved to {a.out}")


if __name__ == "__main__":
    main()
