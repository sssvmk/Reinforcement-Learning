"""On-policy Monte Carlo control (first-visit, epsilon-greedy with GLIE) for daily OHLCV trading with RSI + PSAR.

Episode : fixed window of --horizon trading days starting flat at a random day in the training period.
State   : (RSI bucket, PSAR trend, PSAR distance bucket, current position) = 6 x 2 x 3 x 2 = 72
Action  : 0 = be flat, 1 = be long for the next bar
Reward  : action * next-day log return - cost * |action - position|
Update  : Q(s,a) <- Q(s,a) + (1/N(s,a)) * (G - Q(s,a)),   epsilon_k = 1/k (GLIE)

Usage:
  python mc_control_trading.py --demo
  python mc_control_trading.py --file sample_ohlcv.csv [--episodes 30000 --horizon 20 --train-frac 0.7]
Input columns (case-insensitive): open, high, low, close; optional date/timestamp column.
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
    r[1:] += 0.1 * r[:-1]
    c = 100 * np.exp(np.cumsum(r))
    o = np.r_[c[0], c[:-1]]
    sp = np.abs(rng.normal(0, 0.005, n)) * c
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) + sp, "low": np.minimum(o, c) - sp, "close": c})


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
    n = len(high)
    sar, trend = np.zeros(n), np.ones(n)
    bull, af, ep = True, step, high[0]
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


def features(df, a, split):
    h, l, c = (df[k].to_numpy(float) for k in ("high", "low", "close"))
    r = rsi(c, a.rsi_n)
    sar, tr = psar(h, l)
    dist = np.abs(c - sar) / c
    cuts = np.quantile(dist[:split], [1 / 3, 2 / 3])  # train-only cut points: no look-ahead
    base = (np.digitize(r, [30, 40, 50, 60, 70]) * 2 + (tr > 0)) * 3 + np.digitize(dist, cuts)
    ret = np.diff(np.log(c), append=np.nan)
    return base.astype(int), ret, r


def sid(base, t, pos):
    return base[t] * 2 + pos


def greedy(Q, s):
    return int(Q[s, 1] > Q[s, 0])


def label(s):
    pos, s = s % 2, s // 2
    db, s = s % 3, s // 3
    up, rb = s % 2, s // 2
    rs = ["<30", "30-40", "40-50", "50-60", "60-70", ">70"][rb]
    return f"RSI {rs:>5} | SAR {'bull' if up else 'bear'} | dist {['near', 'mid', 'far'][db]:<4} | pos {'LONG' if pos else 'FLAT'}"


def train(base, ret, a, lo, hi, rng):
    nS = 72
    Q = np.zeros((nS, 2))
    N = np.zeros((nS, 2))
    H = a.horizon
    for k in range(1, a.episodes + 1):
        eps = 1.0 / k if a.glie else a.epsilon          # GLIE: eps_k = 1/k
        eps = max(eps, a.min_eps)
        t0 = rng.integers(lo, hi - H - 1)
        pos, traj = 0, []
        for t in range(t0, t0 + H):
            s = sid(base, t, pos)
            act = rng.integers(2) if rng.random() < eps else greedy(Q, s)
            rew = act * ret[t] - a.cost * abs(act - pos)
            traj.append((s, act, rew))
            pos = act
        G, rets = 0.0, []
        for s, act, rew in reversed(traj):
            G = rew + a.gamma * G
            rets.append((s, act, G))
        seen = set()
        for s, act, G in reversed(rets):                # first-visit
            if (s, act) in seen:
                continue
            seen.add((s, act))
            N[s, act] += 1
            Q[s, act] += (G - Q[s, act]) / N[s, act]
    return Q, N


def backtest(Q, base, ret, a, lo, hi):
    pos, pnl = 0, []
    for t in range(lo, hi - 1):
        act = greedy(Q, sid(base, t, pos))
        pnl.append(act * ret[t] - a.cost * abs(act - pos))
        pos = act
    return np.array(pnl), np.array(ret[lo:hi - 1])


def stats(x):
    tot = float(np.exp(x.sum()) - 1)
    sh = float(x.mean() / x.std() * np.sqrt(252)) if x.std() > 0 else float("nan")
    cum = np.cumsum(x)
    dd = float((np.exp(cum) / np.exp(np.maximum.accumulate(cum)) - 1).min())
    return tot, sh, dd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--episodes", type=int, default=30000)
    p.add_argument("--horizon", type=int, default=20, help="episode length in trading days")
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--cost", type=float, default=0.0005)
    p.add_argument("--glie", action=argparse.BooleanOptionalAction, default=True, help="eps_k=1/k; --no-glie uses --epsilon")
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--min-eps", type=float, default=0.0, help="floor on epsilon (0 = pure GLIE)")
    p.add_argument("--rsi-n", type=int, default=14)
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="mc_q_values.csv")
    a = p.parse_args()
    if not a.demo and not a.file:
        p.error("give --file or --demo")
    df = demo() if a.demo else load(a.file)
    rng = np.random.default_rng(a.seed)

    split = int(len(df) * a.train_frac)
    warm = max(a.rsi_n, 30)
    base, ret, r = features(df, a, split)
    Q, N = train(base, ret, a, warm, split, rng)

    for name, lo, hi in (("train", warm, split), ("test ", split, len(df))):
        pnl, bh = backtest(Q, base, ret, a, lo, hi)
        s, b = stats(pnl), stats(bh)
        print(f"{name}  strategy: ret {s[0]:+.1%} sharpe {s[1]:+.2f} maxDD {s[2]:.1%}"
              f"   | buy&hold: ret {b[0]:+.1%} sharpe {b[1]:+.2f} maxDD {b[2]:.1%}")

    tab = pd.DataFrame({"state": [label(s) for s in range(72)], "Q_flat": Q[:, 0], "Q_long": Q[:, 1],
                        "N_flat": N[:, 0].astype(int), "N_long": N[:, 1].astype(int)})
    tab["best_action"] = np.where(tab.Q_long > tab.Q_flat, "LONG", "FLAT")
    tab = tab[(tab.N_flat + tab.N_long) > 0]
    tab.to_csv(a.out, index=False)
    cur = sid(base, len(df) - 1, 0)
    print(f"\nLatest bar (if flat): {label(cur)}  RSI={r[-1]:.1f}  ->  {'LONG' if greedy(Q, cur) else 'FLAT'}")
    print(f"Q table saved to {a.out}")


if __name__ == "__main__":
    main()
