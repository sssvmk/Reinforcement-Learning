"""Synthetic fixtures (no real data needed)."""
import numpy as np
import inventory_env as ie
import trading_env as te
import problems as pb


def make_inv(n=400, L=2, H=30, seed=0, qmax=None):
    rng = np.random.default_rng(seed)
    d = rng.poisson(8, n).astype(float)
    price = np.full(n, 5.0)
    wday = np.arange(n) % 7
    ev = (rng.random(n) < 0.05).astype(float)
    sn = (rng.random(n) < 0.3).astype(float)
    cfg = ie.InvCfg(L=L, unit_cost=3.0, holding=0.015, stockout=1.0, fixed=1.5, init_inv=8.0 * (L + 1))
    env = ie.InvData(d, price, wday, ev, sn, rng.normal(0, 0.05, n), 8.0, cfg, qmax=qmax)
    sp = {"train": (0, 240), "val": (240, 320), "test": (320, 400)}
    pr = pb.Problem("inventory", "syn", env, sp, H, np.ones(n, bool))
    pr.norm = 50.0
    return pr


def make_trade(n=600, H=40, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, 5))
    R = 0.001 + 0.01 * rng.normal(0, 1, n) + 0.004 * np.tanh(X[:, 0])      # feature 0 is weakly predictive
    env = te.TradeData(X, R, X.copy(), np.zeros(5), np.ones(5), 5e-4, np.arange(n))
    sp = {"train": (0, 360), "val": (360, 480), "test": (480, 600)}
    pr = pb.Problem("trading", "syn", env, sp, H, np.ones(n, bool))
    pr.norm = 0.05
    return pr
