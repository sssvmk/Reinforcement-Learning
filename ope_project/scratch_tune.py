import time, numpy as np, pandas as pd, sys
from common import *
from data_trading import load_all
from domains import TradingDomain
import engine
cfg = load_config(); data, _ = load_all(cfg); d = data[0]; folds = make_folds(d.N, cfg)
dom = TradingDomain(d, folds['primary'], cfg)
t = time.time(); pool = dom.build_pool(1, cache_dir="outputs/pools"); print('pool', len(pool), time.time() - t)
e = cfg["estimators"]
e["alpha_grid"] = [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
t = time.time(); chosen, table = engine.tune_hparams(dom, pool, cfg, 150, 0.3, 0.1, 5, 5); print('tune', time.time() - t)
for m, tb in table.items():
    print(m)
    for p, s in tb: print('   ', p, round(s, 4))
