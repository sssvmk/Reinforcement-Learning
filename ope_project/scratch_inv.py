import time, copy, numpy as np
from common import *
import data_m5, engine
from domains_inventory import InvDomain
cfg = load_config()
ser = data_m5.load_inventory_series(cfg, "outputs")
c2 = copy.deepcopy(cfg); c2["splits"] = cfg["inventory"]["splits"]
s = ser["primary"]; folds = make_folds(s.N, c2)
f = folds["primary"]; print(f)
dom = InvDomain(s, f, c2)
print("mean_d", dom.mean_d, "Q", dom.Q, "dim", dom.dim, "scale", dom.reward_scale)
t = time.time(); pool = dom.build_pool(1); print("pool", len(pool), time.time() - t, "behavior S", dom.behavior_S_days)
rng = np.random.default_rng(0)
t = time.time(); log = dom.sample_log("ope", 300, 0.3, rng); print("log", time.time() - t, log.raw.shape)
sidx = np.stack([rng.integers(0, log.W, 500), rng.integers(0, dom.state_step_max, 500)], 1)
t = time.time(); Jw, vs = dom.truth(pool, log, 0.1, "ope", sidx); print("truth", time.time() - t)
J = Jw.mean(1)
for a, j in zip(pool, J): print(f"  {a.name:18s} {j * dom.reward_scale:8.3f}  test {dom.test_perf(a)}")
print("behavior", dom.behavior_value(log, 0.3, "ope") * dom.reward_scale)
