"""Test-tuned ("oracle") base-stock and (s,S) benchmarks: an upper bound for the train-tuned benchmarks, NOT a fair competitor.
Writes outputs/business_eval/full/inventory_oracle.csv
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd
import policies as P
import run_fdpg as R
from audit_data import HERE, load_config


def profit(p, pol, th, split):
    D = p.detail(pol, np.asarray(th, float), split)
    return float((D["revenue"] - D["purchase"] - D["holding"] - D["stockout"] - D["fixed"]).sum(axis=1).mean())


def main():
    cfg = load_config()
    rows = []
    for sid in R.list_series(cfg, "inventory"):
        p = R.get_problem(cfg, "inventory", sid, "I1", 1.0)
        mu = p.extra["mu"]
        i1, i2 = P.I1(mu), P.I2(mu)
        g1 = [(profit(p, i1, [t], "test"), [t]) for t in np.linspace(0, 10, 101)]
        g2 = [(profit(p, i2, [a, b], "test"), [a, b]) for a in np.linspace(0, 8, 17) for b in np.linspace(-3, 3, 13)]
        for name, g in (("oracle_base_stock", g1), ("oracle_sS", g2)):
            v, th = max(g, key=lambda x: x[0])
            rows.append({"series": sid, "benchmark": name, "test_profit": v, "theta": th})
            print(sid, name, round(v, 1), th, flush=True)
    out = os.path.join(HERE, "outputs", "business_eval", "full", "inventory_oracle.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
