"""Walk-forward comparison on FDPG's fold structure (3 expanding folds over the test region; each fold retrains on all prior data,
last 25% of it for validation), same seeds / budget for every method: REINFORCE R-T1, R-T2, R-T3 (neural) and the equal-budget
comparison methods on the R-T1 mean policy. FDPG's own walk-forward CSVs are READ (text only) and shown next to ours.

python walkforward_compare.py --quick|--full      (writes outputs/wf_<mode>/)
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

from common import load_config, resolve_fdpg_dir, outdir, md
import policies as P
import setup_problems as sp
import problems as pb
import trading_env as te
import data_trading as dt
import stats as st
from reinforce.algorithm import reinforce
from reinforce import comparison_methods as cm

FOLDS = 3
MODES = {"quick": dict(seeds=2, episodes=3000), "full": dict(seeds=5, episodes=20000)}
METHODS = [("rf", "T1", None), ("rf", "T2", None), ("rf", "T3", None), ("cm", "T1", "central"), ("cm", "T1", "spsa"), ("cm", "T1", "random")]
LABEL = {("rf", "T1", None): "R-T1(loo_time)", ("rf", "T2", None): "R-T2(loo_time)", ("rf", "T3", None): "R-T3-neural(loo_time)",
         ("cm", "T1", "central"): "T1-position central-FD", ("cm", "T1", "spsa"): "T1-position SPSA", ("cm", "T1", "random"): "T1-position random-search"}


def fold_edges(cfg):
    data = sp.trading_data(cfg)
    n = data.n
    t0 = int(n * cfg["splits"][1])
    return data, np.linspace(t0, n, FOLDS + 1).astype(int)


def fold_problem(cfg, data, s0, s1):
    spl = {"train": (0, int(0.75 * s0)), "val": (int(0.75 * s0), s0), "test": (s0, s1)}
    return pb.build_trading(data, "TECHM", cfg, None, splits=spl)


def replay(pr, pol, th, mode, s0, s1):
    o = te.simulate(pol, th, pr.env, np.array([s0]), s1 - 1 - s0, np.random.default_rng(0), mode, detail=True)
    return o["net"][0], o["pos"][0], o["gross"][0]


def task(t):
    cfg = load_config()
    data, edges = fold_edges(cfg)
    s0, s1 = int(edges[t["fold"]]), int(edges[t["fold"] + 1])
    pr = fold_problem(cfg, data, s0, s1)
    typ, kind, meth = t["spec"]
    pol = P.make_stoch("trading", kind)
    rng0 = sp.seed_rng(2000 + t["seed"], "wf", t["fold"], kind)
    theta0 = pol.init(rng0)
    rng = sp.seed_rng(2000 + t["seed"], "wf", t["fold"], kind, "train")
    t0 = time.time()
    if typ == "rf":
        h = t["hp"]
        r = reinforce(pr, pol, theta0, rng, t["episodes"], lr=h["lr"], B=h["B"], ent_coef=h["ent"], baseline="loo_time", gamma=cfg["trading"]["gamma"],
                      schedule="cosine", keep_theta=False)
        mode = "det"
    else:
        r = cm.METHODS[meth](pr, pol, theta0, rng, t["episodes"], B=cfg["trading"]["n_paths"] // 2, **t["hp"])
        mode = "mean"
    net, pos, gross = replay(pr, pol, np.asarray(r["theta_best"]), mode, s0, s1)
    return {"t": t, "net": net, "pos": pos, "gross": gross, "secs": time.time() - t0, "dates": [str(data.dates[s0])[:10], str(data.dates[s1 - 2])[:10]]}


def benchmarks(cfg, data, edges):
    out = {}
    for f in range(FOLDS):
        s0, s1 = int(edges[f]), int(edges[f + 1])
        pr = fold_problem(cfg, data, s0, s1)
        for nm, (pol, th) in pb.trading_benchmarks(pr).items():
            if nm in ("buy_and_hold", "RSI_rule"):
                out.setdefault(nm, {})[f] = replay(pr, pol, th, "mean", s0, s1)
    return out


def metrics(chunks, cost):
    net = np.concatenate([c[0] for c in chunks])
    pos = np.concatenate([c[1] for c in chunks])
    gross = np.concatenate([c[2] for c in chunks])
    return pb.trading_metrics(net, pos, gross)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--quick", action="store_true")
    g.add_argument("--full", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    mode = "quick" if a.quick else "full"
    M = MODES[mode]
    cfg = load_config()
    workers = a.workers or cfg["workers"]
    out = outdir("wf_" + mode)
    hp_rf = json.load(open(os.path.join(outdir("full", "models"), "hparams_rf.json")))
    hp_cm = json.load(open(os.path.join(outdir("full", "models"), "hparams_cm.json")))
    nmodels = os.path.join(outdir("neural_full", "models"), "hparams_rf.json")
    hp_rf.update(json.load(open(nmodels)) if os.path.exists(nmodels) else {})
    if "trading|TECHM|T3" not in hp_rf:
        print("neural hyper-parameters missing -> R-T3 skipped")
    data, edges = fold_edges(cfg)
    print("[%s] estimated runtime ~%d min; fold edges %s" % (mode, 2 if mode == "quick" else 35, list(map(int, edges))), flush=True)
    tasks = []
    for spec in METHODS:
        typ, kind, meth = spec
        hp = hp_rf.get("trading|TECHM|" + kind) if typ == "rf" else hp_cm.get("trading|TECHM|%s|%s" % (kind, meth))
        if hp is None:
            continue
        for f in range(FOLDS):
            for s in range(M["seeds"]):
                tasks.append(dict(spec=spec, fold=f, seed=s, hp=hp, episodes=M["episodes"]))
    print("tasks:", len(tasks), flush=True)
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(task, tasks, chunksize=1))
    bm = benchmarks(cfg, data, edges)
    cost = cfg["trading"]["cost_bps"] * 1e-4

    rows, fold_rows, pooled_rows = [], [], []
    by = {}
    for r in res:
        by.setdefault((r["t"]["spec"], r["t"]["seed"]), {})[r["t"]["fold"]] = (r["net"], r["pos"], r["gross"])
        fold_rows.append({"strategy": LABEL[r["t"]["spec"]], "fold": r["t"]["fold"] + 1, "seed": r["t"]["seed"], "start": r["dates"][0], "end": r["dates"][1],
                          "cum_return": float(np.prod(1 + r["net"]) - 1)})
    for (spec, seed), d in by.items():
        m = metrics([d[f] for f in range(FOLDS)], cost)
        rows.append({"strategy": LABEL[spec], "seed": seed, **m})
    for nm, d in bm.items():
        rows.append({"strategy": nm, "seed": 0, **metrics([d[f] for f in range(FOLDS)], cost)})
        for f in range(FOLDS):
            fold_rows.append({"strategy": nm, "fold": f + 1, "seed": 0, "start": "", "end": "", "cum_return": float(np.prod(1 + d[f][0]) - 1)})
    S = pd.DataFrame(rows)
    F = pd.DataFrame(fold_rows)
    S.to_csv(os.path.join(out, "walkforward_per_seed.csv"), index=False)
    F.to_csv(os.path.join(out, "walkforward_per_fold.csv"), index=False)
    keys = ["cum_return", "sharpe", "max_drawdown", "turnover_per_year", "exposure"]
    agg = S.groupby("strategy")[keys].agg(["mean", "std"])
    agg.columns = ["%s_%s" % c for c in agg.columns]
    agg = agg.reset_index()
    agg["n_seeds"] = S.groupby("strategy").size().to_numpy()

    # FDPG's own saved walk-forward results (text CSV only)
    d = resolve_fdpg_dir(cfg)
    fd_rows, fold_dates = [], None
    if d:
        base = os.path.join(d, "outputs", "business_eval", "full")
        for pol in ("T1", "T2"):
            p = os.path.join(base, "trading_walkforward_pooled_TECHM_%s.csv" % pol)
            q = os.path.join(base, "trading_walkforward_TECHM_%s.csv" % pol)
            if os.path.exists(p):
                t = pd.read_csv(p)
                r = t[t.strategy.str.startswith("FD-")].iloc[0]
                fd_rows.append({"strategy": "FDPG " + r.strategy + " (own code, 5 seeds)", "cum_return_mean": r.cum_return, "cum_return_std": r.cum_return_sd,
                                "sharpe_mean": r.sharpe, "sharpe_std": r.sharpe_sd, "max_drawdown_mean": r.max_drawdown, "max_drawdown_std": r.max_drawdown_sd,
                                "turnover_per_year_mean": r.turnover, "exposure_mean": r.exposure, "n_seeds": 5})
                if fold_dates is None and os.path.exists(q):
                    w = pd.read_csv(q)
                    fold_dates = w.groupby("fold")[["start", "end"]].first()
    A = pd.concat([agg, pd.DataFrame(fd_rows)], ignore_index=True)
    A.to_csv(os.path.join(out, "walkforward_pooled_summary.csv"), index=False)

    # paired-by-seed tests among our methods (same seeds/folds/initialisation) vs R-T1
    tests = []
    ref = S[S.strategy == LABEL[("rf", "T1", None)]].set_index("seed").cum_return
    for lab in S.strategy.unique():
        if lab in ("buy_and_hold", "RSI_rule") or lab == LABEL[("rf", "T1", None)]:
            continue
        x = S[S.strategy == lab].set_index("seed").cum_return
        i = ref.index.intersection(x.index)
        tests.append({"vs": "R-T1(loo_time)", "strategy": lab, "mean_diff_cum_return": float((x[i] - ref[i]).mean()), "p_mannwhitney": st.mannwhitney(x[i], ref[i])})
    T = pd.DataFrame(tests)
    T.to_csv(os.path.join(out, "walkforward_tests_vs_RT1.csv"), index=False)
    mine_dates = F[F.strategy == LABEL[("rf", "T1", None)]].groupby("fold")[["start", "end"]].first()
    L = ["# Walk-forward comparison on FDPG's fold structure\n",
         "%d expanding folds over the test region (n=%d, fold edges %s); each fold retrains on all prior data (last 25%% validation); %d seeds; %d training episodes/rollouts per run "
         "(equal budget for REINFORCE and the comparison methods; same seeds => same initial parameters across methods of the same policy).\n" % (FOLDS, data.n, list(map(int, edges)), M["seeds"], M["episodes"]),
         "## Pooled out-of-sample (mean and sd over seeds)\n", md(A), "\n## Per fold (cum. return, mean over seeds)\n",
         md(F.groupby(["strategy", "fold"]).cum_return.mean().unstack().reset_index()),
         "\n## Fold dates: ours vs FDPG's own CSV\n", "ours:\n", md(mine_dates.reset_index()), "\nFDPG:\n", md(fold_dates.reset_index()) if fold_dates is not None else "n/a",
         "\n## Tests vs R-T1 (Mann-Whitney over seeds, pooled cum. return)\n", md(T) if len(T) else "n/a",
         "\nNotes: FDPG rows come from FDPG's saved CSVs (its own code, estimators chosen by validation, 40,000 rollouts, seeds 2000+); ours use "
         "this project's code and budget, so FDPG rows are indicative, not paired. FDPG's fold-specific policies were not saved, hence not re-evaluated here; the CSVs are sufficient for this comparison.\n"]
    open(os.path.join(out, "WALKFORWARD.md"), "w", encoding="utf8").write("\n".join(L))
    print(A.round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
