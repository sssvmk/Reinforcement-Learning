"""Agent 1: forecasting inputs - conditional quantiles of load error e_L and price gap g."""
import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
import data

LEVELS = np.round(np.arange(0.05, 0.951, 0.05), 2)
OUTF = os.path.join(data.OUT, "forecasts")


# ---------------- core quantile model ----------------
def pinball(y, q, tau):
    d = y - q
    return np.maximum(tau * d, (tau - 1) * d)


def repair(Q):
    """Fix quantile crossing by sorting along the last axis."""
    return np.sort(Q, axis=-1)


class QuantileGBM:
    def __init__(self, levels=LEVELS, max_iter=80, seed=0):
        self.levels = np.asarray(levels)
        kw = dict(max_iter=max_iter, learning_rate=0.08, max_leaf_nodes=15, min_samples_leaf=100,
                  l2_regularization=1.0, random_state=seed)
        self.kw = kw

    def fit(self, X, y):
        self.models = [HistGradientBoostingRegressor(loss="quantile", quantile=float(t), **self.kw).fit(X, y)
                       for t in self.levels]
        self.mean_model = HistGradientBoostingRegressor(loss="squared_error", **self.kw).fit(X, y)
        return self

    def predict(self, X):
        return repair(np.stack([m.predict(X) for m in self.models], -1))

    def predict_mean(self, X):
        return self.mean_model.predict(X)


def flat(X, idx):
    return X[idx].reshape(-1, X.shape[-1])


def fit_predict(P, target, fit_idx, pred_idx, max_iter=80):
    """Train on days fit_idx, predict quantiles and mean for pred_idx. target is (D,24)."""
    m = QuantileGBM(max_iter=max_iter).fit(flat(P.X, fit_idx), target[fit_idx].ravel())
    Xp = flat(P.X, pred_idx)
    return (m.predict(Xp).reshape(len(pred_idx), 24, -1), m.predict_mean(Xp).reshape(len(pred_idx), 24))


def make_forecasts(P, n_folds=4, max_iter=80, cache=True):
    """Quantile bundle for all days: train days cross-fitted in contiguous blocks (no in-sample optimism),
    validation and test days predicted from a model trained on all training days."""
    os.makedirs(OUTF, exist_ok=True)
    f = os.path.join(OUTF, f"forecast_bundle_rho_independent_c{P.cutoff}.npz")
    if cache and os.path.exists(f):
        z = np.load(f)
        return {k: z[k] for k in z.files}
    t0 = time.time()
    D = P.D
    out = {"eL_q": np.zeros((D, 24, len(LEVELS))), "g_q": np.zeros((D, 24, len(LEVELS))),
           "g_mean": np.zeros((D, 24)), "eL_mean": np.zeros((D, 24))}
    tr = P.idx("train")
    blocks = np.array_split(tr, n_folds)
    for b in blocks:
        fit = np.setdiff1d(tr, b)
        for nm, tgt in (("eL", P.eL), ("g", P.g)):
            q, mu = fit_predict(P, tgt, fit, b, max_iter)
            out[nm + "_q"][b] = q
            out[nm + "_mean"][b] = mu
    rest = np.setdiff1d(np.arange(D), tr)
    for nm, tgt in (("eL", P.eL), ("g", P.g)):
        q, mu = fit_predict(P, tgt, tr, rest, max_iter)
        out[nm + "_q"][rest] = q
        out[nm + "_mean"][rest] = mu
    out["time_s"] = np.array(time.time() - t0)
    np.savez(f, **out)
    return out


# ---------------- baselines ----------------
def day_type(P):
    off = P.X[:, :, P.fnames.index("offday")]
    return off.astype(int)


def bias_median(P, target, fit=None, stat=np.median):
    """Median (or other stat) of the fit-period target by hour and day type (weekday/offday)."""
    tr = P.idx("train") if fit is None else fit
    dt = day_type(P)
    out = np.zeros_like(target)
    for t in (0, 1):
        for h in range(24):
            sel = dt[tr, h] == t
            out[:, h][dt[:, h] == t] = stat(target[tr][:, h][sel]) if sel.any() else 0.0
    return out


def lagged(target, lag):
    out = np.full_like(target, np.nan)
    out[lag:] = target[:-lag]
    return out


def point_baselines(P, target):
    return {"tso": np.zeros_like(target), "bias_corrected": bias_median(P, target),
            "persistence": lagged(target, P.cutoff), "seasonal_naive_7d": lagged(target, 7)}


def climatology_quantiles(P, target, fit=None):
    """Unconditional quantiles by hour and day type from the fit period."""
    tr = P.idx("train") if fit is None else fit
    dt = day_type(P)
    out = np.zeros(target.shape + (len(LEVELS),))
    for t in (0, 1):
        for h in range(24):
            sel = dt[tr, h] == t
            q = np.quantile(target[tr][:, h][sel], LEVELS)
            out[:, h][dt[:, h] == t] = q
    return out


def linear_quantiles(P, target, fit_idx, pred_idx):
    """Optional statsmodels linear quantile regression (returns None if unavailable)."""
    try:
        import warnings
        warnings.filterwarnings("ignore")
        from statsmodels.regression.quantile_regression import QuantReg
    except Exception:
        return None
    Xf = np.nan_to_num(flat(P.X, fit_idx))
    sd = Xf.std(0) + 1e-9
    mu = Xf.mean(0)
    Xf = np.c_[np.ones(len(Xf)), (Xf - mu) / sd]
    y = target[fit_idx].ravel()
    Xp = np.nan_to_num(flat(P.X, pred_idx))
    Xp = np.c_[np.ones(len(Xp)), (Xp - mu) / sd]
    cols = []
    for t in LEVELS:
        r = QuantReg(y, Xf).fit(q=float(t), max_iter=400)
        cols.append(Xp @ r.params)
    return repair(np.stack(cols, -1)).reshape(len(pred_idx), 24, -1)


# ---------------- metrics ----------------
def metrics(y, Q, levels=LEVELS):
    """y (n,24); Q (n,24,K). Returns dict of scalar metrics."""
    y3 = y[..., None]
    pb = pinball(y3, Q, levels[None, None, :])
    med = Q[..., np.argmin(np.abs(levels - 0.5))]
    lo80, hi80 = Q[..., list(levels).index(0.1)], Q[..., list(levels).index(0.9)]
    lo90, hi90 = Q[..., list(levels).index(0.05)], Q[..., list(levels).index(0.95)]
    return {"pinball_mean": pb.mean(), "mae_median": np.abs(y - med).mean(),
            "rmse_median": np.sqrt(((y - med) ** 2).mean()),
            "cov80": ((y >= lo80) & (y <= hi80)).mean(), "cov90": ((y >= lo90) & (y <= hi90)).mean(),
            "width80": (hi80 - lo80).mean(), "pinball_by_q": pb.mean((0, 1))}


def pit(y, Q):
    """Approximate PIT: fraction of quantile levels below y."""
    return (Q <= y[..., None]).mean(-1)


def point_metrics(y, f):
    ok = ~np.isnan(f)
    d = (y - f)[ok]
    return {"mae": np.abs(d).mean(), "rmse": np.sqrt((d ** 2).mean())}


def interp_quantile(Q, tau):
    """Piecewise-linear quantile function; Q (...,K) at LEVELS, tau (...) in (0,1). Flat beyond the ends."""
    tau = np.clip(tau, LEVELS[0], LEVELS[-1])
    pos = (tau - LEVELS[0]) / (LEVELS[1] - LEVELS[0])
    lo = np.clip(np.floor(pos).astype(int), 0, len(LEVELS) - 2)
    w = pos - lo
    a = np.take_along_axis(Q, lo[..., None], -1)[..., 0]
    b = np.take_along_axis(Q, (lo + 1)[..., None], -1)[..., 0]
    return a * (1 - w) + b * w


# ---------------- evaluation run ----------------
def evaluate(P, fc, outdir=OUTF):
    os.makedirs(outdir, exist_ok=True)
    rows = []
    months = np.asarray(P.dates.month)
    season = np.select([np.isin(months, [12, 1, 2]), np.isin(months, [3, 4, 5]), np.isin(months, [6, 7, 8])],
                       ["DJF", "MAM", "JJA"], "SON")
    tr, va, te = P.idx("train"), P.idx("val"), P.idx("test")
    lin = {}
    for nm, tgt in (("eL", P.eL), ("g", P.g)):
        climq = climatology_quantiles(P, tgt)
        lin_v = linear_quantiles(P, tgt, tr, np.concatenate([va, te]))
        models = {"gbm": fc[nm + "_q"], "climatology": climq}
        if lin_v is not None:
            full = np.zeros_like(climq)
            full[np.concatenate([va, te])] = lin_v
            models["linear_qr"] = full
        for split, idx in (("val", va), ("test", te)):
            for mn, Q in models.items():
                m = metrics(tgt[idx], Q[idx])
                rows.append({"target": nm, "model": mn, "split": split, **{k: v for k, v in m.items() if k != "pinball_by_q"}})
        # point baselines: pinball of a degenerate quantile forecast = |.|-based, report MAE/RMSE
        for bn, f in point_baselines(P, tgt).items():
            for split, idx in (("val", va), ("test", te)):
                pm = point_metrics(tgt[idx], f[idx])
                Qd = np.repeat(np.nan_to_num(f[idx])[..., None], len(LEVELS), -1)
                rows.append({"target": nm, "model": "pt:" + bn, "split": split, "mae_median": pm["mae"],
                             "rmse_median": pm["rmse"], "pinball_mean": pinball(tgt[idx][..., None], Qd, LEVELS).mean()})
        lin[nm] = models
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(outdir, "forecast_metrics.csv"), index=False)

    # breakdowns for the GBM on val+test
    br = []
    for nm, tgt in (("eL", P.eL), ("g", P.g)):
        Q = fc[nm + "_q"]
        for split, idx in (("val", va), ("test", te)):
            for h in range(24):
                m = metrics(tgt[idx][:, h:h + 1], Q[idx][:, h:h + 1])
                br.append({"target": nm, "split": split, "by": "hour", "key": h, "pinball": m["pinball_mean"], "cov80": m["cov80"], "cov90": m["cov90"], "width80": m["width80"]})
            for s in ["DJF", "MAM", "JJA", "SON"]:
                ii = idx[season[idx] == s]
                m = metrics(tgt[ii], Q[ii])
                br.append({"target": nm, "split": split, "by": "season", "key": s, "pinball": m["pinball_mean"], "cov80": m["cov80"], "cov90": m["cov90"], "width80": m["width80"]})
    pd.DataFrame(br).to_csv(os.path.join(outdir, "forecast_breakdown.csv"), index=False)

    # figures
    fg, ax = plt.subplots(2, 3, figsize=(15, 8))
    for r, (nm, tgt) in enumerate((("eL", P.eL), ("g", P.g))):
        Q = fc[nm + "_q"]
        for split, idx, c in (("val", va, "C0"), ("test", te, "C1")):
            ax[r, 0].hist(pit(tgt[idx], Q[idx]).ravel(), bins=np.linspace(0, 1, 21), alpha=.5, color=c, density=True, label=split)
            emp = [((tgt[idx] <= Q[idx][..., k]).mean()) for k in range(len(LEVELS))]
            ax[r, 1].plot(LEVELS, emp, "o-", color=c, label=split)
            ax[r, 2].plot(LEVELS, metrics(tgt[idx], Q[idx])["pinball_by_q"], "o-", color=c, label=split)
        ax[r, 1].plot([0, 1], [0, 1], "k--")
        ax[r, 0].set_title(f"{nm}: PIT histogram"); ax[r, 1].set_title(f"{nm}: reliability (empirical vs nominal)")
        ax[r, 2].set_title(f"{nm}: pinball by quantile"); ax[r, 0].legend()
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "calibration.png"), dpi=110); plt.close()

    b = pd.DataFrame(br)
    fg, ax = plt.subplots(1, 2, figsize=(12, 4))
    for nm in ("eL", "g"):
        s = b[(b.target == nm) & (b.by == "hour") & (b.split == "test")]
        ax[0].plot(s.key.astype(int), s.cov80, label=f"{nm} cov80 test")
    ax[0].axhline(.8, color="k", ls="--"); ax[0].set_title("80% interval coverage by hour (test)"); ax[0].legend()
    for nm in ("eL", "g"):
        s = b[(b.target == nm) & (b.by == "hour") & (b.split == "test")]
        ax[1].plot(s.key.astype(int), s.width80, label=nm)
    ax[1].set_title("Sharpness: 80% width by hour"); ax[1].legend()
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "coverage_by_hour.png"), dpi=110); plt.close()

    # example outcome table: conditional quantiles for one test day
    d = te[100]
    ex = pd.DataFrame({"hour": range(24), "pDA": P.pDA[d], "pA_mean": P.pDA[d] + fc["g_mean"][d]})
    for k in (0.1, 0.5, 0.9):
        j = list(LEVELS).index(k)
        ex[f"eL_q{k}"] = fc["eL_q"][d][:, j]
        ex[f"g_q{k}"] = fc["g_q"][d][:, j]
    ex.to_csv(os.path.join(outdir, "example_outcome_table.csv"), index=False)
    return tab


if __name__ == "__main__":
    from problem import Problem
    P = Problem()
    fc = make_forecasts(P)
    tab = evaluate(P, fc)
    pd.set_option("display.width", 200)
    print(tab.round(5).to_string())
