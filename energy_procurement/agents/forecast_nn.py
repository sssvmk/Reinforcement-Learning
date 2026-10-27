"""Neural quantile forecaster (PyTorch MLP) - drop-in alternative to the GBM in agents/forecast.py.

One network maps the hourly feature vector to all conditional quantile levels (monotone by construction:
first quantile + cumulative softplus increments) and a conditional mean. Trained with pinball + MSE loss, Adam,
early stopping on the chronologically last days of the fit window, averaged over a small seed ensemble.
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
import data
from agents.forecast import LEVELS, OUTF, flat

torch.set_num_threads(4)


class QuantileNet(nn.Module):
    def __init__(self, n_in, n_q, hidden=64, drop=0.1):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(drop),
                                  nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(drop))
        self.q0 = nn.Linear(hidden, 1)
        self.inc = nn.Linear(hidden, n_q - 1)
        self.mean = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.body(x)
        q = torch.cat([self.q0(h), self.q0(h) + torch.cumsum(nn.functional.softplus(self.inc(h)) * 0.3, -1)], -1)
        return q, self.mean(h).squeeze(-1)


def _pinball_t(y, q, taus):
    d = y[:, None] - q
    return torch.maximum(taus * d, (taus - 1) * d).mean()


class QuantileMLP:
    def __init__(self, levels=LEVELS, hidden=64, epochs=60, lr=2e-3, wd=1e-4, n_ens=3, seed=0, val_frac=0.12):
        self.levels = np.asarray(levels)
        self.hidden, self.epochs, self.lr, self.wd, self.n_ens, self.seed, self.val_frac = hidden, epochs, lr, wd, n_ens, seed, val_frac

    def fit(self, X, y, n_days=None):
        """X (n,F), y (n,) in chronological order, 24 consecutive rows per day."""
        X = np.nan_to_num(X)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        self.ym, self.ys = y.mean(), y.std() + 1e-9
        Xt = torch.tensor((X - self.mu) / self.sd, dtype=torch.float32)
        yt = torch.tensor((y - self.ym) / self.ys, dtype=torch.float32)
        n_val = int(len(y) / 24 * self.val_frac) * 24
        tr, va = slice(0, len(y) - n_val), slice(len(y) - n_val, len(y))
        taus = torch.tensor(self.levels, dtype=torch.float32)
        self.nets = []
        for k in range(self.n_ens):
            torch.manual_seed(self.seed + k)
            net = QuantileNet(X.shape[1], len(self.levels), self.hidden)
            opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.wd)
            best, best_state, bad = np.inf, None, 0
            g = torch.Generator().manual_seed(self.seed + k)
            for ep in range(self.epochs):
                net.train()
                perm = torch.randperm(Xt[tr].shape[0], generator=g)
                for i in range(0, len(perm), 512):
                    b = perm[i:i + 512]
                    q, m = net(Xt[tr][b])
                    loss = _pinball_t(yt[tr][b], q, taus) + 0.5 * ((m - yt[tr][b]) ** 2).mean()
                    opt.zero_grad(); loss.backward(); opt.step()
                net.eval()
                with torch.no_grad():
                    q, m = net(Xt[va])
                    v = (_pinball_t(yt[va], q, taus) + 0.5 * ((m - yt[va]) ** 2).mean()).item()
                if v < best - 1e-5:
                    best, bad = v, 0
                    best_state = {kk: vv.clone() for kk, vv in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= 8:
                        break
            net.load_state_dict(best_state)
            net.eval()
            self.nets.append(net)
        return self

    def _pred(self, X):
        Xt = torch.tensor((np.nan_to_num(X) - self.mu) / self.sd, dtype=torch.float32)
        with torch.no_grad():
            out = [n(Xt) for n in self.nets]
        Q = np.mean([o[0].numpy() for o in out], 0) * self.ys + self.ym
        M = np.mean([o[1].numpy() for o in out], 0) * self.ys + self.ym
        return np.sort(Q, -1), M

    def predict(self, X):
        return self._pred(X)[0]

    def predict_mean(self, X):
        return self._pred(X)[1]


def fit_predict(P, target, fit_idx, pred_idx, **kw):
    fit_idx = np.sort(fit_idx)
    m = QuantileMLP(**kw).fit(flat(P.X, fit_idx), target[fit_idx].ravel())
    Xp = flat(P.X, pred_idx)
    return m.predict(Xp).reshape(len(pred_idx), 24, -1), m.predict_mean(Xp).reshape(len(pred_idx), 24)


def make_forecasts(P, n_folds=4, cache=True, **kw):
    """Same bundle layout as agents.forecast.make_forecasts (cross-fitted on train, train-fit on val/test)."""
    os.makedirs(OUTF, exist_ok=True)
    f = os.path.join(OUTF, f"forecast_bundle_nn_c{P.cutoff}.npz")
    if cache and os.path.exists(f):
        z = np.load(f)
        return {k: z[k] for k in z.files}
    t0 = time.time()
    D = P.D
    out = {"eL_q": np.zeros((D, 24, len(LEVELS))), "g_q": np.zeros((D, 24, len(LEVELS))),
           "g_mean": np.zeros((D, 24)), "eL_mean": np.zeros((D, 24))}
    tr = P.idx("train")
    for b in np.array_split(tr, n_folds):
        fit = np.setdiff1d(tr, b)
        for nm, tgt in (("eL", P.eL), ("g", P.g)):
            q, mu = fit_predict(P, tgt, fit, b, **kw)
            out[nm + "_q"][b], out[nm + "_mean"][b] = q, mu
    rest = np.setdiff1d(np.arange(D), tr)
    for nm, tgt in (("eL", P.eL), ("g", P.g)):
        q, mu = fit_predict(P, tgt, tr, rest, **kw)
        out[nm + "_q"][rest], out[nm + "_mean"][rest] = q, mu
    out["time_s"] = np.array(time.time() - t0)
    np.savez(f, **out)
    return out
