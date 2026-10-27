"""Scenario model: forecast centers plus whole-day training residuals (keeps load-price joint structure)."""
import numpy as np
from agents import forecast as F


class Scenarios:
    """kind: 'tso' (zero load-error center, mean gap), 'bias' (hour x day-type centers), 'gbm' (conditional)."""

    def __init__(self, P, kind="bias", fc=None, fit=None):
        self.P, self.kind = P, kind
        fit = P.idx("train") if fit is None else fit
        self.fit = fit
        if kind == "tso":
            cL = np.zeros_like(P.eL)
            cg = np.full_like(P.g, P.g[fit].mean())
        elif kind == "bias":
            cL = F.bias_median(P, P.eL, fit)
            cg = F.bias_median(P, P.g, fit, stat=np.mean)
        elif kind == "gbm":
            cL = fc["eL_q"][..., list(F.LEVELS).index(0.5)]
            cg = fc["g_mean"]
        else:
            raise ValueError(kind)
        self.cL, self.cg = cL, cg
        self.rL = P.eL[fit] - cL[fit]
        self.rg = P.g[fit] - cg[fit]
        if kind == "gbm":
            self.Q = fc["eL_q"]
        else:  # center + pooled-by-hour empirical residual quantiles
            qres = np.stack([np.quantile(self.rL[:, h], F.LEVELS) for h in range(24)])
            self.Q = cL[..., None] + qres[None]

    def draw(self, d, S=None, rng=None):
        """Scenarios (S,24) of load (MWh) and settlement price for day index d."""
        P = self.P
        rL, rg = self.rL, self.rg
        if S is not None and S < len(rL):
            sel = (rng or np.random.default_rng(0)).choice(len(rL), S, replace=False)
            rL, rg = rL[sel], rg[sel]
        Ls = P.Lhat[d] * (1 + self.cL[d] + rL)
        pAs = np.maximum(P.pDA[d] + self.cg[d] + rg, 0.0)
        return Ls, pAs
