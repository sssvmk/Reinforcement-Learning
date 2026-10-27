"""Shared problem setup: scaled day arrays, splits, features, reference cost."""
import os
import pickle
import numpy as np
import data
import costs
import features

BURN_IN = 10  # first days lack complete lag features


class Problem:
    def __init__(self, cfg=None, rho=None, pi=None, cutoff_days=None, A=None):
        self.cfg = cfg or data.load_config()
        c = self.cfg
        self.rho = c["rho"] if rho is None else rho
        self.pi = c["pi"] if pi is None else pi
        self.cutoff = c["cutoff_days"] if cutoff_days is None else cutoff_days
        self.alpha = c["alpha"]
        self.A = A if A is not None else load_arrays(self.cfg)
        self.B = costs.scaled(self.A)
        self.dates = self.A["dates"]
        self.D = len(self.dates)
        self.masks = data.split_masks(self.dates, self.cfg)
        idx = np.arange(self.D)
        self.usable = idx >= BURN_IN
        for k in ("train",):
            self.masks[k] = self.masks[k] & self.usable
        self.Lhat, self.L, self.pDA, self.pA, self.cap = (self.B[k] for k in ("Lhat", "L", "pDA", "pA", "cap"))
        self.eL = self.L / self.Lhat - 1.0
        self.g = self.pA - self.pDA
        self.Cref = costs.cost_ref(self.B, self.masks["train"], self.rho, self.pi)
        self.X, self.fnames, self.norm = features.build_features(self.A, self.cutoff, train_mask=self.masks["train"])

    def cost(self, q, idx=None):
        """Daily cost (EUR) of purchases q (n,24) for the days idx."""
        idx = slice(None) if idx is None else idx
        return costs.daily_cost(q, self.L[idx], self.pDA[idx], self.pA[idx], self.rho, self.pi)

    def ctilde(self, q, idx=None):
        return self.cost(q, idx) / self.Cref

    def idx(self, split):
        return np.where(self.masks[split])[0]


def load_arrays(cfg):
    path = os.path.join(data.OUT, "cache")
    os.makedirs(path, exist_ok=True)
    f = os.path.join(path, "day_arrays.pkl")
    if os.path.exists(f):
        return pickle.load(open(f, "rb"))
    A = data.day_arrays(data.load_hourly(cfg), cfg)
    pickle.dump(A, open(f, "wb"))
    return A
