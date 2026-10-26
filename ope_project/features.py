"""Linear features: sparse one-hot/tile style binary features with bounded norm, plus a dense (richer) variant.
Bin edges for return features are fitted on the TRAIN period only."""
from __future__ import annotations
import numpy as np


class _Groups:
    def __init__(self, N):
        self.N = N
        self.groups = []  # (name, size, idx (N,2))

    def add(self, name, size, idx):
        idx = np.asarray(idx)
        if idx.ndim == 1:
            idx = np.stack([idx, idx], axis=1)
        self.groups.append((name, size, idx.astype(np.int64)))

    def table(self):
        d = sum(g[1] for g in self.groups)
        T = np.zeros((self.N, 2, d), dtype=np.float32)
        off = 0
        rows = np.arange(self.N)
        for _, size, idx in self.groups:
            for p in (0, 1):
                T[rows, p, off + idx[:, p]] = 1.0
            off += size
        T /= np.sqrt(len(self.groups))
        return T, [g[0] for g in self.groups]


def _bin(x, edges):
    return np.digitize(x, edges)


class TradingFeatures:
    """Day-level lookup tables: phi(raw) with raw = [day_index, position]."""

    def __init__(self, data, train_range):
        self.data = data
        N = data.N
        lo, hi = train_range
        f = data.feat
        q = [0.2, 0.4, 0.6, 0.8]
        e1, e5, e20 = (np.quantile(f[k][lo:hi], q) for k in ("ret1", "ret5", "ret20"))
        rsi = _bin(f["rsi"], [20, 30, 40, 50, 60, 70, 80])            # 8
        flag = f["flag"].astype(int)                                   # 3
        dist = _bin(f["dist"], [-0.05, -0.02, 0.0, 0.02, 0.05])        # 6
        r1, r5, r20 = _bin(f["ret1"], e1), _bin(f["ret5"], e5), _bin(f["ret20"], e20)  # 5 each
        vz = _bin(f["volz"], [-1.0, -0.3, 0.3, 1.0])                   # 5
        pos = np.stack([np.zeros(N, int), np.ones(N, int)], axis=1)
        self._pos = pos
        g = _Groups(N)
        g.add("rsi", 8, rsi); g.add("flag", 3, flag); g.add("dist", 6, dist)
        g.add("r1", 5, r1); g.add("r5", 5, r5); g.add("r20", 5, r20); g.add("vz", 5, vz)
        g.add("pos", 2, pos)
        g.add("rsi_x_pos", 16, rsi[:, None] * 2 + pos)
        g.add("flag_x_pos", 6, flag[:, None] * 2 + pos)
        g.add("bias", 1, np.zeros(N, int))
        self.table, self.names = g.table()
        # dense variant: extra pairwise/triple interactions
        gd = _Groups(N)
        for name, size, idx in g.groups:
            gd.add(name, size, idx)
        gd.add("rsi_x_flag_x_pos", 8 * 3 * 2, (rsi * 3 + flag)[:, None] * 2 + pos)
        gd.add("r5_x_r20", 25, r5 * 5 + r20)
        gd.add("r20_x_vz", 25, r20 * 5 + vz)
        gd.add("dist_x_pos", 12, dist[:, None] * 2 + pos)
        gd.add("flag_x_r5", 15, flag * 5 + r5)
        gd.add("rsi_x_r5", 40, rsi * 5 + r5)
        gd.add("flag_x_r20_x_pos", 30, (flag * 5 + r20)[:, None] * 2 + pos)
        self.table_dense, self.names_dense = gd.table()
        self.dim = self.table.shape[2]
        self.dim_dense = self.table_dense.shape[2]

    def phi(self, raw):
        return self.table[raw[:, 0].astype(int), raw[:, 1].astype(int)]

    def phi_dense(self, raw):
        return self.table_dense[raw[:, 0].astype(int), raw[:, 1].astype(int)]
