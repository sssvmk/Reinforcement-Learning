"""Pure-NumPy forward pass of the exported policy/value network (no torch import). Batch norm is already folded
into the conv weights by `network.export_weights`, so inference is conv+bias -> ReLU.

arch vector (float32): [type(0=cnn,1=mlp), rows, cols, planes, width_or_hidden, value_hidden]
"""
from __future__ import annotations

import time

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def load_weights(path_or_file):
    with np.load(path_or_file) as z:
        return {k: z[k].astype(np.float32) for k in z.files}


def masked_softmax(logits, legal):
    z = np.where(legal, logits, -np.inf)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    e = np.where(legal, e, 0.0)
    return e / e.sum(axis=-1, keepdims=True)


class NumpyNet:
    n_planes = 2

    def __init__(self, weights: dict):
        w = weights
        a = w["arch"]
        self.kind = int(a[0])
        self.rows, self.cols, self.planes = int(a[1]), int(a[2]), int(a[3])
        self.n_planes = self.planes
        self.w = {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in w.items() if k != "arch"}
        if self.kind == 0:
            for i in range(3):   # (out, in, 3, 3) -> (in*9, out) for matmul against sliding windows
                W = self.w[f"conv{i}_w"]
                self.w[f"conv{i}_m"] = np.ascontiguousarray(W.reshape(W.shape[0], -1).T)

    def _conv(self, x, i):
        """x: [B,H,W,Cin] channels-last -> [B,H,W,Cout]."""
        B, H, W, C = x.shape
        xp = np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
        win = sliding_window_view(xp, (3, 3), axis=(1, 2))        # [B,H,W,C,3,3]
        cols = win.reshape(B * H * W, C * 9)
        out = cols @ self.w[f"conv{i}_m"] + self.w[f"conv{i}_b"]
        return np.maximum(out, 0).reshape(B, H, W, -1)

    def forward(self, planes):
        """planes [B,P,R,C] -> (policy logits [B,C], value [B]); matches the torch module in eval mode."""
        x = np.asarray(planes, dtype=np.float32)
        w = self.w
        if self.kind == 0:
            x = np.ascontiguousarray(x.transpose(0, 2, 3, 1))
            for i in range(3):
                x = self._conv(x, i)
            x = np.ascontiguousarray(x.transpose(0, 3, 1, 2)).reshape(len(x), -1)   # torch flatten order (C,H,W)
        else:
            x = x.reshape(len(x), -1)
            x = np.maximum(x @ w["l1_w"].T + w["l1_b"], 0)
            x = np.maximum(x @ w["l2_w"].T + w["l2_b"], 0)
        logits = x @ w["pol_w"].T + w["pol_b"]
        h = np.maximum(x @ w["v1_w"].T + w["v1_b"], 0)
        v = np.tanh(h @ w["v2_w"].T + w["v2_b"])[:, 0]
        return logits, v


class NumpyEvaluator:
    """Same interface as az_mcts evaluators."""

    def __init__(self, weights: dict):
        self.net = NumpyNet(weights)
        self.n_planes = self.net.planes
        self.n_evals = 0
        self.n_calls = 0
        self.eval_seconds = 0.0

    @classmethod
    def from_file(cls, path):
        return cls(load_weights(path))

    def evaluate(self, planes, legal):
        t = time.perf_counter()
        logits, v = self.net.forward(planes)
        p = masked_softmax(logits.astype(np.float64), legal)
        self.eval_seconds += time.perf_counter() - t
        self.n_evals += len(planes)
        self.n_calls += 1
        return p, v.astype(np.float64)
