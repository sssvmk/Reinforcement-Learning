"""Adam with global-norm gradient clipping and an optional cosine learning-rate schedule (gradient ASCENT)."""
import numpy as np


class Adam:
    def __init__(self, n, lr=0.05, clip=5.0, schedule="const", total=1, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.clip, self.schedule, self.total = lr, clip, schedule, max(total, 1)
        self.b1, self.b2, self.eps = b1, b2, eps
        self.m, self.v, self.k = np.zeros(n), np.zeros(n), 0

    def step(self, theta, g):
        nrm = float(np.linalg.norm(g))
        if self.clip and nrm > self.clip:
            g = g * (self.clip / nrm)
        self.k += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        mh, vh = self.m / (1 - self.b1 ** self.k), self.v / (1 - self.b2 ** self.k)
        lr = self.lr * (0.5 * (1 + np.cos(np.pi * min(self.k / self.total, 1.0))) if self.schedule == "cosine" else 1.0)
        return theta + lr * mh / (np.sqrt(vh) + self.eps), nrm
