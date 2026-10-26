"""Policy families. All parameters are physical; `scale` maps them to unit-order internal coordinates."""
import numpy as np


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def softplus(x):
    return np.logaddexp(0.0, x)


class Policy:
    name = "policy"
    n = 0
    lo = hi = scale = np.zeros(0)

    def init(self, rng):
        return np.zeros(self.n)

    @property
    def ilo(self):
        return self.lo / self.scale

    @property
    def ihi(self):
        return self.hi / self.scale


# ---------------- inventory ----------------
class I1(Policy):
    """Base-stock: order = max(0, S - IP), S = theta*mu."""
    name, n = "I1", 1

    def __init__(self, mu):
        self.mu = mu
        self.lo, self.hi, self.scale = np.array([0.0]), np.array([10.0]), np.ones(1)

    def init(self, rng):
        return rng.uniform(0.5, 4.0, size=1)

    def order(self, th, ip, c):
        return np.maximum(0.0, th[0] * self.mu - ip)


class I2(Policy):
    """(s, S): order up to S when IP <= s; s = th1*mu, S = s + softplus(th2)*mu."""
    name, n = "I2", 2

    def __init__(self, mu):
        self.mu = mu
        self.lo, self.hi, self.scale = np.array([0.0, -3.0]), np.array([10.0, 3.0]), np.ones(2)

    def init(self, rng):
        return np.array([rng.uniform(0.5, 3.0), rng.uniform(-1.0, 1.0)])

    def order(self, th, ip, c):
        s = th[0] * self.mu
        S = s + softplus(th[1]) * self.mu
        return np.where(ip <= s, np.maximum(0.0, S - ip), 0.0)


class I3(Policy):
    """Calendar-aware base-stock: S_t = mu*(th0 + w[wday] + te*event + ts*snap + tp*price_rel); w[0]=0 fixed (10 params)."""
    name, n = "I3", 10

    def __init__(self, mu):
        self.mu = mu
        self.lo = np.r_[0.0, -3 * np.ones(6), -3 * np.ones(3)]
        self.hi = np.r_[10.0, 3 * np.ones(6), 3 * np.ones(3)]
        self.scale = np.ones(10)

    def init(self, rng):
        return np.r_[rng.uniform(0.5, 4.0), 0.1 * rng.normal(size=9)]

    def order(self, th, ip, c):
        w = np.r_[0.0, th[1:7]]
        S = self.mu * (th[0] + w[c["wday"]] + th[7] * c["event"] + th[8] * c["snap"] + th[9] * c["price_rel"])
        return np.maximum(0.0, S - ip)


# ---------------- trading ----------------
class T1(Policy):
    """pos = sigmoid(theta . [1, rsi, trend, osar, ret5, volz, prev]); features standardized with train statistics."""
    name, n = "T1", 7

    def __init__(self, mu, sd):
        self.mu, self.sd = np.asarray(mu, float), np.maximum(np.asarray(sd, float), 1e-12)
        self.lo, self.hi, self.scale = -5 * np.ones(7), 5 * np.ones(7), np.ones(7)

    def init(self, rng):
        return 0.5 * rng.normal(size=7)

    def act(self, th, X, prev):
        Z = (X - self.mu) / self.sd
        return sigmoid(th[0] + Z @ th[1:6] + th[6] * prev)


class T2(Policy):
    """Soft RSI/SAR rule: pos = sigmoid(k*(RSI_hi - RSI)/10) * sigmoid(k*(s*trend + off)); params RSI_hi, s, off."""
    name, n = "T2", 3

    def __init__(self, k=2.0):
        self.k = k
        self.lo, self.hi, self.scale = np.array([20.0, 0.0, -2.0]), np.array([90.0, 5.0, 2.0]), np.array([50.0, 1.0, 1.0])

    def init(self, rng):
        return np.array([rng.uniform(40, 70), rng.uniform(0.3, 2.0), rng.uniform(-0.5, 0.5)])

    def act(self, th, X, prev):
        return sigmoid(self.k * (th[0] - X[:, 0]) / 10.0) * sigmoid(self.k * (th[1] * X[:, 1] + th[2]))


class RsiRule(Policy):
    """Long when RSI < entry, flat when RSI > exit, otherwise keep the previous position."""
    name, n = "RSI_rule", 2

    def __init__(self):
        self.lo, self.hi, self.scale = np.array([10.0, 50.0]), np.array([50.0, 90.0]), np.array([50.0, 50.0])

    def act(self, th, X, prev):
        return np.where(X[:, 0] < th[0], 1.0, np.where(X[:, 0] > th[1], 0.0, (prev > 0.5).astype(float)))


class SarRule(Policy):
    """Long when sar < low (uptrend flag), flat otherwise."""
    name, n = "SAR_rule", 0

    def act(self, th, X, prev):
        return (X[:, 1] > 0).astype(float)


class Fixed(Policy):
    n = 0

    def __init__(self, value, name):
        self.value, self.name = float(value), name

    def act(self, th, X, prev):
        return np.full(len(X), self.value)
