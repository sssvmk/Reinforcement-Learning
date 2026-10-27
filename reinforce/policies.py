"""Stochastic policies with analytic score functions (REINFORCE) and the deterministic classical / FDPG policy forms.

Interface used by the simulators:  act(theta, state, rng, mode) -> (action, aux);  score(theta, aux) -> (N, P);
ent_grad(theta, aux) -> (N, P) gradient of the per-step policy entropy; phi_of(aux) -> (N, F) state features;
logp(theta, aux) -> (N,) log-probability of the stored action (for numerical gradient checks).
mode: 'sample' (stochastic), 'det' (Gaussian mean / softmax argmax / Bernoulli p >= 0.5), 'mean' (continuous mean action;
for Bernoulli this is the position p, for Gaussian and softmax the same as 'det').
"""
import numpy as np

SIGMA_FLOOR = 0.05
ORDER_GRID = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0])   # multiples of mu


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def softplus(x):
    return np.logaddexp(0.0, x)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class Stoch:
    """Base class for stochastic policies."""
    name = "stoch"
    stochastic = True
    n_params = 0
    mean_dim = 0            # number of leading parameters that define the deterministic (mean) policy

    def project(self, theta):
        return theta

    def entropy_proxy(self, theta):
        return float("nan")


# ------------------------------------------------------------------ inventory
class GaussInv(Stoch):
    """R-I1 (kind 'I1'): m ~ N(theta.phi, sigma^2), S = m*mu, order = clip(S - IP, 0, Qmax), phi = [1, weekday dummies, event, snap, price_rel].
    R-I2 (kind 'I2'): q/mu ~ N(theta.phi, sigma^2), order = clip(q, 0, Qmax), phi = [1, IP/mu, on_hand/mu, weekday dummies, event, snap, price_rel].
    Parameters: [theta (F), log sigma]; sigma floor 0.05 enforced by `project`."""

    def __init__(self, kind, sigma0=0.5):
        assert kind in ("I1", "I2")
        self.kind, self.name, self.sigma0 = kind, "R-" + kind, sigma0
        self.F = 10 if kind == "I1" else 12
        self.n_params, self.mean_dim = self.F + 1, self.F
        lo = np.r_[0.0, -3 * np.ones(self.F - 1)] if kind == "I1" else np.r_[-1.0, -2 * np.ones(self.F - 1)]
        self.lo, self.hi = lo, np.r_[8.0, 3 * np.ones(self.F - 1)]       # box for the derivative-free mean-policy baselines

    def init(self, rng):
        th = np.zeros(self.F + 1)
        if self.kind == "I1":
            th[0] = rng.uniform(1.0, 4.0)
        else:
            th[0], th[1] = rng.uniform(1.0, 4.0), -rng.uniform(0.2, 0.8)
        th[self.F] = np.log(self.sigma0)
        return th

    def project(self, theta):
        theta = theta.copy()
        theta[self.F] = max(theta[self.F], np.log(SIGMA_FLOOR))
        return theta

    def features(self, st):
        env = st["env"]
        C = env.C[st["t"]]
        one = np.ones((len(st["ip"]), 1))
        if self.kind == "I1":
            return np.hstack([one, C])
        return np.hstack([one, (st["ip"] / env.mu)[:, None], (st["oh"] / env.mu)[:, None], C])

    def act(self, theta, st, rng, mode):
        phi = self.features(st)
        m = phi @ theta[:self.F]
        sig = np.exp(theta[self.F])
        x = m + sig * rng.standard_normal(len(m)) if mode == "sample" else m
        mu = st["env"].mu
        order = x * mu - st["ip"] if self.kind == "I1" else x * mu
        return order, (phi, x, m, sig)

    def score(self, theta, aux):
        phi, x, m, sig = aux
        z = (x - m) / sig
        S = np.empty((len(x), self.F + 1))
        S[:, :self.F] = (z / sig)[:, None] * phi
        S[:, self.F] = z * z - 1.0
        return S

    def ent_grad(self, theta, aux):
        G = np.zeros((len(aux[1]), self.F + 1))
        G[:, self.F] = 1.0
        return G

    def phi_of(self, aux):
        return aux[0]

    def act_of(self, aux):
        return aux[1]            # pre-clipping sample (multiplier m or quantity ratio q/mu)

    def logp(self, theta, aux):
        phi, x, _, _ = aux
        m, ls = phi @ theta[:self.F], theta[self.F]
        return -0.5 * np.log(2 * np.pi) - ls - 0.5 * ((x - m) / np.exp(ls)) ** 2

    def entropy_proxy(self, theta):
        return float(np.exp(theta[self.F]))      # reported as sigma


class SoftmaxInv(Stoch):
    """R-I3: softmax over the order grid ORDER_GRID*mu with logits linear in phi = [1, IP/mu, on_hand/mu, weekday dummies, event, snap, price_rel]."""

    def __init__(self):
        self.kind, self.name, self.grid = "I3", "R-I3", ORDER_GRID
        self.F, self.A = 12, len(ORDER_GRID)
        self.n_params = self.F * self.A
        self.mean_dim = self.n_params

    def init(self, rng):
        W = 0.01 * rng.standard_normal((self.F, self.A))
        W[0] += -0.5 * (self.grid - rng.uniform(0.8, 2.0)) ** 2          # mild prior toward ordering about one to two days of demand
        return W.ravel()

    def features(self, st):
        env = st["env"]
        C = env.C[st["t"]]
        one = np.ones((len(st["ip"]), 1))
        return np.hstack([one, (st["ip"] / env.mu)[:, None], (st["oh"] / env.mu)[:, None], C])

    def act(self, theta, st, rng, mode):
        phi = self.features(st)
        p = softmax(phi @ theta.reshape(self.F, self.A))
        if mode == "sample":
            a = (p.cumsum(axis=1) < rng.random((len(p), 1))).sum(axis=1)
            a = np.minimum(a, self.A - 1)
        else:
            a = p.argmax(axis=1)
        return self.grid[a] * st["env"].mu, (phi, p, a)

    def score(self, theta, aux):
        phi, p, a = aux
        d = -p.copy()
        d[np.arange(len(a)), a] += 1.0
        return (phi[:, :, None] * d[:, None, :]).reshape(len(a), -1)

    def ent_grad(self, theta, aux):
        phi, p, _ = aux
        lp = np.log(np.maximum(p, 1e-300))
        H = -(p * lp).sum(axis=1, keepdims=True)
        d = -p * (lp + H)
        return (phi[:, :, None] * d[:, None, :]).reshape(len(p), -1)

    def phi_of(self, aux):
        return aux[0]

    def act_of(self, aux):
        return aux[2]            # action index

    def logp(self, theta, aux):
        phi, _, a = aux
        p = softmax(phi @ theta.reshape(self.F, self.A))
        return np.log(p[np.arange(len(a)), a])

    def entropy_proxy(self, theta):
        return float("nan")


# ------------------------------------------------------------------ trading
def trade_features(st):
    env = st["env"]
    Z = env.Z[st["t"]]
    return np.hstack([np.ones((len(Z), 1)), Z, st["prev"][:, None]])      # [bias, rsi, trend, osar, ret5, volz, previous position]


class BernoulliT(Stoch):
    """R-T1: a ~ Bernoulli(sigmoid(theta.x)), position = a. 'mean' mode (used by the derivative-free baselines) is the position p itself,
    which is exactly FDPG's F-T1 form."""
    kind, name, F = "T1", "R-T1", 7
    n_params = mean_dim = 7
    lo, hi = -5 * np.ones(7), 5 * np.ones(7)

    def init(self, rng):
        return 0.5 * rng.standard_normal(7)

    def act(self, theta, st, rng, mode):
        x = trade_features(st)
        z = x @ theta
        p = sigmoid(z)
        if mode == "sample":
            a = (rng.random(len(p)) < p).astype(float)
        elif mode == "det":
            a = (p >= 0.5).astype(float)
        else:
            a = p
        return a, (x, p, a, z)

    def score(self, theta, aux):
        x, p, a, _ = aux
        return (a - p)[:, None] * x

    def ent_grad(self, theta, aux):
        x, p, _, z = aux
        return (-z * p * (1 - p))[:, None] * x

    def phi_of(self, aux):
        return aux[0]

    def act_of(self, aux):
        return aux[2]            # sampled action in {0, 1} (in 'sample' mode)

    def logp(self, theta, aux):
        x, _, a, _ = aux
        z = x @ theta
        return a * (-softplus(-z)) + (1 - a) * (-softplus(z))

    def entropy_proxy(self, theta):
        return float("nan")


class SoftmaxT(Stoch):
    """R-T2 (optional): positions {0, 0.5, 1} with linear softmax logits on the R-T1 features."""
    kind, name, F, A = "T2", "R-T2", 7, 3
    levels = np.array([0.0, 0.5, 1.0])
    n_params = mean_dim = 21

    def init(self, rng):
        return (0.3 * rng.standard_normal((self.F, self.A))).ravel()

    def act(self, theta, st, rng, mode):
        x = trade_features(st)
        p = softmax(x @ theta.reshape(self.F, self.A))
        if mode == "sample":
            a = np.minimum((p.cumsum(axis=1) < rng.random((len(p), 1))).sum(axis=1), self.A - 1)
        else:
            a = p.argmax(axis=1)
        return self.levels[a], (x, p, a)

    def score(self, theta, aux):
        x, p, a = aux
        d = -p.copy()
        d[np.arange(len(a)), a] += 1.0
        return (x[:, :, None] * d[:, None, :]).reshape(len(a), -1)

    def ent_grad(self, theta, aux):
        x, p, _ = aux
        lp = np.log(np.maximum(p, 1e-300))
        H = -(p * lp).sum(axis=1, keepdims=True)
        return (x[:, :, None] * (-p * (lp + H))[:, None, :]).reshape(len(p), -1)

    def phi_of(self, aux):
        return aux[0]

    def act_of(self, aux):
        return aux[2]            # action index

    def logp(self, theta, aux):
        x, _, a = aux
        p = softmax(x @ theta.reshape(self.F, self.A))
        return np.log(p[np.arange(len(a)), a])

    def entropy_proxy(self, theta):
        return float("nan")


class _MLP:
    """Scalar-output network f(phi) = lin.phi + w2.tanh(W1^T phi + b1) + b2 implemented in PyTorch (float64, CPU).
    Per-sample gradients d f/d theta come from torch.func (vmap + grad). Flat layout: [lin (F), W1 (F*h), b1 (h), w2 (h), b2 (1)]."""

    def __init__(self, F, h):
        import torch
        self.torch, self.F, self.h = torch, F, h
        self.n_net = F + F * h + h + h + 1

        def f(th, phi):
            lin, W1, b1, w2, b2 = self._split_t(th)
            return phi @ lin + torch.tanh(phi @ W1 + b1) @ w2 + b2
        self._f = f
        self._g = torch.func.vmap(torch.func.grad(f), in_dims=(None, 0))

    def _split_t(self, th):
        F, h = self.F, self.h
        i = F
        lin = th[:F]
        W1 = th[i:i + F * h].reshape(F, h)
        i += F * h
        return lin, W1, th[i:i + h], th[i + h:i + 2 * h], th[i + 2 * h]

    def init(self, rng, bias0):
        th = np.zeros(self.n_net)
        W1 = rng.standard_normal((self.F, self.h)) / np.sqrt(self.F)
        th[self.F:self.F + self.F * self.h] = W1.ravel()
        o = self.F + self.F * self.h + self.h
        th[o:o + self.h] = 0.1 * rng.standard_normal(self.h) / np.sqrt(self.h)
        for k, v in bias0.items():
            th[k] = v
        return th

    def forward(self, th, phi):
        t = self.torch
        with t.no_grad():
            out = self._f(t.from_numpy(np.ascontiguousarray(th, dtype=np.float64)), t.from_numpy(np.ascontiguousarray(phi, dtype=np.float64)))
        return out.numpy(), None

    def grad(self, th, phi, a=None):
        t = self.torch
        return self._g(t.from_numpy(np.ascontiguousarray(th, dtype=np.float64)), t.from_numpy(np.ascontiguousarray(phi, dtype=np.float64))).numpy()


class NeuralGaussInv(GaussInv):
    """R-I4: R-I2 with a hidden tanh layer: q/mu ~ N(f(phi), sigma^2), phi as in R-I2, f = linear skip + 16 tanh units. Parameters [net, log sigma]."""

    def __init__(self, hidden=16, sigma0=0.5):
        super().__init__("I2", sigma0)
        self.kind, self.name = "I4", "R-I4"
        self.net = _MLP(self.F, hidden)
        self.n_params, self.mean_dim = self.net.n_net + 1, self.net.n_net
        self.lo, self.hi = -5 * np.ones(self.net.n_net), 5 * np.ones(self.net.n_net)

    def init(self, rng):
        th = np.zeros(self.n_params)
        th[:self.net.n_net] = self.net.init(rng, {0: rng.uniform(1.0, 4.0), 1: -rng.uniform(0.2, 0.8)})
        th[-1] = np.log(self.sigma0)
        return th

    def project(self, theta):
        theta = theta.copy()
        theta[-1] = max(theta[-1], np.log(SIGMA_FLOOR))
        return theta

    def act(self, theta, st, rng, mode):
        phi = self.features(st)
        m, a = self.net.forward(theta[:-1], phi)
        sig = np.exp(theta[-1])
        x = m + sig * rng.standard_normal(len(m)) if mode == "sample" else m
        return x * st["env"].mu, (phi, x, m, sig, a)

    def score(self, theta, aux):
        phi, x, m, sig, a = aux
        z = (x - m) / sig
        S = np.empty((len(x), self.n_params))
        S[:, :-1] = (z / sig)[:, None] * self.net.grad(theta[:-1], phi, a)
        S[:, -1] = z * z - 1.0
        return S

    def ent_grad(self, theta, aux):
        G = np.zeros((len(aux[1]), self.n_params))
        G[:, -1] = 1.0
        return G

    def logp(self, theta, aux):
        phi, x, _, _, _ = aux
        m, _ = self.net.forward(theta[:-1], phi)
        ls = theta[-1]
        return -0.5 * np.log(2 * np.pi) - ls - 0.5 * ((x - m) / np.exp(ls)) ** 2


class NeuralBernT(Stoch):
    """R-T3: Bernoulli long/flat with logit f(x) = linear skip + 16 tanh units on the R-T1 features."""
    kind, name, F = "T3", "R-T3", 7

    def __init__(self, hidden=16):
        self.net = _MLP(self.F, hidden)
        self.n_params = self.mean_dim = self.net.n_net
        self.lo, self.hi = -5 * np.ones(self.n_params), 5 * np.ones(self.n_params)

    def init(self, rng):
        th = self.net.init(rng, {})
        th[:self.F] = 0.5 * rng.standard_normal(self.F)
        return th

    def act(self, theta, st, rng, mode):
        x = trade_features(st)
        z, h = self.net.forward(theta, x)
        p = sigmoid(z)
        a = (rng.random(len(p)) < p).astype(float) if mode == "sample" else ((p >= 0.5).astype(float) if mode == "det" else p)
        return a, (x, p, a, z, h)

    def score(self, theta, aux):
        x, p, a, _, h = aux
        return (a - p)[:, None] * self.net.grad(theta, x, h)

    def ent_grad(self, theta, aux):
        x, p, _, z, h = aux
        return (-z * p * (1 - p))[:, None] * self.net.grad(theta, x, h)

    def phi_of(self, aux):
        return aux[0]

    def act_of(self, aux):
        return aux[2]

    def logp(self, theta, aux):
        x, _, a, _, _ = aux
        z, _ = self.net.forward(theta, x)
        return a * (-softplus(-z)) + (1 - a) * (-softplus(z))


def make_stoch(domain, kind):
    if domain == "inventory":
        return {"I1": lambda: GaussInv("I1"), "I2": lambda: GaussInv("I2"), "I3": SoftmaxInv, "I4": NeuralGaussInv}[kind]()
    return {"T1": BernoulliT, "T2": SoftmaxT, "T3": NeuralBernT}[kind]()


# ------------------------------------------------------------------ deterministic classical and FDPG policy forms
class Det:
    stochastic = False
    n_params = 0

    def phi_of(self, aux):
        raise NotImplementedError


class BaseStock(Det):
    """order = max(0, S - IP), S = theta*mu  (tuned base-stock, cover rule with theta = L+1, and FDPG F-I1)."""
    name, n_params = "base_stock", 1

    def act(self, theta, st, rng, mode):
        return st["env"].mu * theta[0] - st["ip"], None


class SSPolicy(Det):
    """(s, S): order up to S when IP <= s; s = th1*mu, S = s + softplus(th2)*mu  (tuned (s,S) and FDPG F-I2)."""
    name, n_params = "sS", 2

    def act(self, theta, st, rng, mode):
        mu, ip = st["env"].mu, st["ip"]
        s = theta[0] * mu
        S = s + softplus(theta[1]) * mu
        return np.where(ip <= s, np.maximum(0.0, S - ip), 0.0), None


class CalendarBaseStock(Det):
    """FDPG F-I3: S_t = mu*(th0 + w[weekday] + te*event + ts*snap + tp*price_rel), w[0] = 0 (10 parameters)."""
    name, n_params = "calendar_base_stock", 10

    def act(self, theta, st, rng, mode):
        env, t = st["env"], st["t"]
        w = np.r_[0.0, theta[1:7]]
        S = env.mu * (theta[0] + w[env.wday[t]] + theta[7] * env.event[t] + theta[8] * env.snap[t] + theta[9] * env.price_rel[t])
        return S - st["ip"], None


class FixedPos(Det):
    def __init__(self, value, name):
        self.value, self.name = float(value), name

    def act(self, theta, st, rng, mode):
        return np.full(len(st["t"]), self.value), None


class RsiRule(Det):
    """Long when RSI < entry, flat when RSI > exit, otherwise keep the previous position."""
    name, n_params = "RSI_rule", 2

    def act(self, theta, st, rng, mode):
        r = st["env"].X[st["t"], 0]
        return np.where(r < theta[0], 1.0, np.where(r > theta[1], 0.0, (st["prev"] > 0.5).astype(float))), None


class SarRule(Det):
    """Long when sar < low (uptrend flag), flat otherwise."""
    name = "SAR_rule"

    def act(self, theta, st, rng, mode):
        return (st["env"].X[st["t"], 1] > 0).astype(float), None


class FdpgT2(Det):
    """FDPG F-T2: soft RSI/SAR rule, pos = sigmoid(k(RSI_hi - RSI)/10) * sigmoid(k(s*trend + off)), k = 2."""
    name, n_params, k = "F-T2", 3, 2.0

    def act(self, theta, st, rng, mode):
        X = st["env"].X[st["t"]]
        return sigmoid(self.k * (theta[0] - X[:, 0]) / 10.0) * sigmoid(self.k * (theta[1] * X[:, 1] + theta[2])), None


def make_fdpg_policy(domain, kind):
    """FDPG policy families re-implemented from the formulas in the prompt (physical parameters)."""
    if domain == "inventory":
        return {"I1": BaseStock, "I2": SSPolicy, "I3": CalendarBaseStock}[kind]()
    return {"T1": lambda: BernoulliT(), "T2": FdpgT2}[kind]()
