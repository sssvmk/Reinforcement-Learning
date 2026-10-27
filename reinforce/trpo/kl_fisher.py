"""Per-family (Gaussian, softmax, Bernoulli) log-probabilities, score functions, exact KL(old || new) and Fisher information
for the analytic NumPy policies. All functions take a batch of M samples: features `phi` (M, F) and the stored latent action `act` (M,)
(Gaussian: the PRE-clipping sample; softmax: action index; Bernoulli: 0/1). The KL ignores order clipping (documented).
"""
import numpy as np

SIGMA_FLOOR = 0.05


def _sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def _softplus(x):
    return np.logaddexp(0.0, x)


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class Gauss:
    """theta = [w (F), log sigma]; x ~ N(phi.w, sigma^2)."""
    kind = "gauss"

    def __init__(self, F):
        self.F, self.P = F, F + 1

    def dist(self, th, phi):
        return phi @ th[:self.F], np.exp(th[self.F])

    def logp(self, th, phi, a):
        m, s = self.dist(th, phi)
        return -0.5 * np.log(2 * np.pi) - th[self.F] - 0.5 * ((a - m) / s) ** 2

    def score(self, th, phi, a):
        m, s = self.dist(th, phi)
        z = (a - m) / s
        S = np.empty((len(a), self.P))
        S[:, :self.F] = (z / s)[:, None] * phi
        S[:, self.F] = z * z - 1.0
        return S

    def kl(self, th_old, th_new, phi):
        """Mean over the batch states of KL(pi_old || pi_new)."""
        mo, so = self.dist(th_old, phi)
        mn, sn = self.dist(th_new, phi)
        return float(np.mean(np.log(sn / so) + (so ** 2 + (mo - mn) ** 2) / (2 * sn ** 2) - 0.5))

    def fvp(self, th_old, phi, v):
        _, s = self.dist(th_old, phi)
        out = np.empty(self.P)
        out[:self.F] = phi.T @ (phi @ v[:self.F]) / (s ** 2 * len(phi))
        out[self.F] = 2.0 * v[self.F]
        return out

    def diag(self, th, phi):
        _, s = self.dist(th, phi)
        return np.r_[(phi ** 2).mean(axis=0) / s ** 2, 2.0]

    def fisher_state(self, th, phi_row):
        """Per-state Fisher matrix (P, P), used in tests."""
        s = np.exp(th[self.F])
        Fm = np.zeros((self.P, self.P))
        Fm[:self.F, :self.F] = np.outer(phi_row, phi_row) / s ** 2
        Fm[self.F, self.F] = 2.0
        return Fm

    def sample(self, th, phi, rng):
        m, s = self.dist(th, phi)
        return m + s * rng.standard_normal(len(m))

    def sigma(self, th):
        return float(np.exp(th[self.F]))

    def entropy(self, th, phi):
        return float(0.5 * np.log(2 * np.pi * np.e) + th[self.F])

    def ent_value(self, th, phi):
        return self.entropy(th, phi)

    def ent_grad(self, th, phi):
        g = np.zeros(self.P)
        g[self.F] = 1.0
        return g


class Softmax:
    """theta = vec(W), W (F, A); logits = phi W."""
    kind = "softmax"

    def __init__(self, F, A):
        self.F, self.A, self.P = F, A, F * A

    def probs(self, th, phi):
        return _softmax(phi @ th.reshape(self.F, self.A))

    def logp(self, th, phi, a):
        p = self.probs(th, phi)
        return np.log(np.maximum(p[np.arange(len(a)), a.astype(int)], 1e-300))

    def score(self, th, phi, a):
        p = self.probs(th, phi)
        d = -p
        d[np.arange(len(a)), a.astype(int)] += 1.0
        return (phi[:, :, None] * d[:, None, :]).reshape(len(a), -1)

    def kl(self, th_old, th_new, phi):
        po, pn = self.probs(th_old, phi), self.probs(th_new, phi)
        return float(np.mean((po * (np.log(np.maximum(po, 1e-300)) - np.log(np.maximum(pn, 1e-300)))).sum(axis=1)))

    def fvp(self, th_old, phi, v):
        p = self.probs(th_old, phi)
        U = phi @ v.reshape(self.F, self.A)
        Q = p * U - p * (p * U).sum(axis=1, keepdims=True)
        return (phi.T @ Q / len(phi)).ravel()

    def diag(self, th, phi):
        p = self.probs(th, phi)
        return ((phi ** 2).T @ (p * (1 - p)) / len(phi)).ravel()

    def fisher_state(self, th, phi_row):
        """(diag(p) - p p^T) kron (phi phi^T) in the parameter ordering of vec(W) with W (F, A), i.e. index f*A + a."""
        p = self.probs(th, phi_row[None])[0]
        return np.kron(np.outer(phi_row, phi_row), np.diag(p) - np.outer(p, p))

    def sample(self, th, phi, rng):
        p = self.probs(th, phi)
        return np.minimum((p.cumsum(axis=1) < rng.random((len(p), 1))).sum(axis=1), self.A - 1)

    def sigma(self, th):
        return float("nan")

    def entropy(self, th, phi):
        p = self.probs(th, phi)
        return float(-(p * np.log(np.maximum(p, 1e-300))).sum(axis=1).mean())

    def ent_value(self, th, phi):
        return self.entropy(th, phi)

    def ent_grad(self, th, phi):
        """Gradient of the mean per-state entropy: dH/dlogit_a = -p_a (log p_a + H)."""
        p = self.probs(th, phi)
        lp = np.log(np.maximum(p, 1e-300))
        H = -(p * lp).sum(axis=1, keepdims=True)
        return (phi.T @ (-p * (lp + H)) / len(phi)).ravel()


class Bernoulli:
    """theta (F); p = sigmoid(phi.theta)."""
    kind = "bernoulli"

    def __init__(self, F):
        self.F = self.P = F

    def p(self, th, phi):
        return _sigmoid(phi @ th)

    def logp(self, th, phi, a):
        z = phi @ th
        return a * (-_softplus(-z)) + (1 - a) * (-_softplus(z))

    def score(self, th, phi, a):
        return (a - self.p(th, phi))[:, None] * phi

    def kl(self, th_old, th_new, phi):
        po = np.clip(self.p(th_old, phi), 1e-12, 1 - 1e-12)
        pn = np.clip(self.p(th_new, phi), 1e-12, 1 - 1e-12)
        return float(np.mean(po * np.log(po / pn) + (1 - po) * np.log((1 - po) / (1 - pn))))

    def fvp(self, th_old, phi, v):
        p = self.p(th_old, phi)
        return phi.T @ (p * (1 - p) * (phi @ v)) / len(phi)

    def diag(self, th, phi):
        p = self.p(th, phi)
        return ((phi ** 2).T @ (p * (1 - p))) / len(phi)

    def fisher_state(self, th, phi_row):
        p = float(self.p(th, phi_row[None])[0])
        return p * (1 - p) * np.outer(phi_row, phi_row)

    def sample(self, th, phi, rng):
        return (rng.random(len(phi)) < self.p(th, phi)).astype(float)

    def sigma(self, th):
        return float("nan")

    def entropy(self, th, phi):
        p = np.clip(self.p(th, phi), 1e-12, 1 - 1e-12)
        return float(-(p * np.log(p) + (1 - p) * np.log(1 - p)).mean())

    def ent_value(self, th, phi):
        return self.entropy(th, phi)

    def ent_grad(self, th, phi):
        z = phi @ th
        p = self.p(th, phi)
        return phi.T @ (-z * p * (1 - p)) / len(phi)


class _NeuralBase:
    """Policies whose mean / logit is a network f(phi; theta_net) (policies._MLP). The Fisher is the Gauss-Newton form J^T W J / M with
    J the per-sample Jacobian of f, which is exact for these families. J is cached for the current (theta, batch)."""

    def __init__(self, pol):
        self.pol, self.net = pol, pol.net
        self._c = None

    def _mj(self, th, phi):
        key = (th.tobytes(), phi.shape, phi.ctypes.data)
        if self._c is None or self._c[0] != key:
            thn = th[:self.n_net]
            self._c = (key, self.net.forward(thn, phi)[0], self.net.grad(thn, phi))
        return self._c[1], self._c[2]

    def f(self, th, phi):
        return self.net.forward(th[:self.n_net], phi)[0]


class NeuralGauss(_NeuralBase):
    """R-I4: theta = [net, log sigma]; x ~ N(f(phi), sigma^2)."""
    kind = "neural_gauss"

    def __init__(self, pol):
        super().__init__(pol)
        self.n_net = pol.net.n_net
        self.P = self.n_net + 1

    def logp(self, th, phi, a):
        m = self.f(th, phi)
        return -0.5 * np.log(2 * np.pi) - th[-1] - 0.5 * ((a - m) / np.exp(th[-1])) ** 2

    def score(self, th, phi, a):
        m, J = self._mj(th, phi)
        s = np.exp(th[-1])
        z = (a - m) / s
        S = np.empty((len(a), self.P))
        S[:, :-1] = (z / s)[:, None] * J
        S[:, -1] = z * z - 1.0
        return S

    def kl(self, th_old, th_new, phi):
        mo, mn = self.f(th_old, phi), self.f(th_new, phi)
        so, sn = np.exp(th_old[-1]), np.exp(th_new[-1])
        return float(np.mean(np.log(sn / so) + (so ** 2 + (mo - mn) ** 2) / (2 * sn ** 2) - 0.5))

    def fvp(self, th_old, phi, v):
        _, J = self._mj(th_old, phi)
        s = np.exp(th_old[-1])
        out = np.empty(self.P)
        out[:-1] = J.T @ (J @ v[:-1]) / (s ** 2 * len(phi))
        out[-1] = 2.0 * v[-1]
        return out

    def diag(self, th, phi):
        _, J = self._mj(th, phi)
        return np.r_[(J ** 2).mean(axis=0) / np.exp(2 * th[-1]), 2.0]

    def sample(self, th, phi, rng):
        return self.f(th, phi) + np.exp(th[-1]) * rng.standard_normal(len(phi))

    def sigma(self, th):
        return float(np.exp(th[-1]))

    def entropy(self, th, phi):
        return float(0.5 * np.log(2 * np.pi * np.e) + th[-1])

    def ent_value(self, th, phi):
        return self.entropy(th, phi)

    def ent_grad(self, th, phi):
        g = np.zeros(self.P)
        g[-1] = 1.0
        return g


class NeuralBern(_NeuralBase):
    """R-T3: p = sigmoid(f(phi; theta))."""
    kind = "neural_bernoulli"

    def __init__(self, pol):
        super().__init__(pol)
        self.n_net = self.P = pol.net.n_net

    def p(self, th, phi):
        return _sigmoid(self.f(th, phi))

    def logp(self, th, phi, a):
        z = self.f(th, phi)
        return a * (-_softplus(-z)) + (1 - a) * (-_softplus(z))

    def score(self, th, phi, a):
        m, J = self._mj(th, phi)
        return (a - _sigmoid(m))[:, None] * J

    def kl(self, th_old, th_new, phi):
        po = np.clip(self.p(th_old, phi), 1e-12, 1 - 1e-12)
        pn = np.clip(self.p(th_new, phi), 1e-12, 1 - 1e-12)
        return float(np.mean(po * np.log(po / pn) + (1 - po) * np.log((1 - po) / (1 - pn))))

    def fvp(self, th_old, phi, v):
        m, J = self._mj(th_old, phi)
        p = _sigmoid(m)
        return J.T @ (p * (1 - p) * (J @ v)) / len(phi)

    def diag(self, th, phi):
        m, J = self._mj(th, phi)
        p = _sigmoid(m)
        return ((J ** 2) * (p * (1 - p))[:, None]).mean(axis=0)

    def sample(self, th, phi, rng):
        return (rng.random(len(phi)) < self.p(th, phi)).astype(float)

    def sigma(self, th):
        return float("nan")

    def entropy(self, th, phi):
        p = np.clip(self.p(th, phi), 1e-12, 1 - 1e-12)
        return float(-(p * np.log(p) + (1 - p) * np.log(1 - p)).mean())

    def ent_value(self, th, phi):
        return self.entropy(th, phi)

    def ent_grad(self, th, phi):
        m, J = self._mj(th, phi)
        p = _sigmoid(m)
        return J.T @ (-m * p * (1 - p)) / len(phi)


def family_of(pol):
    """Distribution family for one of the analytic policies in policies.py."""
    k = pol.kind
    if k in ("I1", "I2"):
        return Gauss(pol.F)
    if k in ("I3", "T2"):
        return Softmax(pol.F, pol.A)
    if k == "T1":
        return Bernoulli(pol.F)
    if k == "I4":
        return NeuralGauss(pol)
    if k == "T3":
        return NeuralBern(pol)
    raise ValueError("no analytic family for policy kind %s" % k)


def full_fisher(fam, th, phi, damping=0.0):
    """Dense Fisher matrix assembled from Fisher-vector products on the identity (small parameter counts only)."""
    P = fam.P
    Fm = np.empty((P, P))
    for i in range(P):
        e = np.zeros(P)
        e[i] = 1.0
        Fm[:, i] = fam.fvp(th, phi, e)
    Fm = 0.5 * (Fm + Fm.T)
    return Fm + damping * np.eye(P)
