"""Linear value function on the policy features plus time_remaining = (H - t)/H, fitted by ridge least squares on discounted returns-to-go."""
import numpy as np


def returns_to_go(rew, gamma):
    G = np.zeros_like(rew)
    run = np.zeros(rew.shape[0])
    for t in range(rew.shape[1] - 1, -1, -1):
        run = rew[:, t] + gamma * run
        G[:, t] = run
    return G


def vf_features(phi, H):
    """phi (N, H, F) -> (N, H, F+1) with time_remaining appended."""
    N = phi.shape[0]
    tr = ((H - np.arange(H)) / H)[None, :, None] * np.ones((N, 1, 1))
    return np.concatenate([phi, tr], axis=2)


class ValueFn:
    def __init__(self, H, ridge=1e-3):
        self.H, self.ridge, self.w = H, ridge, None
        self.n_fits = 0

    def fit(self, batches):
        """batches: list of (phi (N,H,F), G (N,H)). Ridge least squares over all of them."""
        X = np.concatenate([vf_features(p, self.H).reshape(-1, p.shape[2] + 1) for p, _ in batches])
        y = np.concatenate([g.reshape(-1) for _, g in batches])
        A = X.T @ X + self.ridge * len(y) * np.eye(X.shape[1])
        self.w = np.linalg.solve(A, X.T @ y)
        self.n_fits += 1

    def predict(self, phi):
        if self.w is None:
            return np.zeros(phi.shape[:2])
        return vf_features(phi, self.H) @ self.w

    def predict_terminal(self, phi):
        """V(s_H): last recorded features with time_remaining = 0 (the features of s_H itself are not simulated; documented approximation)."""
        if self.w is None:
            return np.zeros(phi.shape[0])
        X = np.concatenate([phi[:, -1, :], np.zeros((phi.shape[0], 1))], axis=1)
        return X @ self.w

    def r2(self, phi, G):
        pred = self.predict(phi)
        ss = ((G - G.mean()) ** 2).sum()
        return float(1 - ((G - pred) ** 2).sum() / ss) if ss > 0 else float("nan")
