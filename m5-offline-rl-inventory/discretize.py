"""Shared state discretisation used by CE, Dyna-Q and the CE-based OPE estimator.

NOTE: discretising the continuous inventory state makes the tabular MDP only *approximately* Markov
(price, SNAP, trailing demand and the exact pipeline split are discarded)."""
import numpy as np

from env import EnvParams, unpack


class Discretizer:
    def __init__(self, p: EnvParams, n_pos=12, n_oh=4, use_event=False):
        self.p, self.n_pos, self.n_oh, self.use_event = p, n_pos, n_oh, use_event
        self.pos_edges = np.linspace(0.5, 10.0, n_pos - 1)          # multiples of mean daily demand; last bin open
        self.oh_edges = np.linspace(0.5, 4.0, n_oh - 1) if n_oh > 2 else np.array([1.0][:n_oh - 1])
        self.n_ev = 2 if use_event else 1
        self.n_states = n_pos * n_oh * 7 * self.n_ev

    def __call__(self, s):
        s = np.asarray(s)
        pos, on, wd, ev, _ = unpack(s, self.p)
        m = self.p.mean_demand
        pb = np.digitize(pos / m, self.pos_edges)
        ob = np.digitize(on / m, self.oh_edges)
        idx = (pb * self.n_oh + ob) * 7 + wd
        if self.use_event:
            idx = idx * 2 + (ev > 0.5).astype(int)
        return idx.astype(int)


def coverage_report(disc: Discretizer, s_disc, a, n_actions):
    n_sa = disc.n_states * n_actions
    pairs = np.unique(s_disc * n_actions + a)
    return {"n_states": disc.n_states, "n_state_action": n_sa, "states_visited": int(len(np.unique(s_disc))),
            "sa_visited": int(len(pairs)), "frac_sa_visited": len(pairs) / n_sa}
