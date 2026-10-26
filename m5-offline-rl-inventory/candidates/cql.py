"""CQL-style conservative DQN: DQN loss + alpha * (logsumexp_a Q(s,a) - Q(s, a_data)). alpha = 0 reproduces DQN."""
from candidates.dqn import QPolicy, conservative_penalty, train_dqn  # noqa: F401


def train_cql(S, A, R, S2, D, n_actions, r_scale, gamma, steps, alpha, **kw):
    return train_dqn(S, A, R, S2, D, n_actions, r_scale, gamma, steps, alpha=alpha, **kw)


def cql_policy(name, state_dict, d_in, n_actions, hidden, r_scale, s0, alpha, steps):
    return QPolicy(name, state_dict, d_in, n_actions, hidden, r_scale, s0,
                   {"alpha": alpha, "steps": steps, "hidden": hidden}, family="CQL")
