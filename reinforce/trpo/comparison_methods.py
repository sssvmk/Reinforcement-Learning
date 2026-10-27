"""Equal-budget internal baselines. The derivative-free / finite-difference methods and the REINFORCE (leave-one-out time-dependent baseline)
implementation are the ones of this folder's REINFORCE project (`reinforce.comparison_methods`, `reinforce.algorithm`), re-used unchanged so the
baselines share data, seeds (identical initial parameters) and budgets with TRPO."""
from reinforce.comparison_methods import METHODS, GRIDS            # random, neldermead, spsa, central
from reinforce.algorithm import reinforce


def internal_reinforce(pr, pol, theta0, rng, episodes, lr, B, ent, gamma):
    return reinforce(pr, pol, theta0, rng, episodes, lr=lr, B=B, ent_coef=ent, baseline="loo_time", gamma=gamma, schedule="cosine", keep_theta=False)


__all__ = ["METHODS", "GRIDS", "internal_reinforce"]
