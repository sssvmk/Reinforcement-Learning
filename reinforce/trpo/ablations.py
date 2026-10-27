"""Ablations of TRPO (same data, seeds, advantages and budgets). Each entry modifies the keyword arguments of `trpo.algorithm.trpo`.
(a) vanilla policy gradient with the same GAE advantages and Adam, no trust region
(b) natural gradient with a fixed unchecked step (no KL check, no line search)
(c) TRPO with Monte Carlo advantages (lambda = 1) and the value baseline
(d) advantage standardisation off
(e) delta / CG-iteration / damping sweeps live in run_trpo.sensitivity
"""
VARIANTS = {
    "trpo": {},
    "vpg": {"variant": "vpg"},
    "ng_fixed": {"variant": "ng_fixed"},
    "trpo_mc": {"lam": 1.0},
    "trpo_nostd": {"std": False},
    "trpo_ent": {},                       # entropy bonus: ent_coef is supplied by run_trpo --entropy
}
LABELS = {"trpo": "TRPO", "vpg": "(a) vanilla PG + GAE", "ng_fixed": "(b) natural gradient, fixed step", "trpo_mc": "(c) TRPO lambda=1", "trpo_nostd": "(d) TRPO no adv. standardisation"}


def variant_kwargs(name, base):
    """base: tuned dict(delta, lam, B, lr, ...). Returns kwargs for trpo()."""
    kw = dict(base)
    kw.update(VARIANTS[name])
    return kw
