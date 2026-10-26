"""Builds the candidate pool. Everything here is trained on the TRAIN-LOG only (MCTS: train-period demand model)."""
import numpy as np

from candidates.bc import BCPolicy, bc_metrics, train_bc
from candidates.ce import FittedMDP, fit_ce_policy, start_distribution
from candidates.cql import cql_policy, train_cql
from candidates.dqn import QPolicy, train_dqn
from candidates.dynaq import DynaQPolicy, train_dynaq
from candidates.mcts_policy import MCTSPolicy
from candidates.references import reference_pool
from env import DemandModel


def build_pool(log, p, disc, dm_args, S_star, cfg, seed, behavior, quick=False, families=None, verbose=True):
    pc = cfg["pool"]
    gamma, nS, nA = p.gamma, disc.n_states, p.n_actions
    fams = families or ["ref", "CE", "DynaQ", "BC", "MCTS", "DQN", "CQL"]
    pool = []
    say = print if verbose else (lambda *a, **k: None)
    s0d = disc(log["s0"])
    mu0 = start_distribution(s0d, nS)

    if "ref" in fams:
        pool += reference_pool(p, S_star)

    if "CE" in fams:
        mdps = {}
        settings = pc["ce_settings"][:3] if quick else pc["ce_settings"]
        for i, st in enumerate(settings):
            sm = st.get("smooth", 0.0)
            if sm not in mdps:
                mdps[sm] = FittedMDP(log["s_discrete"], log["a"], log["r"], log["s2_discrete"], nS, nA, sm)
            nm = f"ce_{st['rule']}_" + (f"nmin{st['n_min']}" if st["rule"] == "mask" else f"lam{st['lam']}") + f"_sm{sm}"
            pool.append(fit_ce_policy(nm, mdps[sm], disc, gamma, mu0, st["rule"], st.get("n_min", 1),
                                      st.get("lam", 0.0), sm))
        say(f"  CE: {len(settings)}")

    if "DynaQ" in fams:
        combos = list(zip(pc["dyna_lr"], pc["dyna_epochs"]))
        nplans = [0, 5] if quick else pc["dyna_nplan"]
        if quick:
            combos = [(pc["dyna_lr"][0], 2)]
        for n in nplans:
            for lr, ep in combos:
                Q, vis = train_dynaq(log["s_discrete"], log["a"], log["r"], log["s2_discrete"], nS, nA, gamma, lr, ep,
                                     n, seed=seed)
                pool.append(DynaQPolicy(f"dyna_n{n}_lr{lr}_ep{ep}", disc, Q, vis, s0d,
                                        {"n_planning": n, "lr": lr, "epochs": ep}))
        say(f"  DynaQ: {len(nplans) * len(combos)}")

    if "BC" in fams:
        hs = pc["bc_hidden"][:1] if quick else pc["bc_hidden"]
        eps_ = [3] if quick else pc["bc_epochs"]
        for h in hs:
            for e in eps_:
                net = train_bc(log["s_cont"], log["a"], nA, h, e, seed=seed)
                for stoch in ([False, True] if not quick else [False]):
                    pol = BCPolicy(f"bc_h{h}_e{e}_{'sample' if stoch else 'greedy'}", net, nA, stoch,
                                   {"hidden": h, "epochs": e, "stochastic": stoch})
                    pol.config.update(bc_metrics(pol, log["s_cont"], log["a"], behavior.probs(log["s_cont"])))
                    pool.append(pol)
        say("  BC done")

    d_in = p.state_dim
    if "DQN" in fams:
        steps = [200, 1000] if quick else pc["dqn_steps"]
        for dbl in ([False] if quick else [False, True]):
            r = train_dqn(log["s_cont"], log["a"], log["r"], log["s2_cont"], log["done"], nA, p.r_scale, gamma,
                          max(steps), snapshots=steps, seed=seed, double=dbl)
            for st, sd in r["snapshots"].items():
                pool.append(QPolicy(f"dqn{'_double' if dbl else ''}_s{st}", sd, d_in, nA, 64, p.r_scale, log["s0"],
                                    {"steps": st, "double": dbl}))
        say("  DQN done")

    if "CQL" in fams:
        alphas = [0, 1] if quick else pc["cql_alpha"]
        st = 500 if quick else pc["cql_steps"]
        for a in alphas:
            r = train_cql(log["s_cont"], log["a"], log["r"], log["s2_cont"], log["done"], nA, p.r_scale, gamma, st, a,
                          seed=seed)
            pool.append(cql_policy(f"cql_a{a}", r["snapshots"][st], d_in, nA, 64, p.r_scale, log["s0"], a, st))
        say("  CQL done")

    if "MCTS" in fams:
        budgets = [20] if quick else pc["mcts_budgets"]
        kinds = ["bootstrap"] if quick else pc["mcts_demand"]
        for kind in kinds:
            dmk = DemandModel(*dm_args, kind=kind, seed=seed)
            for b in budgets:
                pool.append(MCTSPolicy(f"mcts_{kind}_b{b}", p, dmk, b, 1.0, pc["mcts_h_plan"], S_star,
                                       pc["mcts_max_decisions"], seed))
        if not quick:
            dmn = DemandModel(*dm_args, kind="negbin", seed=seed)
            pool.append(MCTSPolicy(f"mcts_negbin_b{pc['mcts_negbin_budget']}", p, dmn, pc["mcts_negbin_budget"], 1.0,
                                   pc["mcts_h_plan"], S_star, pc["mcts_max_decisions"], seed))
        say(f"  MCTS: {sum(1 for q in pool if q.family == 'MCTS')}")
    return pool


def distill_mcts(pol, S_states, p, seed=0, epochs=60):
    """Distil an MCTS policy into a classifier (labels = cached MCTS actions on S_states)."""
    a = pol.greedy(S_states)
    net = train_bc(S_states, a, p.n_actions, 64, epochs, seed=seed)
    d = BCPolicy(pol.name + "_distilled", net, p.n_actions, False, {"distilled_from": pol.name})
    d.family = "MCTS-distilled"
    return d
