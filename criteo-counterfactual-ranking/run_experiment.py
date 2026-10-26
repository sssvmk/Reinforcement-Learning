"""End-to-end FQE experiment runner for the Criteo counterfactual-ranking
logs: builds four target policies, evaluates each with slot-MDP FQE
(off-policy and on-policy/SARSA), the H=1 bandit direct method, and the
existing IPS/SNIPS/clipped-SNIPS/DR estimators, and reports a combined
results table with bootstrap 95% confidence intervals, FQE convergence
plots, and a coverage diagnostic.

New module -- does not modify parser.py, features.py, policy.py,
estimators.py, train.py, README.md, conftest.py, requirements.txt, or any
existing test file. Reuses their public functions directly.

Run:
    python run_experiment.py --data "C:\\Users\\username\\Downloads\\sample.txt"
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from features import DEFAULT_DIM, featurize_candidate
from parser import Impression
from policy import PlackettLucePolicy
from estimators import bootstrap_ci, evaluate_policy, snips
from train import (
    dr_baseline_mc,
    featurize_impressions,
    fit_reward_model,
    load_impressions,
    policy_weights,
    r_hat_logged_banner,
    reward_model_theta,
    reward_target_arrays,
    temporal_split,
)
from fqe import (
    SlotMDPDataset,
    bandit_direct_method,
    build_slot_mdp_dataset,
    fit_fqe,
    slot1_coverage,
)

SEED = 0
TEMPERATURES = (0.5, 1.0, 2.0)
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------
# Target policy construction
# --------------------------------------------------------------------------

def build_popularity_policy(train_impressions: List[Impression], dim: int, alpha: float = 1.0, temperature: float = 1.0) -> PlackettLucePolicy:
    """Laplace-smoothed per-hashed-token click rate -> PL log-odds score.
    Deliberately NOT a jointly-fit model (that's the reward-model policy
    below); this is the simplest "popular products win" heuristic, built
    only from TRAIN displayed candidates (product features only, since
    display features are shared/uninformative per-token for popularity)."""
    shown = np.zeros(dim)
    clicked = np.zeros(dim)
    for imp in train_impressions:
        for cand in imp.displayed:
            phi = featurize_candidate({}, cand.feats, dim=dim)
            for idx, val in phi.items():
                shown[idx] += abs(val)
                if cand.clicked:
                    clicked[idx] += abs(val)
    smoothed_ctr = (clicked + alpha) / (shown + 2 * alpha)
    with np.errstate(divide="ignore"):
        theta = np.log(smoothed_ctr / (1 - smoothed_ctr))
    theta[shown == 0] = 0.0
    return PlackettLucePolicy(dim=dim, temperature=temperature, theta=theta)


def select_reward_model_temperature(
    model, dim: int, val_impressions: List[Impression], val_phis
) -> Tuple[float, Dict[float, float]]:
    """Grid-search T in {0.5, 1, 2}: build the PL policy from the reward
    model's coefficients at each T, compute importance weights on the
    VALIDATION slice, score with clipped SNIPS, and select the best T.
    (Selection uses validation only, per the task's temporal-split rule.)"""
    theta = reward_model_theta(model, dim)
    r_val, c_val = reward_target_arrays(val_impressions)
    best_t, best_score = None, -np.inf
    scores: Dict[float, float] = {}
    for t in TEMPERATURES:
        pol = PlackettLucePolicy(dim=dim, temperature=t, theta=theta)
        w = policy_weights(pol, val_impressions, val_phis)
        score = snips(w, r_val, c_val, clip_m=20.0)
        scores[t] = score
        if score > best_score:
            best_score, best_t = score, t
    return best_t, scores


def build_target_policies(train_imp, val_imp, train_phis, val_phis, dim):
    policies: Dict[str, Optional[PlackettLucePolicy]] = {}
    policies["uniform"] = PlackettLucePolicy.uniform(dim)
    policies["popularity"] = build_popularity_policy(train_imp, dim)

    reward_model = fit_reward_model(train_imp, train_phis, dim)
    best_t, val_scores = select_reward_model_temperature(reward_model, dim, val_imp, val_phis)
    theta = reward_model_theta(reward_model, dim)
    policies["reward_model"] = PlackettLucePolicy(dim=dim, temperature=best_t, theta=theta)

    # The logging-policy proxy has no parametric theta (only the logged
    # banner's propensity is known); it is special-cased throughout as
    # `policy=None` (w==1 for IS estimators, on_policy=True for FQE).
    policies["logging_proxy"] = None

    extra = {"reward_model": reward_model, "reward_model_T": best_t, "reward_model_val_scores": val_scores}
    return policies, extra


# --------------------------------------------------------------------------
# FQE value confidence interval: bootstraps the final policy-value
# averaging step over impressions, holding the already-fitted Q-function
# fixed. This is a documented approximation (see README_FQE.md) -- a
# fully rigorous CI would refit Q on every bootstrap resample, which is
# too expensive for n_boot=2000 on a laptop; this CI should be read as a
# lower bound on the true FQE sampling uncertainty.
# --------------------------------------------------------------------------

def fqe_value_ci(q_model, dataset: SlotMDPDataset, target_policy, on_policy: bool, n_boot: int = 2000, seed: int = 0):
    q_pred_all = q_model.predict(dataset.big_x)
    n = dataset.n_impressions
    vals = np.empty(n)
    weights = np.empty(n)
    for i, imp in enumerate(dataset.impressions):
        s, e = imp.row_ranges[0]
        q_vals = q_pred_all[s:e]
        if on_policy:
            v = float(q_vals[0])
        else:
            log_scores = target_policy.log_scores(dataset.plain_phis[s:e])
            m = log_scores.max()
            p = np.exp(log_scores - m)
            p /= p.sum()
            v = float(np.dot(p, q_vals))
        vals[i] = v
        weights[i] = imp.weight

    def estimator_fn(idx):
        return float(np.sum(weights[idx] * vals[idx]) / np.sum(weights[idx]))

    return bootstrap_ci(estimator_fn, n, n_boot=n_boot, seed=seed)


# --------------------------------------------------------------------------
# Per-policy evaluation on the TEST slice
# --------------------------------------------------------------------------

def evaluate_policy_full(
    name: str,
    policy: Optional[PlackettLucePolicy],
    test_imp: List[Impression],
    test_phis,
    dataset: SlotMDPDataset,
    reward_model,
    dim: int,
    seed: int,
    n_boot: int,
    n_iters: int,
    mc_samples: int,
) -> Dict[str, object]:
    is_logging = policy is None
    r, c = reward_target_arrays(test_imp)

    if is_logging:
        w = np.ones(len(test_imp))
        # DR for the logging policy evaluated against itself: baseline =
        # E_{a~pi_logging}[r_hat] has no closed form without pi_logging's
        # theta, so we use r_hat at the logged banner itself, which makes
        # DR_i = r_hat_logged_i + 1*(r_i - r_hat_logged_i) = r_i exactly
        # (DR degenerates to the raw/corrected-CTR estimator here, as it
        # should when evaluating the behavior policy against itself).
        baseline = r_hat_logged_banner(reward_model, test_imp, test_phis, dim) if reward_model is not None else None
    else:
        w = policy_weights(policy, test_imp, test_phis)
        baseline = (
            dr_baseline_mc(reward_model, test_imp, test_phis, policy, dim, n_samples=mc_samples, seed=seed)
            if reward_model is not None
            else None
        )
    r_hat_logged = r_hat_logged_banner(reward_model, test_imp, test_phis, dim) if reward_model is not None else None

    is_result = evaluate_policy(w, r, c, clip_m=20.0, baseline=baseline, r_hat_logged=r_hat_logged, n_boot=n_boot, seed=seed)

    fqe_result = fit_fqe(dataset, target_policy=policy, on_policy=is_logging, gamma=1.0, n_iters=n_iters, alpha=1.0, seed=seed)
    fqe_point, fqe_lo, fqe_hi = fqe_value_ci(fqe_result.q_model, dataset, policy, is_logging, n_boot=n_boot, seed=seed)

    bandit_entry = None
    if not is_logging:
        bandit_model, bandit_value = bandit_direct_method(dataset, policy, alpha=1.0, seed=seed)
        b_point, b_lo, b_hi = fqe_value_ci(bandit_model, dataset, policy, False, n_boot=n_boot, seed=seed)
        bandit_entry = {"point": b_point, "ci_low": b_lo, "ci_high": b_hi}

    coverage = slot1_coverage(dataset, policy, on_policy=is_logging)

    return {
        "name": name,
        "is": is_result,
        "fqe": {"point": fqe_point, "ci_low": fqe_lo, "ci_high": fqe_hi, "history": fqe_result.history},
        "bandit_dm": bandit_entry,
        "coverage_slot1": coverage,
    }


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def _fmt(entry) -> str:
    if entry is None:
        return "n/a"
    return f"{entry['point']:.4f} [{entry['ci_low']:.4f}, {entry['ci_high']:.4f}]"


def print_results_table(results: List[Dict[str, object]]) -> None:
    print("\n=== Results: policy value estimates on the TEST slice (point [95% bootstrap CI]) ===")
    header = f"{'policy':<16}{'FQE (slot-MDP)':<26}{'Bandit DM':<26}{'IPS':<26}{'SNIPS':<26}{'DR':<26}{'ESS/n':<10}{'slot1_cov':<10}"
    print(header)
    print("-" * len(header))
    for res in results:
        isr = res["is"]
        dr = isr.get("dr")
        row = (
            f"{res['name']:<16}"
            f"{_fmt(res['fqe']):<26}"
            f"{_fmt(res['bandit_dm']):<26}"
            f"{_fmt(isr['ips']):<26}"
            f"{_fmt(isr['snips']):<26}"
            f"{_fmt(dr):<26}"
            f"{isr['ess_fraction']:<10.3f}"
            f"{res['coverage_slot1']:<10.3f}"
        )
        print(row)


def plot_fqe_convergence(results: List[Dict[str, object]], out_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for res in results:
        hist = res["fqe"]["history"]
        axes[0].plot(hist["iter"], hist["bellman_error"], marker="o", label=res["name"])
        axes[1].plot(hist["iter"], hist["value"], marker="o", label=res["name"])
    axes[0].set_xlabel("FQE iteration")
    axes[0].set_ylabel("weighted RMS Bellman error")
    axes[0].set_title("FQE Bellman-error convergence")
    axes[1].set_xlabel("FQE iteration")
    axes[1].set_ylabel("V_hat(pi_e)")
    axes[1].set_title("FQE policy-value convergence")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved FQE convergence plot to {out_path}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def run(data_path: str, dim: int = DEFAULT_DIM, seed: int = SEED, n_boot: int = 2000, n_iters: int = 20, mc_samples: int = 50) -> List[Dict[str, object]]:
    print(f"Fixed random seed: {seed}")
    np.random.seed(seed)

    impressions = load_impressions(data_path)
    train_imp, val_imp, test_imp = temporal_split(impressions)
    print(f"Impressions: train={len(train_imp)} val={len(val_imp)} test={len(test_imp)} (dim={dim})")

    train_phis = featurize_impressions(train_imp, dim)
    val_phis = featurize_impressions(val_imp, dim)
    test_phis = featurize_impressions(test_imp, dim)

    policies, extra = build_target_policies(train_imp, val_imp, train_phis, val_phis, dim)
    print(f"Reward-model temperature selected on validation: T={extra['reward_model_T']} "
          f"(val SNIPS by T: { {k: round(v, 4) for k, v in extra['reward_model_val_scores'].items()} })")

    dataset = build_slot_mdp_dataset(test_imp, test_phis, dim, reward_mode="per_slot_click")

    results = []
    for name, policy in policies.items():
        print(f"Evaluating policy: {name} ...")
        res = evaluate_policy_full(
            name, policy, test_imp, test_phis, dataset, extra["reward_model"], dim, seed, n_boot, n_iters, mc_samples
        )
        results.append(res)

    print_results_table(results)
    plot_path = os.path.join(OUT_DIR, "fqe_convergence.png")
    plot_fqe_convergence(results, plot_path)

    results_path = os.path.join(OUT_DIR, "results.json")
    serializable = []
    for res in results:
        entry = dict(res)
        entry["fqe"] = {k: v for k, v in res["fqe"].items() if k != "history"}
        entry["fqe"]["history"] = res["fqe"]["history"]
        serializable.append(entry)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, default=lambda o: float(o))
    print(f"Saved full results JSON to {results_path}")

    return results


def main():
    parser_ = argparse.ArgumentParser(description=__doc__)
    parser_.add_argument("--data", default=r"C:\Users\username\Downloads\sample.txt")
    parser_.add_argument("--dim", type=int, default=DEFAULT_DIM)
    parser_.add_argument("--seed", type=int, default=SEED)
    parser_.add_argument("--n-boot", type=int, default=2000)
    parser_.add_argument("--n-iters", type=int, default=20)
    parser_.add_argument("--mc-samples", type=int, default=50)
    args = parser_.parse_args()
    run(args.data, dim=args.dim, seed=args.seed, n_boot=args.n_boot, n_iters=args.n_iters, mc_samples=args.mc_samples)


if __name__ == "__main__":
    main()
