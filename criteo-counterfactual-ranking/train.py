"""End-to-end off-policy evaluation and learning for the Criteo
counterfactual ranking logs.

Pipeline
--------
1. Parse impressions (parser.py) and featurize every candidate
   (features.py) once into sparse hashed vectors.
2. Temporal split by exID: first 60% train / next 20% validation /
   last 20% test. Never shuffled.
3. Fit a logistic-regression reward model r_hat(x, candidate, slot) on
   TRAIN displayed candidates only (non-displayed candidates are always
   unclicked by construction -- verified against the real sample -- so
   they carry no reward-model signal).
4. Build four policies:
     a. logging policy baseline      (w == 1 by construction)
     b. uniform Plackett-Luce        (theta == 0)
     c. reward-model policy          (theta == reward model's coefficients)
     d. off-policy-optimized (CRM)   (theta trained to maximize clipped
                                       SNIPS - L2 - variance penalty,
                                       initialized from (c), hyperparameters
                                       selected on VALIDATION SNIPS)
5. Evaluate all four on the TEST slice only: IPS, SNIPS, clipped IPS/SNIPS,
   DR, bootstrap CIs, ESS, mean weight, lift over the logging baseline.
"""
from __future__ import annotations

import argparse
import itertools
import math
from typing import Dict, List, Tuple

import numpy as np

from parser import Impression, parse_impressions
from features import DEFAULT_DIM, featurize_candidate, hash_token, sparse_dict_to_csr
from policy import PlackettLucePolicy
from estimators import bootstrap_ci, doubly_robust, effective_sample_size, evaluate_policy, ips, snips

SEED = 0


# --------------------------------------------------------------------------
# Data loading / splitting / featurization
# --------------------------------------------------------------------------

def load_impressions(path: str) -> List[Impression]:
    return list(parse_impressions(path))


def temporal_split(impressions: List[Impression]) -> Tuple[List[Impression], List[Impression], List[Impression]]:
    """First 60% train / next 20% val / last 20% test, ordered by exID.
    exID is already increasing with time in the raw file, but we re-sort
    defensively so the split is correct even if callers pass impressions
    out of order."""
    ordered = sorted(impressions, key=lambda imp: imp.exid)
    n = len(ordered)
    n_train = int(round(0.6 * n))
    n_val = int(round(0.2 * n))
    train = ordered[:n_train]
    val = ordered[n_train : n_train + n_val]
    test = ordered[n_train + n_val :]
    if train and val:
        assert train[-1].exid <= val[0].exid, "train/val temporal overlap"
    if val and test:
        assert val[-1].exid <= test[0].exid, "val/test temporal overlap"
    return train, val, test


def featurize_impressions(impressions: List[Impression], dim: int) -> List[List[Dict[int, float]]]:
    """Per impression: list of candidate feature dicts, in original
    candidate order (displayed candidates occupy indices [0, nb_slots))."""
    out = []
    for imp in impressions:
        out.append([featurize_candidate(imp.display_feats, cand.feats, dim=dim) for cand in imp.candidates])
    return out


def reward_target_arrays(impressions: List[Impression]) -> Tuple[np.ndarray, np.ndarray]:
    r = np.array([imp.reward for imp in impressions], dtype=np.float64)
    c = np.array([imp.weight for imp in impressions], dtype=np.float64)
    return r, c


# --------------------------------------------------------------------------
# Reward model: per-product click probability, trained on displayed slots only
# --------------------------------------------------------------------------

def _with_slot_feature(phi: Dict[int, float], slot_1indexed: int, dim: int) -> Dict[int, float]:
    out = dict(phi)
    idx, sign = hash_token(f"slot={slot_1indexed}", dim)
    out[idx] = out.get(idx, 0.0) + sign
    return out


def fit_reward_model(impressions: List[Impression], phis: List[List[Dict[int, float]]], dim: int):
    """Logistic regression predicting wasProductClicked from a candidate's
    hashed features + a hashed slot-position indicator, trained only on
    the displayed candidates of `impressions` (non-displayed candidates
    are deterministically unclicked -- including them would teach the
    model 'not shown => 0' rather than real click propensity)."""
    from sklearn.linear_model import LogisticRegression

    rows: List[Dict[int, float]] = []
    labels: List[int] = []
    for imp, cand_phis in zip(impressions, phis):
        for slot0 in range(imp.nb_slots):
            cand = imp.candidates[slot0]
            rows.append(_with_slot_feature(cand_phis[slot0], slot0 + 1, dim))
            labels.append(cand.clicked)
    X = sparse_dict_to_csr(rows, dim=dim)
    y = np.asarray(labels, dtype=np.float64)
    if y.sum() == 0 or y.sum() == len(y):
        return None  # degenerate: no variation to fit on
    model = LogisticRegression(max_iter=300, C=1.0)
    model.fit(X, y)
    return model


def predict_click_proba(model, phis_with_slots: List[Tuple[Dict[int, float], int]], dim: int) -> np.ndarray:
    if model is None or len(phis_with_slots) == 0:
        return np.zeros(len(phis_with_slots))
    rows = [_with_slot_feature(phi, slot, dim) for phi, slot in phis_with_slots]
    X = sparse_dict_to_csr(rows, dim=dim)
    return model.predict_proba(X)[:, 1]


def reward_model_theta(model, dim: int) -> np.ndarray:
    """Expose the fitted reward model's linear coefficients as a theta
    vector in the same hashed feature space, usable directly as a
    Plackett-Luce score vector (the intercept is a global additive
    constant and cancels exactly in PL's softmax ratios, so it is
    dropped)."""
    if model is None:
        return np.zeros(dim)
    return np.asarray(model.coef_, dtype=np.float64).reshape(-1)


def r_hat_logged_banner(model, impressions: List[Impression], phis: List[List[Dict[int, float]]], dim: int) -> np.ndarray:
    """r_hat(x_i, a_i) for the ACTUALLY LOGGED banner: noisy-OR aggregation
    of the reward model's per-slot click probabilities, i.e.
    1 - prod_k (1 - p_hat_k). This mirrors the observed near-exact
    equivalence (409/412 impressions in the sample) between the banner-
    level reward and "any displayed product was clicked"."""
    out = np.empty(len(impressions), dtype=np.float64)
    for i, (imp, cand_phis) in enumerate(zip(impressions, phis)):
        pairs = [(cand_phis[k], k + 1) for k in range(imp.nb_slots)]
        p = predict_click_proba(model, pairs, dim)
        out[i] = 1.0 - np.prod(1.0 - p)
    return out


def dr_baseline_mc(
    model,
    impressions: List[Impression],
    phis: List[List[Dict[int, float]]],
    policy: PlackettLucePolicy,
    dim: int,
    n_samples: int = 50,
    seed: int = 0,
) -> np.ndarray:
    """Monte-Carlo estimate of E_{a~pi_e}[r_hat(x, a)] for the DR
    baseline term: sample `n_samples` banners per impression from the
    policy's Plackett-Luce distribution over its full candidate set, score
    each with the noisy-OR reward model, and average."""
    rng = np.random.default_rng(seed)
    out = np.empty(len(impressions), dtype=np.float64)
    local_policy = PlackettLucePolicy(dim=policy.dim, temperature=policy.T, theta=policy.theta, seed=seed)
    for i, (imp, cand_phis) in enumerate(zip(impressions, phis)):
        vals = np.empty(n_samples, dtype=np.float64)
        for s in range(n_samples):
            banner = local_policy.sample_banner(cand_phis, imp.nb_slots)
            pairs = [(cand_phis[idx], k + 1) for k, idx in enumerate(banner)]
            p = predict_click_proba(model, pairs, dim)
            vals[s] = 1.0 - np.prod(1.0 - p)
        out[i] = vals.mean()
    return out


# --------------------------------------------------------------------------
# Importance weights for a given policy
# --------------------------------------------------------------------------

def policy_weights(policy: PlackettLucePolicy, impressions: List[Impression], phis: List[List[Dict[int, float]]]) -> np.ndarray:
    log_w = np.empty(len(impressions), dtype=np.float64)
    for i, (imp, cand_phis) in enumerate(zip(impressions, phis)):
        log_scores = policy.log_scores(cand_phis)
        log_pi_e = policy.log_prob_logged_banner(log_scores, imp.nb_slots)
        log_w[i] = log_pi_e - math.log(imp.propensity)
    return np.exp(log_w)


# --------------------------------------------------------------------------
# Off-policy-optimized (CRM) policy, trained with PyTorch
# --------------------------------------------------------------------------

def _build_padded_batch(phis: List[List[Dict[int, float]]], nb_slots_list: List[int]):
    """Pad every impression's candidate feature dicts into dense tensors
    so the whole Plackett-Luce log-probability computation can run as
    vectorized tensor ops (no per-impression/per-candidate Python loop
    inside the gradient-descent hot loop). Padded candidate slots get
    index 0 / value 0 (contribute exactly 0 to the dot product regardless
    of theta) and are masked out of every logsumexp via `valid_mask`."""
    import torch

    n = len(phis)
    max_m = max((len(cp) for cp in phis), default=1)
    max_nnz = max((len(phi) for cp in phis for phi in cp), default=1)
    max_nnz = max(max_nnz, 1)
    max_slots = max(nb_slots_list, default=1)

    idx = np.zeros((n, max_m, max_nnz), dtype=np.int64)
    val = np.zeros((n, max_m, max_nnz), dtype=np.float64)
    valid_mask = np.zeros((n, max_m), dtype=bool)

    for i, cand_phis in enumerate(phis):
        for j, phi in enumerate(cand_phis):
            valid_mask[i, j] = True
            if phi:
                keys = list(phi.keys())
                vals = list(phi.values())
                idx[i, j, : len(keys)] = keys
                val[i, j, : len(keys)] = vals

    return {
        "idx": torch.tensor(idx, dtype=torch.long),
        "val": torch.tensor(val, dtype=torch.float64),
        "valid_mask": torch.tensor(valid_mask, dtype=torch.bool),
        "max_m": max_m,
        "max_slots": max_slots,
    }


def _vectorized_log_pi_e(theta, temperature: float, batch, nb_slots_t, position_masks):
    """log pi_e(logged banner | x) for every impression in `batch` at once.
    `position_masks[k]` = (position_index >= k) & valid_mask, precomputed
    once outside the training loop since it never depends on theta."""
    import torch

    theta_gathered = theta[batch["idx"]]  # (n, max_m, max_nnz)
    raw_scores = (theta_gathered * batch["val"]).sum(dim=-1) / temperature  # (n, max_m)
    log_scores = raw_scores.masked_fill(~batch["valid_mask"], float("-inf"))

    total = torch.zeros(log_scores.shape[0], dtype=torch.float64)
    for k in range(batch["max_slots"]):
        mask_k = position_masks[k]
        scores_k = log_scores.masked_fill(~mask_k, float("-inf"))
        logz_k = torch.logsumexp(scores_k, dim=-1)
        chosen_valid = (k < nb_slots_t)

        logz_safe = torch.where(torch.isfinite(logz_k), logz_k, torch.zeros_like(logz_k))
        score_k = log_scores[:, k]
        score_safe = torch.where(torch.isfinite(score_k), score_k, torch.zeros_like(score_k))
        term_k = chosen_valid.to(torch.float64) * (score_safe - logz_safe)
        total = total + term_k
    return total


def train_crm_policy(
    train_impressions: List[Impression],
    train_phis: List[List[Dict[int, float]]],
    dim: int,
    theta_init: np.ndarray,
    temperature: float,
    clip_m: float,
    l2: float,
    var_penalty: float,
    n_epochs: int = 150,
    lr: float = 0.05,
    seed: int = 0,
) -> np.ndarray:
    """Gradient-ascent on the clipped-SNIPS counterfactual-risk-minimization
    objective:

        maximize   SNIPS_clipped(theta)  -  l2 * ||theta||^2  -  var_penalty * sqrt(Var[c*w*r] / n)

    (POEM / CRM-style: the variance term penalizes high-variance, i.e.
    unreliable, policies even if their point estimate looks good). Fully
    vectorized across impressions via padded tensors (see
    `_build_padded_batch` / `_vectorized_log_pi_e`) for tractable runtime
    even with a multi-point hyperparameter grid."""
    import torch

    torch.manual_seed(seed)
    r, c = reward_target_arrays(train_impressions)
    r_t = torch.tensor(r, dtype=torch.float64)
    c_t = torch.tensor(c, dtype=torch.float64)
    log_prop = torch.tensor([math.log(imp.propensity) for imp in train_impressions], dtype=torch.float64)
    nb_slots_list = [imp.nb_slots for imp in train_impressions]
    nb_slots_t = torch.tensor(nb_slots_list, dtype=torch.long)

    batch = _build_padded_batch(train_phis, nb_slots_list)
    position_idx = torch.arange(batch["max_m"]).unsqueeze(0)  # (1, max_m)
    position_masks = [((position_idx >= k) & batch["valid_mask"]) for k in range(batch["max_slots"])]

    n = len(train_impressions)
    theta = torch.tensor(theta_init, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([theta], lr=lr)

    for _epoch in range(n_epochs):
        optimizer.zero_grad()
        log_pi_e = _vectorized_log_pi_e(theta, temperature, batch, nb_slots_t, position_masks)
        w = torch.exp(log_pi_e - log_prop)
        w_clipped = torch.clamp(w, max=clip_m)
        numer = (c_t * w_clipped * r_t).sum()
        denom = (c_t * w_clipped).sum().clamp(min=1e-8)
        snips_val = numer / denom

        terms = c_t * w_clipped * r_t
        variance = terms.var(unbiased=False)
        var_term = var_penalty * torch.sqrt(variance / n + 1e-12)

        loss = -(snips_val - l2 * (theta ** 2).sum() - var_term)
        loss.backward()
        optimizer.step()

    return theta.detach().numpy().copy()


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------

def run(path: str, dim: int = DEFAULT_DIM, seed: int = SEED, n_boot: int = 2000, mc_samples: int = 50, verbose: bool = True) -> Dict[str, object]:
    np.random.seed(seed)
    print(f"[seed={seed}] Loading and featurizing impressions from {path} ...")
    impressions = load_impressions(path)
    train_imp, val_imp, test_imp = temporal_split(impressions)
    print(f"Split sizes: train={len(train_imp)} val={len(val_imp)} test={len(test_imp)}")

    train_phis = featurize_impressions(train_imp, dim)
    val_phis = featurize_impressions(val_imp, dim)
    test_phis = featurize_impressions(test_imp, dim)

    print("Fitting reward model (logistic regression, displayed slots only, train split) ...")
    reward_model = fit_reward_model(train_imp, train_phis, dim)

    # ---- Build policies ----
    policies: Dict[str, PlackettLucePolicy] = {}
    policies["uniform"] = PlackettLucePolicy.uniform(dim=dim, seed=seed)

    reward_theta = reward_model_theta(reward_model, dim)
    policies["reward_model"] = PlackettLucePolicy(dim=dim, temperature=1.0, theta=reward_theta, seed=seed)

    # ---- Hyperparameter selection for reward_model's temperature + crm policy, via validation SNIPS ----
    def val_clipped_snips(policy: PlackettLucePolicy, clip_m: float) -> float:
        w = policy_weights(policy, val_imp, val_phis)
        r, c = reward_target_arrays(val_imp)
        return snips(w, r, c, clip_m=clip_m)

    temperature_grid = [0.5, 1.0, 2.0]
    clip_grid = [5.0, 10.0, 20.0]

    best_rm_T, best_rm_clip, best_rm_score = 1.0, 10.0, -np.inf
    for T in temperature_grid:
        cand_policy = PlackettLucePolicy(dim=dim, temperature=T, theta=reward_theta, seed=seed)
        for M in clip_grid:
            score = val_clipped_snips(cand_policy, M)
            if score > best_rm_score:
                best_rm_T, best_rm_clip, best_rm_score = T, M, score
    policies["reward_model"] = PlackettLucePolicy(dim=dim, temperature=best_rm_T, theta=reward_theta, seed=seed)
    print(f"reward_model policy: selected T={best_rm_T}, clip_M={best_rm_clip} (val clipped-SNIPS={best_rm_score:.4f})")

    print("Training off-policy-optimized (CRM) policy via clipped-SNIPS maximization ...")
    l2_grid = [1e-3]
    var_penalty_grid = [0.0, 0.2]
    best_crm = None
    best_crm_score = -np.inf
    best_crm_config = None
    for T, M, l2, vp in itertools.product([1.0, 2.0], [10.0, 20.0], l2_grid, var_penalty_grid):
        theta = train_crm_policy(
            train_imp, train_phis, dim,
            theta_init=reward_theta, temperature=T, clip_m=M, l2=l2, var_penalty=vp,
            n_epochs=150, lr=0.05, seed=seed,
        )
        cand_policy = PlackettLucePolicy(dim=dim, temperature=T, theta=theta, seed=seed)
        score = val_clipped_snips(cand_policy, M)
        if score > best_crm_score:
            best_crm_score = score
            best_crm = theta
            best_crm_config = (T, M, l2, vp)
    policies["crm"] = PlackettLucePolicy(dim=dim, temperature=best_crm_config[0], theta=best_crm, seed=seed)
    print(f"crm policy: selected (T, clip_M, l2, var_penalty)={best_crm_config} (val clipped-SNIPS={best_crm_score:.4f})")

    # ---- Evaluate every policy on TEST only ----
    r_test, c_test = reward_target_arrays(test_imp)
    r_hat_logged_test = r_hat_logged_banner(reward_model, test_imp, test_phis, dim)

    results: Dict[str, object] = {}

    # Logging policy baseline: w == 1 by construction.
    w_logging = np.ones(len(test_imp))
    logging_baseline_mc = dr_baseline_mc(reward_model, test_imp, test_phis, PlackettLucePolicy.uniform(dim, seed), dim, n_samples=1, seed=seed)
    # (logging policy's own baseline would require its unknown true scores;
    # we do not report DR for the logging policy itself for this reason --
    # see README "Limitations: missing logging-policy scores".)
    results["logging_policy"] = evaluate_policy(w_logging, r_test, c_test, clip_m=best_rm_clip, n_boot=n_boot, seed=seed)

    for name in ["uniform", "reward_model", "crm"]:
        policy = policies[name]
        w = policy_weights(policy, test_imp, test_phis)
        baseline = dr_baseline_mc(reward_model, test_imp, test_phis, policy, dim, n_samples=mc_samples, seed=seed)
        clip_m_used = best_crm_config[1] if name == "crm" else best_rm_clip
        results[name] = evaluate_policy(
            w, r_test, c_test, clip_m=clip_m_used,
            baseline=baseline, r_hat_logged=r_hat_logged_test,
            n_boot=n_boot, seed=seed,
        )

    logging_snips = results["logging_policy"]["snips"]["point"]
    for name, res in results.items():
        res["lift_over_logging_snips"] = res["snips"]["point"] - logging_snips

    if verbose:
        print_results_table(results)

    return {
        "results": results,
        "config": {
            "dim": dim, "seed": seed,
            "reward_model_T": best_rm_T, "reward_model_clip_M": best_rm_clip,
            "crm_config": best_crm_config,
            "n_train": len(train_imp), "n_val": len(val_imp), "n_test": len(test_imp),
        },
    }


def print_results_table(results: Dict[str, object]) -> None:
    print("\n=== Test-slice results (policy x estimator) ===")
    header = f"{'policy':16s} {'n':>5s} {'mean_w':>8s} {'ESS':>8s} {'IPS':>22s} {'SNIPS':>22s} {'SNIPS_clip':>22s} {'DR':>22s} {'lift(SNIPS)':>12s}"
    print(header)
    for name, res in results.items():
        def fmt(key):
            if key not in res:
                return "n/a".rjust(22)
            d = res[key]
            return f"{d['point']:.4f} [{d['ci_low']:.4f},{d['ci_high']:.4f}]".rjust(22)

        print(
            f"{name:16s} {res['n']:5d} {res['mean_weight']:8.3f} {res['ess']:8.2f} "
            f"{fmt('ips')} {fmt('snips')} {fmt('snips_clipped')} {fmt('dr')} "
            f"{res.get('lift_over_logging_snips', float('nan')):12.4f}"
        )


if __name__ == "__main__":
    parser_cli = argparse.ArgumentParser()
    parser_cli.add_argument("--path", default="sample.txt")
    parser_cli.add_argument("--dim", type=int, default=DEFAULT_DIM)
    parser_cli.add_argument("--seed", type=int, default=SEED)
    parser_cli.add_argument("--n_boot", type=int, default=2000)
    parser_cli.add_argument("--mc_samples", type=int, default=50)
    args = parser_cli.parse_args()
    run(args.path, dim=args.dim, seed=args.seed, n_boot=args.n_boot, mc_samples=args.mc_samples)
