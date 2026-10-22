"""Fitted Q Evaluation (FQE) on the Criteo logs, formulated as a slot-by-
slot Markov Decision Process that mirrors Plackett-Luce banner sampling.

This module is new (added for the FQE task) and does NOT modify any of
the pre-existing files (parser.py, features.py, policy.py, estimators.py,
train.py, README.md, tests/*). It imports from them instead.

Slot-MDP formulation (see README_FQE.md for the full rationale)
-----------------------------------------------------------------
- Episode = one impression. Horizon H = nb_slots. Discount gamma = 1
  (configurable; gamma=0 + horizon_cap=1 recovers the single-step bandit
  special case used as a baseline / sanity check).
- Because the displayed candidates are always the first `nb_slots`
  entries of an impression's candidate list (in slot order -- see
  parser.py / policy.py), the remaining candidate pool at slot k
  (0-indexed) is *exactly* the suffix `candidates[k:]`. This means:
    * action a_k == candidates[k] (always local index 0 of the slot-k
      remaining list)
    * the logged next action a_{k+1} is always local index 0 of the
      slot-(k+1) remaining list -- used directly by the on-policy/SARSA
      FQE variant, no extra bookkeeping needed.
- State s_k = (display features, slot index k, horizon H, a summary of
  the k already-placed candidates [count + mean hashed feature vector,
  placed into *separate* hashed buckets via a "placed:" token prefix so
  they do not collide with the evaluated candidate's own features]). The
  Q-function also takes the candidate under evaluation as the action
  input, so Q(s_k, a) is a single joint hashed feature vector.
- Reward r_k defaults to `wasProductClicked` of the candidate placed at
  slot k (REWARD_MODE="per_slot_click"). An alternative
  REWARD_MODE="banner_click_last_slot" assigns the banner-level
  `wasAdClicked` label only at the terminal slot (reward 0 elsewhere).
  Step 0 found these are consistent in 409/412 (99.3%) sample
  impressions; the per-slot default is used because it gives the Q
  model real, position-resolved learning signal (see README_FQE.md).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse import csr_matrix

from features import featurize_candidate, hash_token, sparse_dict_to_csr
from parser import Candidate, Impression
from policy import PlackettLucePolicy

REWARD_MODES = ("per_slot_click", "banner_click_last_slot")


# --------------------------------------------------------------------------
# State-action featurization
# --------------------------------------------------------------------------

def build_state_action_features(
    display_feats: Dict[int, object],
    placed: Sequence[Candidate],
    slot0: int,
    nb_slots: int,
    candidate: Candidate,
    dim: int,
) -> Dict[int, float]:
    """Q(s_k, a)'s input feature vector: the candidate's own
    display+product hashed features (reusing features.featurize_candidate
    verbatim), plus hashed slot/horizon/placed-so-far-summary tokens in
    separate buckets (via distinct token prefixes) so they do not
    interfere with the plain candidate features used for PL *scoring*."""
    f = dict(featurize_candidate(display_feats, candidate.feats, dim=dim))

    idx, sign = hash_token(f"slot={slot0 + 1}", dim)
    f[idx] = f.get(idx, 0.0) + sign
    idx, sign = hash_token(f"horizon={nb_slots}", dim)
    f[idx] = f.get(idx, 0.0) + sign
    idx, sign = hash_token("n_placed", dim)
    f[idx] = f.get(idx, 0.0) + sign * math.log1p(len(placed))

    if placed:
        acc: Dict[int, float] = {}
        for cand in placed:
            cand_phi = featurize_candidate({}, cand.feats, dim=dim)  # product-only, avoid double-counting display feats
            for idxp, valp in cand_phi.items():
                acc[idxp] = acc.get(idxp, 0.0) + valp
        n_placed = len(placed)
        for idxp, valp in acc.items():
            idx2, sign2 = hash_token(f"placed:{idxp}", dim)
            f[idx2] = f.get(idx2, 0.0) + sign2 * (valp / n_placed)
    return f


# --------------------------------------------------------------------------
# Dataset: precompute everything that is independent of the Q-function's
# current parameters and of the target policy, once per impression set.
# --------------------------------------------------------------------------

@dataclass
class ImpressionSlots:
    nb_slots: int
    nb_candidates: int
    weight: float  # c_i, the 10x non-click sampling correction
    rewards: List[float]  # length nb_slots: r_0 .. r_{H-1}
    row_ranges: List[Tuple[int, int]]  # length nb_slots+1; row_ranges[k] = rows for remaining(s_k) in BigX/plain_phis


@dataclass
class SlotMDPDataset:
    dim: int
    big_x: csr_matrix  # stacked Q-input features, (n_rows, dim)
    plain_phis: List[Dict[int, float]]  # same row alignment: plain candidate phis (for PL policy scoring)
    impressions: List[ImpressionSlots]

    @property
    def n_impressions(self) -> int:
        return len(self.impressions)


def _reward_for_slot(imp: Impression, k: int, mode: str) -> float:
    if mode == "per_slot_click":
        return float(imp.candidates[k].clicked)
    if mode == "banner_click_last_slot":
        return float(imp.reward) if k == imp.nb_slots - 1 else 0.0
    raise ValueError(f"unknown reward_mode {mode!r}, expected one of {REWARD_MODES}")


def build_slot_mdp_dataset(
    impressions: List[Impression],
    phis: List[List[Dict[int, float]]],
    dim: int,
    reward_mode: str = "per_slot_click",
) -> SlotMDPDataset:
    """Build the slot-MDP dataset once; reusable across every target
    policy and every FQE run (on-policy / off-policy / bandit special
    case), since the Q-input features and plain candidate features do
    not depend on the policy being evaluated."""
    if reward_mode not in REWARD_MODES:
        raise ValueError(f"reward_mode must be one of {REWARD_MODES}, got {reward_mode!r}")

    big_q_rows: List[Dict[int, float]] = []
    big_phi_rows: List[Dict[int, float]] = []
    imp_slots: List[ImpressionSlots] = []
    cursor = 0
    for imp, cand_phis in zip(impressions, phis):
        H = imp.nb_slots
        M = imp.nb_candidates
        row_ranges: List[Tuple[int, int]] = []
        for k in range(H + 1):
            placed = imp.candidates[:k]
            start = cursor
            for j in range(k, M):
                qf = build_state_action_features(imp.display_feats, placed, k, H, imp.candidates[j], dim)
                big_q_rows.append(qf)
                big_phi_rows.append(cand_phis[j])
                cursor += 1
            row_ranges.append((start, cursor))
        rewards = [_reward_for_slot(imp, k, reward_mode) for k in range(H)]
        imp_slots.append(ImpressionSlots(nb_slots=H, nb_candidates=M, weight=imp.weight, rewards=rewards, row_ranges=row_ranges))

    big_x = sparse_dict_to_csr(big_q_rows, dim=dim)
    return SlotMDPDataset(dim=dim, big_x=big_x, plain_phis=big_phi_rows, impressions=imp_slots)


# --------------------------------------------------------------------------
# Q-model wrapper. theta_0 (unfitted) predicts 0 everywhere, matching the
# "Q=0 at/after the terminal slot" convention used by the iteration.
# --------------------------------------------------------------------------

class QModel:
    def __init__(self, dim: int, alpha: float = 1.0, seed: int = 0, kind: str = "ridge"):
        self.dim = dim
        self.alpha = alpha
        self.seed = seed
        self.kind = kind
        self.model = None

    def predict(self, X: csr_matrix) -> np.ndarray:
        if self.model is None:
            return np.zeros(X.shape[0], dtype=np.float64)
        return self.model.predict(X)

    def fit(self, X: csr_matrix, y: np.ndarray, sample_weight: np.ndarray) -> None:
        if self.kind == "ridge":
            from sklearn.linear_model import Ridge

            self.model = Ridge(alpha=self.alpha, random_state=self.seed)
        else:
            raise ValueError(f"unknown QModel kind {self.kind!r}")
        self.model.fit(X, y, sample_weight=sample_weight)


def _softmax(log_scores: np.ndarray) -> np.ndarray:
    m = np.max(log_scores)
    p = np.exp(log_scores - m)
    p /= p.sum()
    return p


# --------------------------------------------------------------------------
# FQE iteration
# --------------------------------------------------------------------------

@dataclass
class FQEResult:
    q_model: QModel
    history: Dict[str, List[float]] = field(default_factory=lambda: {"iter": [], "bellman_error": [], "value": []})
    value: float = float("nan")


def _build_transitions(dataset: SlotMDPDataset, horizon_cap: Optional[int]):
    """Flatten (impression, slot) real transitions into parallel arrays,
    plus, for each non-terminal transition, the BigX/plain_phis row-range
    of the *next* state's remaining-candidate pool (needed both for the
    exact off-policy expectation and for the on-policy/SARSA bootstrap,
    which just reads local row 0 of that range)."""
    x_rows: List[int] = []
    rewards: List[float] = []
    weights: List[float] = []
    terminal: List[bool] = []
    next_ranges: List[Optional[Tuple[int, int]]] = []
    for imp in dataset.impressions:
        h_eff = imp.nb_slots if horizon_cap is None else min(imp.nb_slots, horizon_cap)
        for k in range(h_eff):
            start_k, _ = imp.row_ranges[k]
            x_rows.append(start_k)
            rewards.append(imp.rewards[k])
            weights.append(imp.weight)
            is_terminal = (k + 1 >= imp.nb_slots) or (horizon_cap is not None and k + 1 >= horizon_cap)
            terminal.append(is_terminal)
            next_ranges.append(None if is_terminal else imp.row_ranges[k + 1])
    return (
        np.array(x_rows, dtype=np.int64),
        np.array(rewards, dtype=np.float64),
        np.array(weights, dtype=np.float64),
        np.array(terminal, dtype=bool),
        next_ranges,
    )


def fit_fqe(
    dataset: SlotMDPDataset,
    target_policy: Optional[PlackettLucePolicy] = None,
    on_policy: bool = False,
    gamma: float = 1.0,
    n_iters: int = 20,
    alpha: float = 1.0,
    seed: int = 0,
    horizon_cap: Optional[int] = None,
) -> FQEResult:
    """Fitted Q iteration on the slot-MDP.

    `on_policy=True` ignores `target_policy` and bootstraps on the
    *logged* next action (SARSA-style), estimating the value of the
    logging policy itself. `on_policy=False` requires `target_policy` and
    computes the slot-(k+1) bootstrap expectation *exactly* (a weighted
    sum over the remaining candidates' PL probabilities under
    `target_policy`), per the spec.

    `horizon_cap=1, gamma=0.0, n_iters=1` recovers the single-step bandit
    direct method as a special case of this same code path (sanity check
    3): no bootstrap term ever applies, so this degenerates to one
    weighted ridge regression of reward on (s_0, a_0) features, identical
    to `bandit_direct_method`.
    """
    if not on_policy and target_policy is None:
        raise ValueError("target_policy is required unless on_policy=True")

    x_rows, rewards, weights, terminal, next_ranges = _build_transitions(dataset, horizon_cap)
    n_transitions = len(rewards)
    x_sa = dataset.big_x[x_rows]

    # Precompute target-policy probabilities over each non-terminal
    # transition's next-state remaining pool. These depend only on the
    # target policy's (fixed) theta, NOT on the Q-model, so they are
    # computed once and reused across all FQE iterations.
    next_probs: List[Optional[np.ndarray]] = [None] * n_transitions
    if not on_policy:
        for t in range(n_transitions):
            if terminal[t]:
                continue
            s, e = next_ranges[t]
            log_scores = target_policy.log_scores(dataset.plain_phis[s:e])
            next_probs[t] = _softmax(log_scores)

    q_model = QModel(dataset.dim, alpha=alpha, seed=seed)
    history: Dict[str, List[float]] = {"iter": [], "bellman_error": [], "value": []}

    for n_iter in range(1, n_iters + 1):
        q_pred_all = q_model.predict(dataset.big_x)  # one predict() call over the whole precomputed matrix
        bootstrap = np.zeros(n_transitions, dtype=np.float64)
        for t in range(n_transitions):
            if terminal[t]:
                continue
            s, e = next_ranges[t]
            if on_policy:
                bootstrap[t] = q_pred_all[s]  # logged next action == local row 0
            else:
                bootstrap[t] = float(np.dot(next_probs[t], q_pred_all[s:e]))
        y = rewards + gamma * bootstrap * (~terminal)

        q_pred_sa_prev = q_pred_all[x_rows]
        bellman_err = float(np.sqrt(np.average((q_pred_sa_prev - y) ** 2, weights=weights)))

        q_model.fit(x_sa, y, sample_weight=weights)
        value = estimate_policy_value(q_model, dataset, target_policy, on_policy=on_policy)

        history["iter"].append(n_iter)
        history["bellman_error"].append(bellman_err)
        history["value"].append(value)

    return FQEResult(q_model=q_model, history=history, value=history["value"][-1] if history["value"] else float("nan"))


def estimate_policy_value(
    q_model: QModel,
    dataset: SlotMDPDataset,
    target_policy: Optional[PlackettLucePolicy],
    on_policy: bool = False,
) -> float:
    """V_hat(pi_e) = sum_i c_i * E_{a~pi_e(.|s_1,i)}[Q(s_1,i, a)] / sum_i c_i,
    averaged (c_i-weighted) over first-slot states across impressions."""
    numer = 0.0
    denom = 0.0
    q_pred_all = q_model.predict(dataset.big_x)
    for imp in dataset.impressions:
        s, e = imp.row_ranges[0]
        q_vals = q_pred_all[s:e]
        if on_policy:
            v = float(q_vals[0])  # logged first action
        else:
            log_scores = target_policy.log_scores(dataset.plain_phis[s:e])
            p = _softmax(log_scores)
            v = float(np.dot(p, q_vals))
        numer += imp.weight * v
        denom += imp.weight
    return numer / denom if denom else float("nan")


# --------------------------------------------------------------------------
# Bandit direct method: H=1, gamma=0 special case, implemented
# independently of fit_fqe (a plain weighted reward regression) so that
# sanity check 3 is a genuine cross-check of two separate code paths, not
# a tautology.
# --------------------------------------------------------------------------

def bandit_direct_method(
    dataset: SlotMDPDataset,
    target_policy: PlackettLucePolicy,
    alpha: float = 1.0,
    seed: int = 0,
) -> Tuple[QModel, float]:
    """Direct-method (reward regression) estimate of V(pi_e): fit
    Q(s_0, a_0) = E[r_0] by weighted ridge regression on slot-0
    transitions only, then take the policy-weighted average over all
    slot-0 candidates (not just the logged one)."""
    x_rows = []
    y = []
    w = []
    for imp in dataset.impressions:
        s, _ = imp.row_ranges[0]
        x_rows.append(s)
        y.append(imp.rewards[0])
        w.append(imp.weight)
    x = dataset.big_x[np.array(x_rows, dtype=np.int64)]
    model = QModel(dataset.dim, alpha=alpha, seed=seed)
    model.fit(x, np.array(y, dtype=np.float64), sample_weight=np.array(w, dtype=np.float64))
    value = estimate_policy_value(model, dataset, target_policy, on_policy=False)
    return model, value


# --------------------------------------------------------------------------
# Coverage diagnostic
# --------------------------------------------------------------------------

def slot1_coverage(dataset: SlotMDPDataset, target_policy: Optional[PlackettLucePolicy], on_policy: bool = False) -> float:
    """c_i-weighted average share of pi_e's slot-1 probability mass that
    falls on candidates actually displayed somewhere in THAT SAME
    impression (i.e. positions < nb_slots in the original candidate list).

    Note: the Criteo schema has no stable product identifier across
    impressions (candidate pools differ per impression), so a literal
    "products the logging policy ever placed in that slot" (pooled across
    the whole log) global diagnostic is not computable. This is an
    explicitly documented per-impression proxy for that idea: low
    coverage means pi_e wants to show candidates for which this
    impression's logged banner gives no direct reward evidence at all."""
    numer = 0.0
    denom = 0.0
    for imp in dataset.impressions:
        s, e = imp.row_ranges[0]
        if on_policy:
            cov = 1.0  # the logging policy, by construction, always has 100% mass on displayed-at-each-step
        else:
            log_scores = target_policy.log_scores(dataset.plain_phis[s:e])
            p = _softmax(log_scores)
            mask = np.zeros(e - s, dtype=bool)
            mask[: imp.nb_slots] = True
            cov = float(p[mask].sum())
        numer += imp.weight * cov
        denom += imp.weight
    return numer / denom if denom else float("nan")
