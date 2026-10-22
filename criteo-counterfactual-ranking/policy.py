"""Plackett-Luce ranking policy over Criteo candidates.

The logging policy sampled a banner of `nbSlots` products, without
replacement, from a Plackett-Luce (sequential softmax) model defined by
non-negative per-candidate scores. We never see the logging policy's raw
scores -- only the resulting `propensity` (the probability it assigned to
the exact logged banner). We define a *parametric* candidate policy with
the same Plackett-Luce structure:

    s(x, candidate) = exp(theta . phi(x, candidate) / T)      (T = temperature > 0)

    pi_e(banner | x) = prod_{k=1}^{nbSlots}  s(chosen_k) / sum_{j in R_k} s(j)

where `R_k` is the set of candidates not yet placed at slot k (including
the one chosen at slot k). Because the displayed candidates are always the
first `nbSlots` entries of the per-impression candidate list (in slot
order), `R_k = {k, k+1, ..., M-1}` (0-indexed) -- the chosen-so-far prefix
is simply removed from the tail-to-head remaining set.

All probability computations are done in log-space with log-sum-exp for
numerical stability, since scores can span many orders of magnitude.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


class PlackettLucePolicy:
    def __init__(self, dim: int, temperature: float = 1.0, theta: np.ndarray = None, seed: int = 0):
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.dim = dim
        self.T = temperature
        self.theta = np.zeros(dim, dtype=np.float64) if theta is None else np.asarray(theta, dtype=np.float64)
        self.rng = np.random.default_rng(seed)

    @classmethod
    def uniform(cls, dim: int, seed: int = 0) -> "PlackettLucePolicy":
        """All-zero theta => all candidates score exp(0)=1 => uniform PL,
        i.e. a uniformly random permutation/subset at every slot."""
        return cls(dim=dim, temperature=1.0, theta=np.zeros(dim), seed=seed)

    def log_scores(self, candidate_phis: Sequence[Dict[int, float]]) -> np.ndarray:
        """log s(x, candidate) = theta . phi / T for every candidate in the
        impression (order preserved)."""
        out = np.empty(len(candidate_phis), dtype=np.float64)
        for i, phi in enumerate(candidate_phis):
            dot = 0.0
            for idx, val in phi.items():
                dot += self.theta[idx] * val
            out[i] = dot / self.T
        return out

    @staticmethod
    def _logsumexp(x: np.ndarray) -> float:
        m = np.max(x)
        if not np.isfinite(m):
            return m
        return m + np.log(np.sum(np.exp(x - m)))

    def log_prob_logged_banner(self, log_scores: np.ndarray, nb_slots: int) -> float:
        """log pi_e(logged banner | x) given precomputed log-scores for
        ALL candidates (displayed ones occupy indices [0, nb_slots))."""
        total = 0.0
        for k in range(nb_slots):
            remaining = log_scores[k:]
            total += log_scores[k] - self._logsumexp(remaining)
        return total

    def log_prob_all_permutations(self, log_scores: np.ndarray, nb_slots: int) -> Dict[tuple, float]:
        """Exhaustively enumerate every ordered selection of `nb_slots` out
        of len(log_scores) candidates and return their PL log-probabilities.
        Only used for small test cases (sanity check 2: probabilities sum
        to 1) -- exponential in nb_slots, never called in the main pipeline."""
        import itertools

        n = len(log_scores)
        result = {}
        for perm in itertools.permutations(range(n), nb_slots):
            remaining_mask = np.ones(n, dtype=bool)
            logp = 0.0
            # Recompute remaining set explicitly (not relying on prefix
            # structure) so this really is an independent check of the PL
            # formula on arbitrary permutations.
            for chosen in perm:
                remaining_idx = np.where(remaining_mask)[0]
                remaining_scores = log_scores[remaining_idx]
                logp += log_scores[chosen] - self._logsumexp(remaining_scores)
                remaining_mask[chosen] = False
            result[perm] = logp
        return result

    def sample_banner(self, candidate_phis: Sequence[Dict[int, float]], nb_slots: int) -> List[int]:
        """Sample a banner (list of candidate indices, in slot order)
        without replacement from this policy's Plackett-Luce distribution.
        Used only for sanity-checking log_prob_logged_banner, not in the
        main estimation pipeline (which only needs probabilities, not
        samples, since actions are already logged)."""
        log_scores = self.log_scores(candidate_phis)
        remaining = list(range(len(candidate_phis)))
        banner: List[int] = []
        for _ in range(nb_slots):
            scores = log_scores[remaining]
            m = np.max(scores)
            probs = np.exp(scores - m)
            probs /= probs.sum()
            choice_pos = self.rng.choice(len(remaining), p=probs)
            banner.append(remaining[choice_pos])
            del remaining[choice_pos]
        return banner
