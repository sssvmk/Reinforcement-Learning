"""Feature hashing for the Criteo ranking logs.

Design decisions (documented here and in README.md):

- Features 1 and 2 are numeric. We apply `log1p` to compress their wide,
  right-skewed range (observed min=2, max~1024/768 in the sample) and feed
  the transformed value directly as the weight of a single hashed slot.
  We deliberately do NOT standardize (subtract mean / divide by std),
  because standardization requires a global pass over the data to compute
  mean/variance, which would break single-pass streaming to 100M+ rows.
  `log1p` needs no corpus-level statistics, so it's streaming-safe.
- All other features (3-10 display, 11-35 product) are categorical. Each
  observed (feature_id, value) pair is hashed into a fixed-size vector
  using the hashing trick (Weinberger et al., 2009): an MD5-derived index
  selects the slot, and a second independent hash bit selects a +-1 sign,
  which keeps hash collisions approximately unbiased in expectation.
- Multi-valued categoricals (comma-separated values) are handled as
  multi-hot: each sub-value is hashed and added independently, so a
  feature with k values contributes k hashed slots (unseen in the sample
  but handled for correctness against the spec).
- A candidate's feature vector is the hashed combination of its own
  product features (11-35) and the impression's display features (1-10).
  No explicit display x product interaction features are added, to keep
  the pipeline simple; see README "Limitations" for the trade-off.
- Hashing requires no fitted vocabulary, so `featurize_candidate` is a
  pure function of its inputs: fully streaming-compatible.
"""
from __future__ import annotations

import hashlib
import math
from typing import Dict, Iterable, List, Tuple, Union

import numpy as np
from scipy.sparse import csr_matrix

NUMERIC_FEATURE_IDS = frozenset({1, 2})
DEFAULT_DIM = 2 ** 16  # 65536 hashed slots


def hash_token(token: str, dim: int) -> Tuple[int, float]:
    """Public deterministic (index, sign) for an arbitrary feature token,
    using two independent slices of an MD5 digest (one for the bucket
    index, one bit of a different slice for the sign) -- the standard
    unbiased hashing-trick construction. Exposed so callers (e.g. the
    reward model's slot-position feature in train.py) can hash their own
    synthetic tokens into the same feature space."""
    digest = hashlib.md5(token.encode("utf-8")).digest()
    index = int.from_bytes(digest[:4], "big") % dim
    sign = 1.0 if (digest[4] & 1) == 0 else -1.0
    return index, sign


def _hash_index_and_sign(token: str, dim: int) -> Tuple[int, float]:
    return hash_token(token, dim)


def _feature_tokens(fid: int, value: Union[str, List[str]]) -> Iterable[str]:
    if isinstance(value, list):
        for v in value:
            yield f"{fid}={v}"
    else:
        yield f"{fid}={value}"


def featurize_candidate(
    display_feats: Dict[int, Union[str, List[str]]],
    product_feats: Dict[int, Union[str, List[str]]],
    dim: int = DEFAULT_DIM,
) -> Dict[int, float]:
    """Return a sparse feature vector (slot index -> accumulated weight)
    for one candidate, combining impression-level display features and
    the candidate's own product features."""
    out: Dict[int, float] = {}

    def add(fid: int, value: Union[str, List[str]]) -> None:
        if fid in NUMERIC_FEATURE_IDS:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return
            scaled = math.log1p(max(numeric, 0.0))
            idx, sign = _hash_index_and_sign(f"num:{fid}", dim)
            out[idx] = out.get(idx, 0.0) + sign * scaled
        else:
            for tok in _feature_tokens(fid, value):
                idx, sign = _hash_index_and_sign(tok, dim)
                out[idx] = out.get(idx, 0.0) + sign

    for fid, value in display_feats.items():
        add(fid, value)
    for fid, value in product_feats.items():
        add(fid, value)
    return out


def sparse_dict_to_csr(rows: List[Dict[int, float]], dim: int = DEFAULT_DIM) -> csr_matrix:
    """Stack a list of sparse-feature dicts (as produced by
    `featurize_candidate`) into a single scipy CSR matrix, without ever
    materializing a dense array."""
    indptr = [0]
    indices: List[int] = []
    data: List[float] = []
    for row in rows:
        for idx, val in row.items():
            indices.append(idx)
            data.append(val)
        indptr.append(len(indices))
    return csr_matrix((np.array(data, dtype=np.float64), np.array(indices, dtype=np.int64), np.array(indptr, dtype=np.int64)), shape=(len(rows), dim))
