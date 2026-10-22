"""Streaming parser for the Criteo counterfactual-learning ranking logs.

File format (one impression = 1 header line + nbCandidates candidate lines):

    example ${exID}: ${hashID} ${wasAdClicked} ${propensity} ${nbSlots} ${nbCandidates} ${featId}:${v} ...
    ${wasProductClicked} exid:${exID} ${featId}:${v} ...        <- repeated nbCandidates times

The first `nbSlots` candidate lines (in file order) are the products that
were actually displayed, left-to-right / top-to-bottom. The logged *action*
is therefore the ordered list of those first `nbSlots` candidates.

Only ~10% of non-clicked impressions were retained in the log (all clicked
impressions were kept). We correct for this by attaching a per-impression
sampling weight `c_i`: 10.0 if `reward == 0`, else 1.0.

This module never loads the whole file into memory: `parse_impressions`
is a generator that yields one `Impression` per `example` block, so it
streams correctly whether the source is a 400-impression sample or the
100M+ impression full dataset (also supports `.gz` transparently).
"""
from __future__ import annotations

import gzip
import io
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, TextIO, Union

NON_CLICK_SUBSAMPLE_RATE = 0.10  # only this fraction of non-clicked impressions were kept
NON_CLICK_WEIGHT = 1.0 / NON_CLICK_SUBSAMPLE_RATE  # c_i = 10.0


@dataclass
class Candidate:
    clicked: int  # wasProductClicked label for this candidate (0/1)
    feats: Dict[int, float]  # product feature id -> value (categorical ids kept as float-coded strings handled upstream)


@dataclass
class Impression:
    exid: int
    hash_id: str
    reward: int  # wasAdClicked, the banner-level click label (0/1)
    propensity: float  # pi_b(logged banner | x), in (0, 1]
    nb_slots: int
    nb_candidates: int
    display_feats: Dict[int, float]
    candidates: List[Candidate]
    weight: float = field(init=False)

    def __post_init__(self) -> None:
        self.weight = 1.0 if self.reward == 1 else NON_CLICK_WEIGHT

    @property
    def displayed(self) -> List[Candidate]:
        """The nbSlots candidates that were actually shown, in slot order."""
        return self.candidates[: self.nb_slots]

    @property
    def not_displayed(self) -> List[Candidate]:
        return self.candidates[self.nb_slots :]


def _open(path: str) -> TextIO:
    if path.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _parse_feat_pairs(tokens: List[str]) -> Dict[int, object]:
    """Parse ``featId:value`` tokens into a dict. Multi-valued categoricals
    (comma-separated values, not observed in the sample but allowed for by
    the spec) are kept as a list of raw value strings; single-valued
    features are kept as a raw string (numeric parsing of features 1/2
    happens downstream in features.py, since whether a feature is numeric
    or categorical is a modeling decision, not a parsing one)."""
    feats: Dict[int, object] = {}
    for tok in tokens:
        fid_str, _, val = tok.partition(":")
        fid = int(fid_str)
        if "," in val:
            feats[fid] = val.split(",")
        else:
            feats[fid] = val
    return feats


def _parse_header(line: str) -> Impression:
    # "example 20184824: hash wasAdClicked propensity nbSlots nbCandidates f:v ..."
    parts = line.split()
    if parts[0] != "example":
        raise ValueError(f"expected header line to start with 'example', got: {line[:60]!r}")
    exid = int(parts[1].rstrip(":"))
    hash_id = parts[2]
    reward = int(parts[3])
    propensity = float(parts[4])
    nb_slots = int(parts[5])
    nb_candidates = int(parts[6])
    display_feats = _parse_feat_pairs(parts[7:])

    if not (0.0 < propensity <= 1.0):
        raise ValueError(f"exid={exid}: propensity {propensity} not in (0, 1]")
    if nb_slots > nb_candidates:
        raise ValueError(f"exid={exid}: nbSlots ({nb_slots}) > nbCandidates ({nb_candidates})")
    if nb_slots < 1:
        raise ValueError(f"exid={exid}: nbSlots must be >= 1, got {nb_slots}")

    return Impression(
        exid=exid,
        hash_id=hash_id,
        reward=reward,
        propensity=propensity,
        nb_slots=nb_slots,
        nb_candidates=nb_candidates,
        display_feats=display_feats,
        candidates=[],
    )


def _parse_candidate_line(line: str, expected_exid: int) -> Candidate:
    parts = line.split()
    clicked = int(parts[0])
    exid_tok = parts[1]
    if not exid_tok.startswith("exid:"):
        raise ValueError(f"expected 'exid:...' as second token, got {exid_tok!r}")
    exid = int(exid_tok.split(":", 1)[1])
    if exid != expected_exid:
        raise ValueError(f"candidate line exid {exid} does not match header exid {expected_exid}")
    feats = _parse_feat_pairs(parts[2:])
    return Candidate(clicked=clicked, feats=feats)


def parse_impressions(path: str) -> Iterator[Impression]:
    """Stream Impression records from `path` ('.txt' or '.txt.gz'). Never
    loads the whole file; holds only the current impression (header +
    up to nbCandidates candidate lines, typically a few dozen) in memory."""
    with _open(path) as f:
        current: Union[Impression, None] = None
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if not line:
                continue
            if line.startswith("example "):
                if current is not None:
                    _validate_complete(current)
                    yield current
                current = _parse_header(line)
            else:
                if current is None:
                    raise ValueError("candidate line encountered before any header line")
                current.candidates.append(_parse_candidate_line(line, current.exid))
        if current is not None:
            _validate_complete(current)
            yield current


def _validate_complete(impression: Impression) -> None:
    if len(impression.candidates) != impression.nb_candidates:
        raise ValueError(
            f"exid={impression.exid}: expected {impression.nb_candidates} candidate lines, "
            f"got {len(impression.candidates)}"
        )


def summarize(path: str, max_impressions: Union[int, None] = None) -> Dict[str, object]:
    """Stream through the file once and print/return summary statistics:
    impression count, raw CTR, 10x-subsampling-corrected CTR, slot-count
    distribution, propensity distribution, and number of distinct banner
    types (unique (feat1, feat2, feat3, feat5) combos, per the spec's
    'banner type' definition)."""
    n_impressions = 0
    n_clicked = 0
    weighted_clicked = 0.0
    weighted_total = 0.0
    slot_counts: Dict[int, int] = {}
    propensities: List[float] = []
    banner_types = set()

    for imp in parse_impressions(path):
        if max_impressions is not None and n_impressions >= max_impressions:
            break
        n_impressions += 1
        n_clicked += imp.reward
        weighted_clicked += imp.weight * imp.reward
        weighted_total += imp.weight
        slot_counts[imp.nb_slots] = slot_counts.get(imp.nb_slots, 0) + 1
        propensities.append(imp.propensity)
        banner_type = tuple(imp.display_feats.get(fid) for fid in (1, 2, 3, 5))
        banner_types.add(banner_type)

    import statistics

    raw_ctr = n_clicked / n_impressions if n_impressions else float("nan")
    corrected_ctr = weighted_clicked / weighted_total if weighted_total else float("nan")
    prop_summary = {
        "min": min(propensities) if propensities else float("nan"),
        "max": max(propensities) if propensities else float("nan"),
        "mean": statistics.mean(propensities) if propensities else float("nan"),
        "median": statistics.median(propensities) if propensities else float("nan"),
    }

    result = {
        "n_impressions": n_impressions,
        "n_clicked_raw": n_clicked,
        "raw_ctr": raw_ctr,
        "corrected_ctr": corrected_ctr,
        "slot_count_distribution": dict(sorted(slot_counts.items())),
        "propensity_summary": prop_summary,
        "n_banner_types": len(banner_types),
    }

    print("=== Dataset summary:", path, "===")
    print(f"Impressions: {result['n_impressions']}")
    print(f"Raw CTR (uncorrected for subsampling): {result['raw_ctr']:.4%}")
    print(f"Corrected CTR (10x-weighted non-clicks): {result['corrected_ctr']:.4%}")
    print(f"Slot-count distribution: {result['slot_count_distribution']}")
    print(f"Propensity summary: {result['propensity_summary']}")
    print(f"Distinct banner types (feat 1,2,3,5 combos): {result['n_banner_types']}")
    return result


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "sample.txt"
    summarize(target)
