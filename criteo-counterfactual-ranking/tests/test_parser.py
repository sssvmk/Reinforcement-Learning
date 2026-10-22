import gzip
import os
import tempfile

import numpy as np
import pytest

from parser import parse_impressions

SAMPLE_CONTENT = """example 100: hashA 0 0.5 2 3 1:10 2:20 3:cat1
0 exid:100 11:a 12:b
1 exid:100 11:c 12:d
0 exid:100 11:e 12:f
example 101: hashB 1 0.25 1 2 1:5 2:7 3:cat2
1 exid:101 11:x 12:y
0 exid:101 11:z 12:w
"""


@pytest.fixture()
def sample_path(tmp_path):
    p = tmp_path / "mini.txt"
    p.write_text(SAMPLE_CONTENT, encoding="utf-8")
    return str(p)


@pytest.fixture()
def sample_gz_path(tmp_path):
    p = tmp_path / "mini.txt.gz"
    with gzip.open(p, "wt", encoding="utf-8") as f:
        f.write(SAMPLE_CONTENT)
    return str(p)


def test_parses_expected_impression_count(sample_path):
    impressions = list(parse_impressions(sample_path))
    assert len(impressions) == 2


def test_header_fields_parsed_correctly(sample_path):
    imp = list(parse_impressions(sample_path))[0]
    assert imp.exid == 100
    assert imp.hash_id == "hashA"
    assert imp.reward == 0
    assert imp.propensity == 0.5
    assert imp.nb_slots == 2
    assert imp.nb_candidates == 3
    assert imp.display_feats[1] == "10"
    assert imp.display_feats[3] == "cat1"


def test_candidate_lines_parsed_and_linked(sample_path):
    imp = list(parse_impressions(sample_path))[0]
    assert len(imp.candidates) == 3
    assert imp.candidates[0].clicked == 0
    assert imp.candidates[1].clicked == 1
    assert imp.candidates[0].feats[11] == "a"


def test_displayed_vs_not_displayed_split(sample_path):
    imp = list(parse_impressions(sample_path))[0]
    assert len(imp.displayed) == imp.nb_slots == 2
    assert len(imp.not_displayed) == 1


def test_subsample_weight_logic(sample_path):
    impressions = list(parse_impressions(sample_path))
    # imp0: reward=0 -> weight should be 10 (1/0.10)
    assert impressions[0].weight == pytest.approx(10.0)
    # imp1: reward=1 -> weight should be 1
    assert impressions[1].weight == pytest.approx(1.0)


def test_gz_support(sample_gz_path):
    impressions = list(parse_impressions(sample_gz_path))
    assert len(impressions) == 2


def test_rejects_bad_propensity(tmp_path):
    bad = "example 1: h 0 0.0 1 1 1:1\n0 exid:1 11:a\n"
    p = tmp_path / "bad.txt"
    p.write_text(bad, encoding="utf-8")
    with pytest.raises(ValueError):
        list(parse_impressions(str(p)))


def test_rejects_slots_greater_than_candidates(tmp_path):
    bad = "example 1: h 0 0.5 5 1 1:1\n0 exid:1 11:a\n"
    p = tmp_path / "bad2.txt"
    p.write_text(bad, encoding="utf-8")
    with pytest.raises(ValueError):
        list(parse_impressions(str(p)))


def test_rejects_wrong_candidate_count(tmp_path):
    bad = "example 1: h 0 0.5 1 2 1:1\n0 exid:1 11:a\n"
    p = tmp_path / "bad3.txt"
    p.write_text(bad, encoding="utf-8")
    with pytest.raises(ValueError):
        list(parse_impressions(str(p)))


def test_rejects_exid_mismatch(tmp_path):
    bad = "example 1: h 0 0.5 1 1 1:1\n0 exid:2 11:a\n"
    p = tmp_path / "bad4.txt"
    p.write_text(bad, encoding="utf-8")
    with pytest.raises(ValueError):
        list(parse_impressions(str(p)))
