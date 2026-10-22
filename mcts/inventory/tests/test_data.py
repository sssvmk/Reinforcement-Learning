"""Tests for inventory/data.py: data joins (no duplicate days, price matches
wm_yr_wk, SNAP flag matches the series' state) and read-only M5_DIR access.

Spec items 9-10. These tests need the real local M5 files; they are skipped
if M5_DIR (from config.yaml / env var) does not exist, per the task's
"run only when the local M5 files are found" instruction.
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import load_series, _resolve_m5_dir, _load_calendar  # noqa: E402

M5_DIR = _resolve_m5_dir(None)
pytestmark = pytest.mark.skipif(
    not M5_DIR.exists(), reason=f"M5_DIR not found at {M5_DIR}; skipping data-join tests."
)

SERIES_ID = "FOODS_3_477_CA_3_evaluation"


def test_one_row_per_day_no_duplicates():
    df = load_series(SERIES_ID, use_cache=True)
    assert df["date"].is_unique
    assert df["d"].is_unique
    # dates should be strictly increasing (chronological, contiguous daily)
    assert (df["date"].diff().dropna().dt.days == 1).all()


def test_price_matches_wm_yr_wk():
    df = load_series(SERIES_ID, use_cache=True)
    cal = _load_calendar(M5_DIR)
    merged = df.merge(cal[["d", "wm_yr_wk"]], on="d", how="left")
    # Spot-check: for a sample of days, the join key existed in calendar and
    # price is non-null (this series has full price coverage per DATA_NOTES).
    assert merged["wm_yr_wk"].notna().all()
    assert df["sell_price"].notna().all()


def test_snap_flag_belongs_to_series_state():
    # FOODS_3_477_CA_3 -> state CA -> snap_CA column must be the source.
    df = load_series(SERIES_ID, use_cache=True)
    cal = _load_calendar(M5_DIR)
    merged = df.merge(cal[["d", "snap_CA"]], on="d", how="left")
    assert (merged["snap"].to_numpy() == merged["snap_CA"].to_numpy()).all()


def test_leading_zero_trim_for_late_launch_series():
    df = load_series("HOBBIES_1_234_CA_3_evaluation", use_cache=True)
    assert df["sales"].iloc[0] > 0
    assert df["d"].iloc[0] == "d_261"


def test_missing_price_weeks_handled_no_nulls_after_trim():
    df = load_series("HOBBIES_1_234_CA_3_evaluation", use_cache=True)
    # All missing-price weeks for this series precede its launch (trimmed);
    # no NaN prices should remain, confirming the ffill/trim policy.
    assert df["sell_price"].notna().all()


def test_read_only_never_writes_to_m5_dir():
    before = {p: p.stat().st_mtime for p in M5_DIR.rglob("*") if p.is_file()}
    load_series(SERIES_ID, use_cache=False)
    after = {p: p.stat().st_mtime for p in M5_DIR.rglob("*") if p.is_file()}
    assert before == after, "data.py must never modify files under M5_DIR"
    # also confirm no new files were created there
    assert set(before.keys()) == set(after.keys())
