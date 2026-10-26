import os
import time

import numpy as np
import pytest

from data import find_file, load_config, load_series, read_csv_any


def _snapshot(d):
    return sorted((str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in d.rglob("*"))


def test_m5_dir_is_read_only(cfg, tmp_path):
    m5 = load_config()["m5_dir"]
    from pathlib import Path
    m5 = Path(os.environ.get("M5_DIR", m5))
    if not m5.exists():
        pytest.skip("M5_DIR not available")
    before = _snapshot(m5)
    cfg = dict(cfg)
    cfg["out_dir"] = str(tmp_path)                        # outputs go elsewhere
    ser = load_series(cfg, cfg["series"]["primary"], cache=True)
    assert len(ser) > 1000
    assert (tmp_path / "series").exists()                  # cache written under outputs, not M5_DIR
    assert _snapshot(m5) == before


def test_missing_file_names_searched_path(tmp_path):
    with pytest.raises(FileNotFoundError) as e:
        find_file(tmp_path, "calendar.csv")
    assert str(tmp_path) in str(e.value) and "calendar.csv" in str(e.value)
