"""Config loading, path handling and the read-only KAGGLE_DIR check."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.yaml"


def load_config(preset: str | None = None, path: Path | None = None) -> dict:
    path = Path(path) if path else CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    if preset:
        if preset not in cfg["presets"]:
            raise KeyError(f"unknown preset {preset!r}; options: {list(cfg['presets'])} (file: {path})")
        cfg["train"].update(cfg["presets"][preset])
    cfg["preset"] = preset or "full"
    cfg["kaggle_dir"] = str(kaggle_dir(cfg))
    return cfg


def kaggle_dir(cfg: dict) -> Path:
    return Path(os.environ.get("KAGGLE_DIR") or cfg["kaggle_dir"])


def output_dir(cfg: dict, *sub: str, create: bool = True) -> Path:
    p = (ROOT / cfg.get("output_dir", "outputs")).resolve()
    kd = kaggle_dir(cfg).resolve()
    if kd == p or kd in p.parents:
        raise RuntimeError(f"Output dir {p} would be inside KAGGLE_DIR {kd}; refusing to write there.")
    p = p.joinpath(*sub)
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def verify_kaggle_defaults(cfg: dict) -> dict:
    """Read connectx.json (read-only) and return its default configuration. Raises with exact paths."""
    kd = kaggle_dir(cfg)
    spec = kd / "kaggle_environments" / "envs" / "connectx" / "connectx.json"
    if not kd.is_dir():
        raise FileNotFoundError(f"KAGGLE_DIR does not exist: {kd} (set env var KAGGLE_DIR or kaggle_dir in {CONFIG_PATH})")
    if not spec.is_file():
        raise FileNotFoundError(f"Expected ConnectX spec not found: {spec}")
    with open(spec, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {k: v["default"] for k, v in data["configuration"].items()}
