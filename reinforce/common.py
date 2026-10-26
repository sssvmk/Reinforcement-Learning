"""Shared helpers: config loading, markdown tables, safe output paths."""
import os
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")


def load_config(path=None):
    cfg = yaml.safe_load(open(path or os.path.join(HERE, "config.yaml")))
    if os.environ.get("TRADING_FILES"):
        cfg["trading_files"] = [p for p in os.environ["TRADING_FILES"].split(";") if p]
    if os.environ.get("M5_DIR"):
        cfg["m5_dir"] = os.environ["M5_DIR"]
    if os.environ.get("QMAX_FREE"):
        cfg["inventory"]["q_max_mult"] = None
    if os.environ.get("TRADING_RETURN"):
        cfg["trading"]["return_source"] = os.environ["TRADING_RETURN"]
    if os.environ.get("FDPG_DIR"):
        cfg["fdpg_dir"] = os.environ["FDPG_DIR"]
    return cfg


def resolve_fdpg_dir(cfg):
    """config/env -> relative to cwd -> relative to parent of cwd. Returns an absolute path or None."""
    name = "Finite-difference policy gradient"
    cands = [cfg.get("fdpg_dir"), os.path.join(os.getcwd(), name), os.path.join(os.path.dirname(os.getcwd()), name),
             os.path.join(os.path.dirname(HERE), name)]
    for c in cands:
        if c and os.path.isdir(c):
            return os.path.abspath(c)
    return None


def outdir(*parts):
    p = os.path.join(OUT, *parts)
    os.makedirs(p, exist_ok=True)
    return p


def md(df, floatfmt=4):
    """Markdown table without needing `tabulate`."""
    cols = list(df.columns)
    rows = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for v in r:
            cells.append(f"{v:.{floatfmt}f}" if isinstance(v, (float, np.floating)) else str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)
