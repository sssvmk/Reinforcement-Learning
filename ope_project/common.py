"""Shared helpers: config, M5 availability check, split/fold definitions, Log container."""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent


def load_config(path=None):
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, "r", encoding="utf8") as f:
        cfg = yaml.safe_load(f)
    env_t = os.environ.get("TRADING_FILES")
    if env_t:
        cfg["paths"]["trading_files"] = [p for p in env_t.split(";") if p.strip()]
    env_m = os.environ.get("M5_DIR")
    if env_m:
        cfg["paths"]["m5_dir"] = env_m
    return cfg


def outputs_dir(cfg):
    p = ROOT / cfg["paths"]["outputs"]
    p.mkdir(parents=True, exist_ok=True)
    return p


M5_NEEDED_ALWAYS = ["calendar.csv", "sell_prices.csv"]
M5_SALES = ["sales_train_validation.csv", "sales_train_evaluation.csv"]


def check_m5(cfg):
    """Return dict(status, folder, found, missing, reason). Never raises, never searches elsewhere."""
    m5 = str(cfg["paths"]["m5_dir"]).strip()
    flag = cfg["inventory"]["enabled"]
    out = dict(status="PENDING", folder=m5, found=[], missing=[], reason="")
    if flag is False or str(flag).lower() == "false":
        out["reason"] = "inventory.enabled is false"
        return out
    if "{" in m5 or "}" in m5:
        out["reason"] = "M5_DIR still contains braces"
        out["missing"] = M5_NEEDED_ALWAYS + ["sales_train_validation.csv or sales_train_evaluation.csv"]
        return out
    folder = Path(m5)
    if not folder.is_dir():
        out["reason"] = "M5_DIR is not a folder"
        out["missing"] = M5_NEEDED_ALWAYS + M5_SALES
        return out
    names = {p.name for p in folder.rglob("*") if p.is_file()}
    zips = [p for p in folder.rglob("*.zip")]
    import zipfile
    for z in zips:
        try:
            with zipfile.ZipFile(z) as zf:
                names |= {Path(n).name for n in zf.namelist()}
        except Exception:
            pass
    for n in M5_NEEDED_ALWAYS:
        (out["found"] if n in names else out["missing"]).append(n)
    sales_found = [n for n in M5_SALES if n in names]
    out["found"] += sales_found
    if not sales_found:
        out["missing"].append("sales_train_validation.csv or sales_train_evaluation.csv")
    if out["missing"]:
        out["reason"] = "needed M5 files missing"
        return out
    out["status"] = "READY"
    return out


@dataclass
class Fold:
    name: str
    train: tuple
    val: tuple
    test: tuple
    ope: tuple = None
    tune: tuple = None


def make_folds(n, cfg):
    """Chronological splits over clean-row indices [0, n). Returns dict name -> Fold."""
    s = cfg["splits"]
    H = cfg["mdp"]["H"]
    folds = {}

    def build(name, a, b, c, d):
        f = Fold(name, (a, b), (b, c), (c, d))
        vlo, vhi = f.val
        cut = vlo + int(round(s["ope_frac_of_val"] * (vhi - vlo)))
        f.ope = (vlo, cut)
        f.tune = (cut, vhi)
        return f

    a = int(round(s["train"] * n))
    b = int(round((s["train"] + s["val"]) * n))
    folds["primary"] = build("primary", 0, a, b, n)
    if cfg["walk_forward"]["enabled"]:
        for k, (te, ve, tt) in enumerate(cfg["walk_forward"]["folds"]):
            folds[f"wf{k+1}"] = build(f"wf{k+1}", 0, int(round(te * n)), int(round(ve * n)), int(round(tt * n)))
    return folds


@dataclass
class Log:
    """Windows of H transitions. raw: (W,H+1,k); act,r,bprob: (W,H)."""
    raw: np.ndarray
    act: np.ndarray
    r: np.ndarray
    bprob: np.ndarray
    win_id: np.ndarray
    start: np.ndarray
    kind: str = ""
    eps_b: float = 0.0
    _phi_cache: dict = field(default_factory=dict, repr=False)

    @property
    def W(self):
        return self.act.shape[0]

    @property
    def H(self):
        return self.act.shape[1]

    def subset(self, idx):
        return Log(self.raw[idx], self.act[idx], self.r[idx], self.bprob[idx], self.win_id[idx],
                   self.start[idx], self.kind, self.eps_b)

    def phi(self, domain, dense=False):
        key = ("dense" if dense else "sparse")
        if key not in self._phi_cache:
            fn = domain.phi_dense if dense else domain.phi
            W, H1, k = self.raw.shape
            X = fn(self.raw.reshape(-1, k)).astype(np.float64)
            self._phi_cache[key] = X.reshape(W, H1, -1)
        return self._phi_cache[key]


_WIN_COUNTER = [0]


def new_window_ids(n):
    ids = np.arange(_WIN_COUNTER[0], _WIN_COUNTER[0] + n)
    _WIN_COUNTER[0] += n
    return ids
