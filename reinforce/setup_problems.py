"""Builds the Problem objects (inventory series, trading) from the config. Series are read through the ./outputs/series cache."""
import zlib
import numpy as np
import data_m5
import data_trading as dt
import problems as pb
from common import outdir

_CACHE = {}


def seed_rng(*parts):
    return np.random.default_rng([zlib.crc32(str(p).encode()) % (2 ** 31) if not isinstance(p, int) else p for p in parts])


def trading_data(cfg):
    if "trading" not in _CACHE:
        df = dt.read_trading_safe(cfg["trading_files"][0])
        _CACHE["trading"] = dt.build_features(df, cfg["trading"]["warmup_rows"], cfg["trading"]["allow_adjusted_close"],
                                           return_source=cfg["trading"].get("return_source", "close"))
    return _CACHE["trading"]


def get_problem(cfg, domain, key, horizon=None):
    ck = (domain, key, horizon)
    if ck in _CACHE:
        return _CACHE[ck]
    if horizon is not None:
        cfg = {**cfg, domain: {**cfg[domain], "horizon": horizon}}
    if domain == "inventory":
        ok, miss, found = data_m5.status(cfg["m5_dir"])
        if not ok:
            raise RuntimeError("M5 files missing: %s" % miss)
        df = data_m5.load_series(found, key, outdir("series"))
        pr = pb.build_inventory(df, key, cfg, cfg["splits"])
    else:
        pr = pb.build_trading(trading_data(cfg), key, cfg, cfg["splits"])
    _CACHE[ck] = pr
    return pr


def problem_list(cfg, domain):
    if domain == "inventory":
        return list(cfg["inventory"]["series"])
    return ["TECHM"]


POLICIES = {"inventory": ["I1", "I2", "I3"], "trading": ["T1", "T2"]}
CM_POLICIES = {"inventory": ["I1", "I2"], "trading": ["T1"]}
BASELINES = ["none", "loo_const", "loo_time", "linear", "loo_time_std"]
