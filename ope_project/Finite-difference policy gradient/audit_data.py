"""Stage 0: check data locations, run trading audits and M5 profiling. Writes ./outputs/audit/ only."""
import json
import os
import sys
import numpy as np
import yaml
import data_m5
import data_trading as dt

HERE = os.path.dirname(os.path.abspath(__file__))


def load_config():
    cfg = yaml.safe_load(open(os.path.join(HERE, "config.yaml")))
    if os.environ.get("TRADING_FILES"):
        cfg["trading_files"] = [p for p in os.environ["TRADING_FILES"].split(";") if p]
    if os.environ.get("M5_DIR"):
        cfg["m5_dir"] = os.environ["M5_DIR"]
    return cfg


def main():
    cfg = load_config()
    out = os.path.join(HERE, "outputs", "audit")
    os.makedirs(out, exist_ok=True)
    res = {"trading": {}, "m5": {}}
    for p in cfg["trading_files"]:
        print("trading file:", p, "exists" if os.path.exists(p) else "MISSING")
        df = dt.read_trading_safe(p)
        if df is None:
            continue
        a = dt.audit(df)
        res["trading"][os.path.basename(p)] = a
        print(json.dumps({k: v for k, v in a.items() if k not in ("scale_by_year",)}, indent=1, default=str))
    ok, missing, found = data_m5.status(cfg["m5_dir"])
    print("M5 files found:", sorted(found), "| missing:", missing)
    res["m5"] = {"found": sorted(found), "missing": missing}
    if any(s in found for s in data_m5.SALES) and "calendar.csv" in found:
        prof = data_m5.profile_candidates(found)
        prof.to_csv(os.path.join(out, "m5_profile.csv"), index=False)
        res["m5"]["chosen_series"] = data_m5.choose_series(prof)
        print("candidate series:", res["m5"]["chosen_series"])
    json.dump(res, open(os.path.join(out, "audit.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    sys.exit(main())
