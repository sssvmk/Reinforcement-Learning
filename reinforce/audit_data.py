"""Stage 0: check data locations, run trading audits, M5 profiling and cache the chosen series. Writes ./outputs/ only."""
import json
import os
import sys
import data_m5
import data_trading as dt
from common import load_config, outdir


def main():
    cfg = load_config()
    out = outdir("audit")
    res = {"trading": {}, "m5": {}}
    for p in cfg["trading_files"]:
        print("trading file:", p, "exists" if os.path.exists(p) else "MISSING")
        df = dt.read_trading_safe(p)
        if df is None:
            continue
        a = dt.audit(df)
        res["trading"][os.path.basename(p)] = a
        print(json.dumps({k: v for k, v in a.items() if k != "scale_by_year"}, indent=1, default=str))
    ok, missing, found = data_m5.status(cfg["m5_dir"])
    print("M5 files found:", sorted(found), "| missing:", missing)
    res["m5"] = {"found": sorted(found), "missing": missing}
    if ok:
        prof = data_m5.profile_candidates(found)
        prof.to_csv(os.path.join(out, "m5_profile.csv"), index=False)
        res["m5"]["chosen_by_rule"] = data_m5.choose_series(prof)
        print("series chosen by the profiling rule:", res["m5"]["chosen_by_rule"])
        print("series adopted from FDPG (config):", cfg["inventory"]["series"])
        cache = outdir("series")
        for sid in cfg["inventory"]["series"]:
            df = data_m5.load_series(found, sid, cache)
            print(sid, len(df), "days, price_ok share", float(df.price_ok.mean()))
    json.dump(res, open(os.path.join(out, "audit.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    sys.exit(main())
