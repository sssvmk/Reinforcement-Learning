"""Learning curves (mean validation return over seeds, 95% bootstrap-free band = mean +- 1.96 se) for TRPO and the extensions.
python plot_learning_curves.py            -> outputs/trpo/figures/<set>/<key>__<policy>.png
Read-only on all results.
"""
import glob
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import OUT

SETS = {  # name: (new runs dir, baseline runs dir or None, baseline method filter)
    "main": ("trpo/full", "full", lambda m: m.startswith("reinforce:") or m.startswith("cm:")),
    "neural": ("trpo/neural_full", "neural_full", lambda m: True),
    "qfree": ("trpo/qfree", "qfree", lambda m: True),
    "entropy": ("trpo/ent", "trpo/full", lambda m: m == "trpo:trpo"),
}


def read(d):
    fs = glob.glob(os.path.join(OUT, d, "runs", "runs_*.csv"))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def main():
    for name, (nd, bd, flt) in SETS.items():
        new, old = read(nd), read(bd)
        new = new[new.method != "sensitivity"]
        new = new[new.method.str.startswith("trpo")]
        old = old[old.method.map(flt)]
        allr = pd.concat([new, old], ignore_index=True)
        allr = allr[allr.val_det.abs() < 1e6]  # diverged runs would wreck the scale; counted in the text
        if name == "main":
            allr = allr[allr.method.isin(["trpo:trpo", "trpo:vpg", "reinforce:loo_time", "cm:spsa", "cm:random"])]
        out = os.path.join(OUT, "trpo", "figures", name)
        os.makedirs(out, exist_ok=True)
        for (key, pol), d in allr[allr.policy.isin(new.policy.unique()) & allr.key.isin(new.key.unique())].groupby(["key", "policy"]):
            fig, ax = plt.subplots(figsize=(7, 4.2))
            mins, tops = [], []
            for m, dm in d.groupby("method"):
                g = dm.groupby("episodes").val_det.agg(["mean", "std", "count"]).reset_index()
                se = g["std"].fillna(0) / np.sqrt(g["count"])
                hl = m in ("trpo:trpo", "trpo:trpo_ent")
                ax.plot(g.episodes, g["mean"], label=m, lw=2.4 if hl else 1.2)
                if hl:
                    ax.fill_between(g.episodes, g["mean"] - 1.96 * se, g["mean"] + 1.96 * se, alpha=0.2)
                mins.append(g["mean"].iloc[len(g) // 5:].min())
                tops.append(g["mean"].max())
            lo, top = min(mins), max(tops)
            ax.set_ylim(lo - 0.1 * (top - lo), top + 0.05 * (top - lo))  # clipped after the first 20% of training; early/collapsed values may be off-scale
            ax.set_xlabel("training episodes")
            ax.set_ylabel("validation return (normalised)")
            ax.set_title("%s  %s  [%s]" % (key.replace("_validation", ""), pol, name))
            ax.legend(fontsize=7, ncol=2)
            fig.tight_layout()
            fig.savefig(os.path.join(out, "%s__%s.png" % (key, pol)), dpi=110)
            plt.close(fig)
        print(name, len(os.listdir(out)), "figures")


if __name__ == "__main__":
    main()
