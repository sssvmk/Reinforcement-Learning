"""Training diagnostics for one run, and seed-aggregate comparison figures."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _line(ax, df, x, y, label=None, **kw):
    if y in df and df[y].notna().any():
        d = df.dropna(subset=[y])
        ax.plot(d[x], d[y], label=label or y, **kw)
        return True
    return False


def plot_run(df, hist_a, hist_q, out_dir, title):
    fig, ax = plt.subplots(3, 4, figsize=(20, 11))
    ax = ax.ravel()
    x = "train_days"
    _line(ax[0], df, x, "G_mean", "train return (batch)"); _line(ax[0], df, x, "val_Ct", "val Ctilde")
    ax[0].set_title("return / validation cost"); ax[0].legend(fontsize=7)
    _line(ax[1], df, x, "val_obj", "val objective")
    ax[1].set_title("validation objective"); ax[1].legend(fontsize=7)
    _line(ax[2], df, x, "val_skill", "val skill"); ax[2].axhline(.95, ls=":", c="gray"); ax[2].axhline(.99, ls=":", c="gray")
    ax[2].set_ylim(-0.5, 1.3); ax[2].set_title("skill vs classical reference (1 = reference)")
    for k in ("grad_norm", "update_norm"):
        _line(ax[3], df, x, k)
    ax[3].set_yscale("log"); ax[3].set_title("gradient / update norm"); ax[3].legend(fontsize=7)
    _line(ax[4], df, x, "grad_snr", "grad SNR"); ax[4].set_title("gradient signal-to-noise")
    _line(ax[5], df, x, "sigma", "sigma"); _line(ax[5], df, x, "entropy", "entropy")
    ax[5].set_title("exploration"); ax[5].legend(fontsize=7)
    for k in ("value_loss", "critic_loss", "td_error", "expl_var"):
        _line(ax[6], df, x, k)
    ax[6].set_title("critic diagnostics"); ax[6].legend(fontsize=7)
    for k in ("kl", "kl_after", "clip_frac", "backtracks", "accepted", "ratio_max"):
        _line(ax[7], df, x, k)
    ax[7].set_title("trust region / clipping"); ax[7].legend(fontsize=7)
    _line(ax[8], df, x, "eta", "eta (CVaR)"); _line(ax[8], df, x, "m", "m (mean-var)")
    _line(ax[8], df, x, "batch_cvar95", "batch CVaR95")
    ax[8].set_title("risk state"); ax[8].legend(fontsize=7)
    _line(ax[9], df, x, "theta_dist", "|theta - theta_distilled|"); ax[9].set_title("parameter distance")
    if len(hist_q):
        im = ax[10].imshow(hist_q.T, aspect="auto", origin="lower", cmap="viridis")
        ax[10].set_title("q/Lhat by hour over evals"); ax[10].set_ylabel("hour"); plt.colorbar(im, ax=ax[10])
        _line(ax[11], df, x, "cap_share", "cap share"); _line(ax[11], df, x, "zero_share", "zero share")
        _line(ax[11], df, x, "H_val", "hedge ratio val"); ax[11].set_title("action regime"); ax[11].legend(fontsize=7)
    for a in ax:
        a.set_xlabel("training days (rollouts)"); a.grid(alpha=.3)
        if a.get_legend_handles_labels()[0]:
            a.legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "training.png"), dpi=90)
    plt.close(fig)


def plot_compare(curves, out_path, title, ylabel="validation skill", ylim=(-0.2, 1.2)):
    """curves: {label: list of (days array, values array) per seed}. Median line with IQR band."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for lbl, runs in curves.items():
        grid = np.unique(np.concatenate([r[0] for r in runs]))
        M = np.array([np.interp(grid, r[0], r[1]) for r in runs])
        med, lo, hi = np.median(M, 0), np.percentile(M, 25, 0), np.percentile(M, 75, 0)
        ax.plot(grid, med, label=f"{lbl} (n={len(runs)})"); ax.fill_between(grid, lo, hi, alpha=.15)
    ax.axhline(1, ls=":", c="gray"); ax.set_xscale("symlog", linthresh=100)
    ax.set_xlabel("training days (rollouts)"); ax.set_ylabel(ylabel); ax.set_ylim(*ylim)
    ax.set_title(title); ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=100); plt.close(fig)
