import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# -----------------------
# Load & prep
# -----------------------
df = pd.read_csv("all_results.csv")

# Parse cohort and trial from tags like "rl_trial16_seed3", "ehr_trial0_seed9", "emr_trial9_seed2"
m = df["tag"].str.extract(r"^(?P<cohort>rl|ehr|emr)_(?P<trial>trial\d+)")
df["cohort"] = m["cohort"]
df["trial"]  = m["trial"]
df["trial_key"] = df["cohort"] + "_" + df["trial"]

# Use only rows that matched the pattern
df = df[~df["trial_key"].isna()].copy()

# Order for plotting
TRIALS = [
    "rl_trial16",   # dark blue
    "rl_trial9",    # light blue
    "ehr_trial0",   # green 1
    "ehr_trial16",  # green 2
    "ehr_trial12",  # green 3
    "emr_trial9",   # light orange
]

# Colors (exact tones chosen to match your ask)
COLORS = {
    "rl_trial16":  {"auc": "#0b5394", "acc": "#0b5394", "single": "#0b5394"},  # dark blue
    "rl_trial9":   {"auc": "#6fa8dc", "acc": "#6fa8dc", "single": "#6fa8dc"},  # light blue
    "ehr_trial0":  {"auc": "#38761d", "acc": "#38761d", "single": "#38761d"},  # dark green
    "ehr_trial16": {"auc": "#6aa84f", "acc": "#6aa84f", "single": "#6aa84f"},  # mid green
    "ehr_trial12": {"auc": "#b6d7a8", "acc": "#b6d7a8", "single": "#b6d7a8"},  # light green
    "emr_trial9":  {"auc": "#f6b26b", "acc": "#f6b26b", "single": "#f6b26b"},  # light orange
}
BASELINE_COLOR = "#777777"

# Helper: aggregate mean/std/count by trial_key
def agg_mean_std_n(dataframe, cols):
    g = dataframe.groupby("trial_key", observed=True)[cols]
    return g.mean(), g.std(ddof=1), g.count()

BAR_WIDTH = 0.11  # small bars to fit many
CAPSIZE = 3
USE_SEM = False

def _err(std, n):
    # 95% confidence interval
    return 1.96 * (std / np.sqrt(n))

def _color_for(trial_key, metric, single=False):
    return COLORS[trial_key]["single" if single else ("auc" if metric.endswith("auc") else "acc")]

# ---- first 3 panels: baseline + all trials for AUC/ACC ----
def _plot_with_baseline(ax, df, auc_col, acc_col, title):
    needed = [auc_col, acc_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        ax.text(0.5, 0.5, f"Missing: {', '.join(missing)}", ha="center", va="center")
        ax.axis("off"); return

    # Means by trial
    mean, std, n = agg_mean_std_n(df, needed)

    # x centers for AUC/ACC groups
    groups = ["AUC", "ACC"]
    x_centers = np.arange(len(groups))  # [0, 1]

    # dynamic offsets: baseline + len(TRIALS) bars
    k = 1 + len(TRIALS)
    offsets = np.linspace(- (k-1)/2 * BAR_WIDTH, (k-1)/2 * BAR_WIDTH, k)

    # Baseline: compute from r2r_* versions of each metric across ALL rows
    base_cols = {
        auc_col: auc_col.replace("s2h_", "r2r_").replace("s2r_", "r2r_").replace("r2s_", "r2r_"),
        acc_col: acc_col.replace("s2h_", "r2r_").replace("s2r_", "r2r_").replace("r2s_", "r2r_"),
    }
    base_vals = {c: (df[bc].mean() if bc in df.columns else np.nan) for c, bc in base_cols.items()}
    base_errs = {c: (df[bc].std(ddof=1) if bc in df.columns else np.nan) for c, bc in base_cols.items()}

    # Draw baseline bars (leftmost offset)
    for i, metric in enumerate(needed):
        bars = ax.bar(x_centers[i] + offsets[0], base_vals[metric], yerr=base_errs[metric],
                      width=BAR_WIDTH, color=BASELINE_COLOR, capsize=CAPSIZE,
                      label="r2r baseline" if i == 0 else None, zorder=3)
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)

    # Draw each trial
    for j, trial_key in enumerate(TRIALS, start=1):
        if trial_key not in mean.index:
            continue
        for i, metric in enumerate(needed):
            val  = mean.loc[trial_key, metric]
            ebar = _err(std.loc[trial_key, metric], n.loc[trial_key, metric])
            bars = ax.bar(x_centers[i] + offsets[j], val, yerr=ebar,
                          width=BAR_WIDTH, color=_color_for(trial_key, metric),
                          capsize=CAPSIZE, label=trial_key if (i == 0) else None, zorder=3)
            ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)

    ax.set_xticks(x_centers)
    ax.set_xticklabels(groups)
    ax.set_title(title)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)  # these are AUC/ACC panels
    ax.grid(axis="y", linestyle="--", alpha=0.25, zorder=0)
    ax.legend(frameon=False, ncols=3, fontsize=9)

# ---- single-metric panels (CWC / elapsed_time) across trials ----
def _plot_single_metric(ax, df, col, title, ylabel=None):
    if col not in df.columns:
        ax.text(0.5, 0.5, f"{col} missing", ha="center", va="center")
        ax.axis("off"); return

    mean, std, n = agg_mean_std_n(df, [col])
    # positions for all trials
    xs = np.arange(len(TRIALS))
    vals = [mean.loc[t, col] if t in mean.index else np.nan for t in TRIALS]
    errs = [_err(std.loc[t, col], n.loc[t, col]) if t in std.index else np.nan for t in TRIALS]
    cols = [_color_for(t, col, single=True) for t in TRIALS]

    bars = ax.bar(xs, vals, yerr=errs, width=BAR_WIDTH+.3, color=cols, capsize=CAPSIZE, zorder=3)
    ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(TRIALS, rotation=15, ha="right")
    ax.set_title(title)
    if ylabel: ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.25, zorder=0)

# ---- mem AUC panels: real baseline + each trial's synth ----
def _plot_mem_auc(ax, df, real_col, synth_col, title):
    missing = [c for c in [real_col, synth_col] if c not in df.columns]
    if missing:
        ax.text(0.5, 0.5, f"Missing: {', '.join(missing)}", ha="center", va="center")
        ax.axis("off"); return

    real_mean, real_std = df[real_col].mean(), df[real_col].std(ddof=1)
    mean, std, n = agg_mean_std_n(df, [synth_col])

    xs = [0.0] + list(np.arange(1, 1 + len(TRIALS)))
    vals = [real_mean] + [mean.loc[t, synth_col] if t in mean.index else np.nan for t in TRIALS]
    errs = [real_std] + [_err(std.loc[t, synth_col], n.loc[t, synth_col]) if t in std.index else np.nan for t in TRIALS]
    cols = [BASELINE_COLOR] + [_color_for(t, synth_col, single=True) for t in TRIALS]
    labels = [real_col] + [f"{synth_col} ({t})" for t in TRIALS]

    bars = ax.bar(xs, vals, yerr=errs, width=BAR_WIDTH+.2, color=cols, capsize=CAPSIZE, zorder=3)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_title(title)
    ax.set_ylabel("AUC")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", linestyle="--", alpha=0.25, zorder=0)

# -----------------------
# Figure (4×2)
# -----------------------
fig, axes = plt.subplots(3, 2, figsize=(20, 15))

# 1) Real → Synthetic
_ = _plot_with_baseline(axes[0, 0], df, "r2s_auc", "r2s_acc", "Real → Synthetic")

# 2) Synthetic → Hold-out
_ = _plot_with_baseline(axes[0, 1], df, "s2h_auc", "s2h_acc", "Synthetic → Hold-out")

# 3) Synthetic → Real
_ = _plot_with_baseline(axes[1, 0], df, "s2r_auc", "s2r_acc", "Synthetic → Real")

# 4) CWC
_ = _plot_single_metric(axes[1, 1], df, "cwc", "CWC", "CWC")

# 5) Mem AUC (Unbalanced)
_ = _plot_mem_auc(axes[2, 0], df, "real_mem_auc", "synth_mem_auc", "Mem AUC (Unbalanced)")

# 6) Mem AUC (Balanced)
_ = _plot_mem_auc(axes[2, 1], df, "real_mem_auc_bal", "synth_mem_auc_bal", "Mem AUC (Balanced)")

# 7) Elapsed time
#_ = _plot_single_metric(axes[3, 0], df, "elapsed_time", "Elapsed Time", "seconds")

# 8) Empty panel for symmetry / notes
#axes[3, 1].axis("off")
plt.tight_layout()
plt.savefig("results_ci.png", bbox_inches="tight")
# plt.show()
