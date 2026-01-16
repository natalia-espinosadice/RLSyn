import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# -----------------------
# Load & parse
# -----------------------
df = pd.read_csv("all_results.csv")

# model kind from tag: rl_mimic_seed3 -> 'rl', ehr_mimic_seed2 -> 'ehr', emr_* -> 'emr'
df["model"] = df["tag"].str.extract(r"^(rl|ehr|emr)")
present_models = [m for m in ["rl", "ehr", "emr"] if m in set(df["model"].dropna())]

# Colors
COLORS = {
    "rl":  {"auc": "#6fa8dc", "acc": "#6fa8dc", "prauc": "#6fa8dc", "single": "#6fa8dc"}, 
    "ehr": {"auc": "#6aa84f", "acc": "#6aa84f", "prauc": "#6aa84f", "single": "#6aa84f"}, 
    "emr": {"auc": "#f6b26b", "acc": "#f6b26b", "prauc": "#f6b26b", "single": "#f6b26b"},  
}
BASELINE_COLOR = "#777777"

BAR_WIDTH = 0.2
CAPSIZE = 3
USE_SEM = False  # set True if you want SEM instead of std

def _err(std, n):
    # 95% confidence interval
    return 1.96 * (std / np.sqrt(n))

# Aggregate helper by model
def agg_mean_std_n(frame, cols):
    g = frame.groupby("model", observed=True)[cols]
    return g.mean(), g.std(ddof=1), g.count()

def _color_for(model, metric, single=False):
    if single:
        return COLORS[model]["single"]
    if metric.endswith("prauc"):
        return COLORS[model]["prauc"]
    if metric.endswith("auc"):
        return COLORS[model]["auc"]
    if metric.endswith("acc"):
        return COLORS[model]["acc"]
    return COLORS[model]["single"]

# -----------------------
# Panel 1 & 2: baseline + per-model bars for AUC/ACC/PRAUC
# -----------------------
def plot_with_baseline(ax, df, auc_col, acc_col, prauc_col, title):
    needed = [auc_col, acc_col, prauc_col]
    have = [c for c in needed if c in df.columns]
    if len(have) < 2:
        ax.text(0.5, 0.5, f"Missing cols for {title}", ha="center", va="center")
        ax.axis("off"); return

    mean, std, n = agg_mean_std_n(df, have)

    # group centers: AUC, ACC, PRAUC (only those present)
    group_labels = []
    col_order = []
    for name, col in [("AUC", auc_col), ("ACC", acc_col), ("PRAUC", prauc_col)]:
        if col in have:
            group_labels.append(name)
            col_order.append(col)
    x_centers = np.arange(len(col_order))

    # Offsets: baseline + one bar per present model
    k = 1 + len(present_models)
    offsets = np.linspace(-(k-1)/2*BAR_WIDTH, (k-1)/2*BAR_WIDTH, k)

    # r2r baselines for each metric group
    base_map = {c: c.replace("s2h_", "r2r_").replace("r2s_", "r2r_") for c in col_order}
    base_vals = [df[base_map[c]].mean() if base_map[c] in df.columns else np.nan for c in col_order]
    base_errs = [df[base_map[c]].std(ddof=1) if base_map[c] in df.columns else np.nan for c in col_order]

    # Draw baseline
    for i, (v, e) in enumerate(zip(base_vals, base_errs)):
        bars = ax.bar(x_centers[i] + offsets[0], v, yerr=e, width=BAR_WIDTH,
                      color=BASELINE_COLOR, capsize=CAPSIZE, label="r2r baseline" if i == 0 else None, zorder=3)
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)

    # Draw per-model
    for j, model in enumerate(present_models, start=1):
        for i, metric in enumerate(col_order):
            if (model in mean.index) and (metric in mean.columns):
                v  = mean.loc[model, metric]
                e  = _err(std.loc[model, metric], n.loc[model, metric])
                bars = ax.bar(x_centers[i] + offsets[j], v, yerr=e, width=BAR_WIDTH,
                              color=_color_for(model, metric), capsize=CAPSIZE,
                              label=model if i == 0 else None, zorder=3)
                ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)

    ax.set_xticks(x_centers)
    ax.set_xticklabels(group_labels)
    ax.set_title(title)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", linestyle="--", alpha=0.25, zorder=0)
    ax.legend(frameon=False, ncols=3, fontsize=9, loc="upper right")

# -----------------------
# Mem AUC panels: real baseline + per-model synth
# -----------------------
def plot_mem_auc(ax, df, real_col, synth_col, title):
    if real_col not in df.columns or synth_col not in df.columns:
        ax.text(0.5, 0.5, f"Missing {real_col}/{synth_col}", ha="center", va="center")
        ax.axis("off"); return

    real_mean, real_std = df[real_col].mean(), df[real_col].std(ddof=1)
    mean, std, n = agg_mean_std_n(df, [synth_col])

    xs = [0.0] + list(np.arange(1, 1+len(present_models)))
    vals = [real_mean] + [mean.loc[m, synth_col] if (m in mean.index) else np.nan for m in present_models]
    errs = [real_std] + [_err(std.loc[m, synth_col], n.loc[m, synth_col]) if (m in std.index) else np.nan
                         for m in present_models]
    cols = [BASELINE_COLOR] + [_color_for(m, synth_col, single=True) for m in present_models]
    labels = [real_col] + [f"{synth_col} ({m})" for m in present_models]

    bars = ax.bar(xs, vals, yerr=errs, width=BAR_WIDTH, color=cols, capsize=CAPSIZE, zorder=3)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_title(title)
    ax.set_ylabel("AUC")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", linestyle="--", alpha=0.25, zorder=0)

# -----------------------
# Single metric panel (CWC)
# -----------------------
def plot_single_metric(ax, df, col, title, ylabel=None):
    if col not in df.columns:
        ax.text(0.5, 0.5, f"Missing {col}", ha="center", va="center")
        ax.axis("off"); return
    mean, std, n = agg_mean_std_n(df, [col])
    xs = np.arange(len(present_models))
    vals = [mean.loc[m, col] if (m in mean.index) else np.nan for m in present_models]
    errs = [_err(std.loc[m, col], n.loc[m, col]) if (m in std.index) else np.nan for m in present_models]
    cols = [_color_for(m, col, single=True) for m in present_models]

    bars = ax.bar(xs, vals, yerr=errs, width=BAR_WIDTH, color=cols, capsize=CAPSIZE, zorder=3)
    ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(present_models, rotation=0)
    ax.set_title(title)
    if ylabel: ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.25, zorder=0)

# -----------------------
# Figure (3x2 -> 5 panels used)
# -----------------------
fig, axes = plt.subplots(3, 2, figsize=(22, 12))

# 1) Real → Synthetic (AUC/ACC/PRAUC) + r2r baseline
plot_with_baseline(axes[0, 0], df, "r2s_auc", "r2s_acc", "r2s_prauc", "Real → Synthetic")

# 2) Synthetic → Hold-out (AUC/ACC/PRAUC) + r2r baseline
plot_with_baseline(axes[0, 1], df, "s2h_auc", "s2h_acc", "s2h_prauc", "Synthetic → Hold-out")

# 3) Mem AUC (Unbalanced)
plot_mem_auc(axes[1, 0], df, "real_mem_auc", "synth_mem_auc", "Mem AUC (Unbalanced)")

# 4) Mem AUC (Balanced)
plot_mem_auc(axes[1, 1], df, "real_mem_auc_bal", "synth_mem_auc_bal", "Mem AUC (Balanced)")

# 5) CWC
plot_single_metric(axes[2, 0], df, "cwc", "CWC", "CWC")

# 6) Empty slot for notes
axes[2, 1].axis("off")
plt.tight_layout() 
plt.savefig("results_ci.png", bbox_inches="tight") 
# plt.show()
