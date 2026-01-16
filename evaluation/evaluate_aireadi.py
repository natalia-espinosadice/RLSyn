
import pandas as pd
import matplotlib.pyplot as plt
import os 
from pathlib import Path
import numpy as np 
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedGroupKFold
from numpy.linalg import norm
from sklearn.metrics import (roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score)
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix
import seaborn as sns
import re
import math 
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn import metrics
import csv 

#------------------------------------------------------------RL MISCELLANEOUS--------------------------------------------------------------# 
#plot losses
def plot_rl_losses(out_dir):
    out_dir = Path(out_dir)
    log_pat = re.compile(r"iteration\s+(\d+)\s*\|\s*D LOSS\s*=\s*([-\d.]+)\s*\|\s*G LOSS\s*=\s*([-\d.]+)\s*\|\s*AVG R\s*=?\s*([-\d.]+)")
    rows = []
    with open(out_dir/"losses/output.txt") as fh:
        for ln in fh:
            m = log_pat.search(ln)
            if m: rows.append([int(m[1]), *map(float, m.groups()[1:])])
    df_log = pd.DataFrame(rows, columns=["iter","d_loss","g_loss","avg_reward"]).sort_values("iter")
    if df_log.empty: 
        df_log = pd.DataFrame(columns=["iter","d_loss","g_loss","avg_reward"])
    g_pat = re.compile(r"ITERATION\s+([0-9]+(?:\.[0-9]+)?)\s*\|\s*LOSS PI\s+([-\d.]+)\s*\|\s*LOSS V\s+([-\d.]+)\s*\|\s*ENTROPY\s+([-\d.]+)\s*\|\s*MEAN PEN.*?\s+([-\d.]+)\s*\|\s*TOTAL G LOSS\s+([-\d.]+)")
    g_rows = []
    with open(out_dir/"losses/G_loss.txt") as fh:
        for ln in fh:
            m = g_pat.search(ln)
            if m: g_rows.append([float(x) for x in m.groups()])
    df_g = pd.DataFrame(g_rows, columns=["iter","loss_pi","loss_v","entropy","mean_pen","g_total"])
    if not df_g.empty:
        df_g["iter"] = df_g["iter"].round().astype(int)
        df_g["entropy"] = df_g["entropy"]
        df_g = df_g.groupby("iter", as_index=False).mean()
        df_g["bin25"] = (df_g["iter"]//25)*25
        df_g25 = df_g.groupby("bin25", as_index=False).mean()
    else:
        df_g25 = pd.DataFrame(columns=["bin25","loss_pi","loss_v","entropy","mean_pen","g_total"])
    df50 = df_log.copy()
    if not df50.empty:
        df50["bin50"] = (df50["iter"]//50)*50
        df50 = df50.groupby("bin50", as_index=False).mean()
    fig, axs = plt.subplots(2,2, figsize=(10,7))
    if not df50.empty:
        axs[0,0].plot(df50["bin50"], df50["d_loss"])
        axs[0,0].set_title("D loss")
        axs[0,1].plot(df50["bin50"], df50["g_loss"])
        axs[0,1].set_title("G loss")
        axs[1,0].plot(df50["bin50"], df50["d_loss"], label="D")
        axs[1,0].plot(df50["bin50"], df50["g_loss"], label="G")
        axs[1,0].legend()
        axs[1,0].set_title("D vs G")
    if not df_log.empty:
        axs[1,1].plot(df_log["iter"], df_log["avg_reward"])
        axs[1,1].set_title("Avg reward")
    for ax in axs.ravel(): ax.set_xlabel("iteration")
    ax.grid(True, linewidth=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_dir/"losses/losses.png", dpi=300)
    plt.close(fig)
    fig, axs = plt.subplots(3,2, figsize=(12,8), sharex=True)
    comps = ["loss_pi","loss_v","entropy","mean_pen","g_total"]
    for ax, c in zip(axs.ravel(), comps):
        if not df_g25.empty: ax.plot(df_g25["bin25"], df_g25[c])
        ax.set_title(c)
        ax.grid(True, linewidth=0.3, alpha=0.5)
    axs.ravel()[-1].axis("off")
    axs[-1,0].set_xlabel("iteration")
    axs[-1,1].set_xlabel("iteration")
    fig.suptitle("Generator components (avg every 25 iters)", y=0.98)
    fig.tight_layout()
    fig.savefig(out_dir/"losses/G_loss.png", dpi=300)
    plt.close(fig)

def log_result_RL(RESULTS_CSV, tag, iters, data_size, seed, cwc, overall_score_real_num, overall_score_hold_num,
               overall_score_real_cat, overall_score_hold_cat, overall_score_real, overall_score_hold,
               r2r_auc, r2r_acc, s2h_auc, s2h_acc, s2r_auc, s2r_acc, r2s_auc, r2s_acc, rvs_auc, rvs_acc, real_mem_auc, real_mem_auc_bal, synth_mem_auc, synth_mem_auc_bal, elapsed_time):
    p = Path(RESULTS_CSV)
    p.parent.mkdir(parents=True, exist_ok=True)
    row = {"tag": tag, "iters": iters, "data_size" : data_size, "seed" : seed,
        "cwc": cwc, "overall_real_num" : overall_score_real_num, "overall_hold_num" : overall_score_hold_num,
        "overall_real_cat" : overall_score_real_cat,  "overall_hold_cat": overall_score_hold_cat,
        "overall_real" : overall_score_real, "overall_hold" : overall_score_hold,
        "r2r_auc" : r2r_auc, "r2r_acc" : r2r_acc, "s2h_auc" : s2h_auc, "s2h_acc" : s2h_acc,
        "s2r_auc" : s2r_auc, "s2r_acc" : s2r_acc, "r2s_auc" : r2s_auc, "r2s_acc" : r2s_acc,
        "rvs_auc" : rvs_auc, "rvs_acc" : rvs_acc, 
        "real_mem_auc" : real_mem_auc, "real_mem_auc_bal" : real_mem_auc_bal, "synth_mem_auc" : synth_mem_auc, "synth_mem_auc_bal" : synth_mem_auc_bal,
        "elapsed_time" : elapsed_time,
    }
    write_header = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)

def log_result_RL_search(RESULTS_CSV, tag, iters, data_size, seed, cwc, synth_mem_auc,
               s2h_auc, s2h_acc, r2s_auc, r2s_acc):
    p = Path(RESULTS_CSV)
    p.parent.mkdir(parents=True, exist_ok=True)
    row = {"tag": tag, "iters": iters, "data_size" : data_size, "seed" : seed,
        "cwc": cwc, "synth_mem_auc" : synth_mem_auc,
        "s2h_auc" : s2h_auc, "s2h_acc" : s2h_acc, "r2s_auc" : r2s_auc, "r2s_acc" : r2s_acc,
    }
    write_header = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)

#------------------------------------------------------------EMR MISCELLANEOUS--------------------------------------------------------------# 
#plot losses
def plot_emr_losses(log_file, OUT_DIR):
    pat = re.compile(r"epoch:\s*(\d+),\s*loss\s*=\s*([-\d.]+),\s*w\s*=\s*([-\d.]+)")
    epochs, losses, ws = [], [], []
    with open(log_file) as f:
        for ln in f:
            m = pat.search(ln)
            if m:
                e, l, w = m.groups()
                epochs.append(int(e)); losses.append(float(l)); ws.append(float(w))
    if not epochs:
        raise ValueError("No lines matched in the log.")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 4), sharex=True)
    ax1.plot(epochs, losses, marker="o")
    ax1.set_ylabel("loss")
    ax1.set_title("Loss over Training Epochs")
    ax1.grid(True, alpha=0.3)
    ax2.plot(epochs, ws, marker="o")
    ax2.set_xlabel("epoch")
    ax2.set_title("W over Training Epochs")
    ax2.set_ylabel("w")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/losses.png", dpi=200)
    plt.close(fig)

def make_logger(model_id, trial_dir):
    logfile  = Path(f"{trial_dir}/{model_id}_outs.txt")
    fh = open(logfile, "a", buffering=1)   
    def log(msg, end="\n"): 
        fh.write(msg + end)   
        fh.flush()
    return log, fh 

def log_result_EMR(RESULTS_CSV, tag, iters, data_size, seed, checkpoint, cwc, overall_score_real_num, overall_score_hold_num,
               overall_score_real_cat, overall_score_hold_cat, overall_score_real, overall_score_hold,
               r2r_auc, r2r_acc, s2h_auc, s2h_acc, s2r_auc, s2r_acc, r2s_auc, r2s_acc, rvs_auc, rvs_acc, real_mem_auc, real_mem_auc_bal, synth_mem_auc, synth_mem_auc_bal, elapsed_time):
    p = Path(RESULTS_CSV)
    p.parent.mkdir(parents=True, exist_ok=True)
    row = {"tag": tag, "iters": iters, "data_size" : data_size, "seed" : seed, "checkpoint": checkpoint, 
        "cwc": cwc, "overall_real_num" : overall_score_real_num, "overall_hold_num" : overall_score_hold_num,
        "overall_real_cat" : overall_score_real_cat,  "overall_hold_cat": overall_score_hold_cat,
        "overall_real" : overall_score_real, "overall_hold" : overall_score_hold,
        "r2r_auc" : r2r_auc, "r2r_acc" : r2r_acc, "s2h_auc" : s2h_auc, "s2h_acc" : s2h_acc,
        "s2r_auc" : s2r_auc, "s2r_acc" : s2r_acc, "r2s_auc" : r2s_auc, "r2s_acc" : r2s_acc,
        "rvs_auc" : rvs_auc, "rvs_acc" : rvs_acc, 
        "real_mem_auc" : real_mem_auc, "real_mem_auc_bal" : real_mem_auc_bal, "synth_mem_auc" : synth_mem_auc, "synth_mem_auc_bal" : synth_mem_auc_bal,
        "elapsed_time" : elapsed_time,
    }
    write_header = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)

def log_result_EMR_search(RESULTS_CSV, tag, iters, data_size, seed, checkpoint, cwc, overall_score_real_num, overall_score_hold_num,
               overall_score_real_cat, overall_score_hold_cat, overall_score_real, overall_score_hold, s2h_auc, s2h_acc, r2s_auc, r2s_acc):
    p = Path(RESULTS_CSV)
    p.parent.mkdir(parents=True, exist_ok=True)
    row = {"tag": tag, "iters": iters, "data_size" : data_size, "seed" : seed, "checkpoint": checkpoint, 
        "cwc": cwc, "overall_real_num" : overall_score_real_num, "overall_hold_num" : overall_score_hold_num,
        "overall_real_cat" : overall_score_real_cat,  "overall_hold_cat": overall_score_hold_cat,
        "overall_real" : overall_score_real, "overall_hold" : overall_score_hold,
        "s2h_auc" : s2h_auc, "s2h_acc" : s2h_acc, "r2s_auc" : r2s_auc, "r2s_acc" : r2s_acc,
    }
    write_header = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


#------------------------------------------------------------FIDELITY ANALYSIS------------------------------------------------------------# 
#print out value statistics
def print_stats(EVAL_PATH, df_real, df_syn, df_hold, df_train, NUM_COLS):
    def _block(fp, title, A, B):
        fp.write(f"{title} \n")
        for col in NUM_COLS:
            a, b = A[col], B[col]
            fp.write(f"{col}\nmean: {a.mean():10.3f} | {b.mean():10.3f} | \nstd: {a.std():10.3f} | {b.std():10.3f} | \nmin: {a.min():10.3f} | {b.min():10.3f} | \n"
                f"max: {a.max():10.3f} | {b.max():10.3f} | \nmed: {a.median():10.3f} | {b.median():10.3f} \n\n"
            )
    with open(EVAL_PATH, "w") as f:
        _block(f, "OVERALL STATISTICS (real vs syn)", df_real, df_syn)
        _block(f, "HOLD OUT STATISTICS (hold vs. syn)", df_hold, df_syn)
        _block(f, "TRAIN STATISTICS (train vs. syn)", df_train, df_syn)
        f.write("Checking duplicates:")
        f.write(f"Df real shape: {df_real.shape}\n")
        f.write(f"Df syn shape: {df_syn.shape}\n ")
        f.write(f"Df real after dropping duplicates: {df_real.drop_duplicates().shape[0]}\n")
        f.write(f"Df syn after dropping duplicates: {df_syn.drop_duplicates().shape[0]}\n")
        f.write(f"Df syn + df real after dropping duplicates: {pd.concat([df_real, df_syn]).drop_duplicates().shape[0]}\n")

#examine interfeature correlations
def get_column_wise_correlations(df_real, df_syn, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    real_corr = df_real.corr()
    syn_corr  = df_syn.corr().fillna(0)
    real_corr.to_csv(out / "real_corr_matrix.csv")
    syn_corr.to_csv(out / "syn_corr_matrix.csv")
    n = real_corr.shape[1]
    cwc = norm((real_corr - syn_corr).values, "fro") / (n ** 2) * 1e6
    np.savetxt(out / "column_wise_corr_score.txt", [cwc], header="cwc_score_×1e-6")
    fig, axes = plt.subplots(1, 2, figsize=(25,20), constrained_layout=True)
    for ax, corr, title in zip(axes, (real_corr, syn_corr), ("Real correlation", "Synthetic correlation")):
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_title(title, fontsize=11, pad=6)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(corr.columns, fontsize=7)
    fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.02, pad=0.02, label="Pearson r").ax.tick_params(labelsize=9)
    fig.savefig(out / "corr_heatmaps_real_vs_synth.png", dpi=300)
    plt.close(fig)
    return float(cwc)

def drop_parens(s: str) -> str:
    # remove " ( ... )" chunks (and any leading space before them)
    s = re.sub(r'\s*\([^)]*\)', '', s)
    # turn spaces and slashes into underscores, collapse repeats, trim edges
    s = re.sub(r'[\\/\s]+', '_', s).strip('_')
    return s

#get single feature histograms
def get_histograms(df_real, df_syn, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pos = "Type 2 Diabetes"
    df_real_neg = df_real[df_real[pos] < 0.5]
    df_syn_neg  = df_syn [df_syn [pos] < 0.5]
    df_real_pos = df_real[df_real[pos] >= 0.5]
    df_syn_pos  = df_syn [df_syn [pos] >= 0.5]
    for col in df_real.columns:
        fig, axs = plt.subplots(2, 3, figsize=(15, 10), tight_layout=True, sharey=True)
        axs = axs.ravel()
        panels = [("Overall: real vs syn", [(df_real[col], "real"), (df_syn[col], "synthetic")]),
            ("All groups", [(df_real_pos[col], "real_pos"), (df_real_neg[col], "real_neg"), (df_syn_pos[col], "syn_pos"), (df_syn_neg[col], "syn_neg")]),
            ("POS: real vs syn", [(df_real_pos[col], "real_pos"), (df_syn_pos[col], "syn_pos")]),
            ("NEG: real vs syn", [(df_real_neg[col], "real_neg"), (df_syn_neg[col], "syn_neg")]),
            ("Real: pos vs neg", [(df_real_pos[col], "real_pos"), (df_real_neg[col], "real_neg")]),
            ("Syn: pos vs neg",  [(df_syn_pos[col],  "syn_pos"),  (df_syn_neg[col],  "syn_neg")]),
        ]
        for ax, (title, series) in zip(axs, panels):
            for s, label in series:
                ax.hist(s, bins=30, alpha=0.5, density=True, label=label)
            ax.set_title(title)
            ax.legend()
        fig.suptitle(col)
        fig.savefig(out / f"{drop_parens(col)}_comp.png")
        plt.close(fig)

#pca analysis
def latent_cluster_analysis(df_real, syn_data_list, out_dir, random_state, var_threshold=0.80):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    X_real = df_real.to_numpy()
    fig = plt.figure(figsize=(6, 6))
    mixdev, silh = [], []
    for k in range(2, 12, 2):
        kdir = out_dir / f"num_clusters{k}"
        kdir.mkdir(parents=True, exist_ok=True)
        scores, first = [], (k == 2)
        pca_for_load, Z = None, None
        for sid, df_syn in enumerate(syn_data_list):
            X_syn = df_syn.to_numpy()
            mixed = np.vstack([X_real, X_syn])
            pca_full = PCA(random_state=random_state).fit(mixed)
            if first and sid == 0:
                vr = pca_full.explained_variance_ratio_
                cv = np.cumsum(vr)
                fig = plt.figure(figsize=(6, 4))
                ax = plt.gca()
                ax.bar(range(1, len(vr)+1), vr, alpha=.6, label='Individual PC')
                ax.step(range(1, len(cv)+1), cv, where='mid', linewidth=2, label='Cumulative')
                ax.axhline(var_threshold, color='red', ls='--', label=f'{var_threshold:.0%} thresh')
                ax.set_xlabel('Principal component')
                ax.set_ylabel('Explained variance ratio')
                ax.set_title('PCA – variance explained')
                ax.legend(loc='best')
                fig.tight_layout()
                fig.savefig(out_dir / 'variance_explained.png')
                plt.close(fig)
            n_pc = max(2, np.searchsorted(np.cumsum(pca_full.explained_variance_ratio_), var_threshold) + 1)
            Z = PCA(n_components=n_pc, random_state=random_state).fit_transform(mixed)
            labels = KMeans(n_clusters=k, n_init=20, random_state=random_state).fit(Z).labels_
            real_flag = np.r_[np.ones(len(X_real)), np.zeros(len(X_syn))]
            dev = np.mean([(real_flag[labels == c].mean() - 0.5) ** 2 for c in range(k) if np.any(labels == c)])
            scores.append({'set_id': sid, 'n_components': n_pc, 'score': math.log2(dev + 1e-12), 'raw_dev': dev})
            if Z.shape[1] >= 2:
                fig = plt.figure(figsize=(6, 6))
                plt.scatter(Z[:, 0], Z[:, 1], c=labels, cmap='tab10', s=4, alpha=.6)
                plt.title(f'PCA by cluster (K={k}) – set {sid}')
                plt.xlabel('PC 1')
                plt.ylabel('PC 2')
                plt.tight_layout()
                fig.savefig(kdir / f'pca_clusters_set{sid}.png')
                plt.close(fig)
                if first:
                    fig = plt.figure(figsize=(6, 6))
                    plt.scatter(Z[:len(X_real), 0], Z[:len(X_real), 1], s=4, alpha=.5, label='Real')
                    plt.scatter(Z[len(X_real):, 0], Z[len(X_real):, 1], s=4, alpha=.5, label='Synthetic')
                    plt.title(f'PCA scatter – set {sid}')
                    plt.xlabel('PC 1')
                    plt.ylabel('PC 2')
                    plt.legend()
                    plt.tight_layout()
                    fig.savefig(out_dir / f'pca_real_syn_set{sid}.png')
                    plt.close(fig) 
                    fig = plt.figure(figsize=(6, 6))
                    plt.scatter(Z[len(X_real):, 0], Z[len(X_real):, 1], s=4, alpha=.5, label='Synthetic')
                    plt.title(f'PCA scatter – synthetic only (set {sid})')
                    plt.xlabel('PC 1')
                    plt.ylabel('PC 2')
                    plt.legend()
                    plt.tight_layout()
                    fig.savefig(out_dir / f'pca_synthetic_set{sid}.png')
                    plt.close(fig)
                plt.close(fig)
            if sid == 0 and k > 1: silh.append((k, silhouette_score(Z, labels)))
            pca_for_load = pca_full
        pd.DataFrame(scores).to_csv(kdir / 'latent_cluster_scores.csv', index=False)
        with (kdir / 'cluster_scores.txt').open('w') as fh:
            fh.writelines(f"set {s['set_id']}: score={s['score']:.4f} (PCs={s['n_components']})\n" for s in scores)
        if first:
            load = pd.DataFrame(pca_for_load.components_.T[:, :2], index=df_real.columns, columns=['PC1', 'PC2'])
            with (kdir / 'cluster_scores.txt').open('a') as fh:
                for pc in ['PC1', 'PC2']:
                    top = load[pc].abs().nlargest(10)
                    fh.write(f"\nTop 10 features for {pc}:\n")
                    fh.write(top.to_string(float_format='%.4f') + "\n")
                    fig = plt.figure(figsize=(6, 4))
                    top.sort_values().plot(kind='barh')
                    plt.title(f'{pc} loadings (top 10)')
                    plt.xlabel('|loading|')
                    plt.tight_layout()
                    fig.savefig(out_dir / f'{pc}_loadings.png')
                    plt.close(fig)
            if Z is not None and Z.shape[1] >= 2:
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.scatter(Z[:len(X_real), 0], Z[:len(X_real), 1], s=4, alpha=.4, label='Real')
                ax.scatter(Z[len(X_real):, 0], Z[len(X_real):, 1], s=4, alpha=.4, label='Synthetic')
                for feat, (l1, l2) in load[['PC1', 'PC2']].iterrows():
                    if abs(l1) + abs(l2) < 0.35: continue
                    ax.arrow(0, 0, l1, l2, color='red', alpha=.5, width=.002, head_width=.03)
                    ax.text(l1*1.2, l2*1.2, feat, fontsize=7, color='red')
                ax.set_xlabel('PC 1')
                ax.set_ylabel('PC 2')
                ax.legend()
                plt.tight_layout()
                fig.savefig(out_dir / 'biplot.png')
                plt.close(fig)
        mixdev.append((k, np.mean([s['raw_dev'] for s in scores])))
    ks, devs = map(np.array, zip(*mixdev))
    fig = plt.figure()
    plt.plot(ks, devs, marker='o')
    plt.xlabel('K')
    plt.ylabel('mix dev')
    plt.title('Mix-deviation vs K')
    plt.tight_layout()
    fig.savefig(out_dir / 'mixdev_vs_k.png')
    plt.close(fig)
    if silh:
        a, b = map(np.array, zip(*silh))
        fig = plt.figure()
        plt.plot(a, b, marker='o')
        plt.xlabel('K')
        plt.ylabel('silhouette')
        plt.title('Silhouette vs K')
        plt.tight_layout()
        fig.savefig(out_dir / 'silhouette_vs_k.png')
        plt.close(fig)

#distributional scores by value statistics
def compute_value_stats(df_real, df_hold, df_syn, VAL_PATH, NUM_COLS, CAT_COLS):
    def _score(cols):
        diffs_real, diffs_hold = [], []
        for col in cols:
            rng = max(float(df_real[col].max() - df_real[col].min()), 1e-9)
            real  = dict(mean=df_real[col].mean(),  std=df_real[col].std(),  median=df_real[col].median(),  min=df_real[col].min(),  max=df_real[col].max())
            hold  = dict(mean=df_hold[col].mean(),  std=df_hold[col].std(),  median=df_hold[col].median(),  min=df_hold[col].min(),  max=df_hold[col].max())
            synth = dict(mean=df_syn[col].mean(),   std=df_syn[col].std(),   median=df_syn[col].median(),   min=df_syn[col].min(),   max=df_syn[col].max())
            for k in ("mean","std","median","min","max"):
                diffs_real.append(abs(synth[k] - real[k]) / rng)
                diffs_hold.append(abs(synth[k] - hold[k]) / rng)
        return float(np.mean(diffs_real)), float(np.mean(diffs_hold))
    with open(VAL_PATH, "w") as f:
        real_num, hold_num = _score(NUM_COLS)
        real_cat, hold_cat = _score(CAT_COLS)
        overall_real = (real_cat + real_num) / 2
        overall_hold = (hold_num + hold_cat) / 2
        f.write(
            f"overall_score_real_num={real_num:.4f}\n"
            f"overall_score_hold_num={hold_num:.4f}\n"
            f"overall_score_real_cat={real_cat:.4f}\n"
            f"overall_score_hold_cat={hold_cat:.4f}\n"
            f"overall_score_real={overall_real:.4f}\n"
            f"overall_score_hold={overall_hold:.4f}\n"
        )
    return real_num, hold_num, real_cat, hold_cat, overall_real, overall_hold


#------------------------------------------------------------UTILITY ANALYSIS------------------------------------------------------------# 
#helpers 
def _ensure_dir(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p
def _svm(rs):
    return SVC(kernel="rbf", probability=True, C=1.0, gamma="scale", random_state=rs)
def _interp(mean_fpr, fpr, tpr):
    x = np.interp(mean_fpr, fpr, tpr)
    x[0] = 0.0
    return x
def _append(fp, s):
    with Path(fp).open("a") as f: f.write(s)
def _save_roc(tprs, mean_fpr, aucs, title, path):
    plt.figure(figsize=(6,5))
    for t in tprs: plt.plot(mean_fpr, t, color="grey", alpha=0.3)
    mt, st = np.mean(tprs,0), np.std(tprs,0)
    plt.plot(mean_fpr, mt, label=f"Mean ROC (AUC = {np.mean(aucs):.3f})")
    plt.fill_between(mean_fpr, np.maximum(mt-st,0), np.minimum(mt+st,1), color="blue", alpha=0.2, label="±1 SD")
    plt.plot([0,1],[0,1],"k--",lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
def _save_perm_importance(all_imps, out_dir, title):
    imps = pd.concat(all_imps, ignore_index=True)
    avg = imps.groupby("feature")["importance"].mean().sort_values(ascending=False)
    avg.to_csv(Path(out_dir)/"feature_importance_perm.csv", header=["avg_drop_acc"])
    top20 = avg.head(20)
    plt.figure(figsize=(8,5))
    top20.plot(kind="barh")
    plt.gca().invert_yaxis()
    plt.xlabel("Mean ↓ accuracy (perm)")
    plt.title(f"Top-20 permutation importances\n({title})")
    plt.tight_layout()
    plt.savefig(Path(out_dir)/"feature_importance_top20.png")
    plt.close()
def plot_confusion_matrix(y_true, y_pred, labels, title, out_path):
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    plt.figure(figsize=(5,4))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def train_on_synth_test_on_real(df_syn, df_real, out_dir, seed):
    label_col = "Type 2 Diabetes"
    out_dir = _ensure_dir(out_dir)
    X_train, y_train = df_syn.drop(columns=[label_col]), (df_syn[label_col] >= 0.5).astype(int)
    X_test_full, y_test_full, groups = df_real.drop(columns=[label_col, "patient_id"]), df_real[label_col], df_real["patient_id"]
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    mean_fpr = np.linspace(0,1,100)
    tprs, aucs, accs = [], [], []
    y_true_all, y_pred_all, all_imps = [], [], []
    (out_dir/"results.txt").write_text(f"--------5-FOLD: TRAIN ON SYNTHETIC / TEST ON REAL--------\nTarget distribution (real):\n{y_test_full.value_counts()}\n\n")
    for fold, (_, vidx) in enumerate(sgkf.split(X_test_full, y_test_full, groups=groups), 1):
        Xv, yv = X_test_full.iloc[vidx], y_test_full.iloc[vidx]
        clf = _svm(fold)
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(Xv)[:,1]
        pred = (proba >= 0.5).astype(int)
        y_true_all.extend(yv)
        y_pred_all.extend(pred)
        auc = roc_auc_score(yv, proba)
        acc = accuracy_score(yv, pred)
        prec = precision_score(yv, pred)
        rec = recall_score(yv, pred)
        f1 = f1_score(yv, pred)
        aucs.append(auc)
        accs.append(acc)
        _append(out_dir/"results.txt", f"Fold {fold}  AUC={auc:.4f}  Acc={acc:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  F1={f1:.3f}\n")
        fpr, tpr, _ = roc_curve(yv, proba)
        tprs.append(_interp(mean_fpr, fpr, tpr))
        perm = permutation_importance(clf, Xv, yv, scoring="accuracy", n_repeats=5, n_jobs=-1, random_state=fold)
        all_imps.append(pd.DataFrame({"feature": X_train.columns, "importance": perm.importances_mean, "fold": fold}))
    _append(out_dir/"results.txt", "\nCross-validation summary\nMean AUC : %.4f ± %.4f\nMean Acc : %.3f ± %.3f\n" % (np.mean(aucs), np.std(aucs), np.mean(accs), np.std(accs)))
    _save_roc(tprs, mean_fpr, aucs, "ROC – Train Synth, Test Real", out_dir/"roc_synth2real.png")
    _save_perm_importance(all_imps, out_dir, "SVM – Synth2Real")
    plot_confusion_matrix(y_true_all, y_pred_all, labels=["No Diabetes", "Type 2 Diabetes"], title="Confusion Matrix – Synth Train / Real Test", out_path=out_dir / "confusion_matrix.png")
    return np.mean(aucs), np.mean(accs)


def train_on_real_test_on_synth(df_real, df_syn, out_dir, seed):
    label_col = "Type 2 Diabetes"
    out_dir = _ensure_dir(out_dir)
    X_real, y_real, groups = df_real.drop(columns=[label_col, "patient_id"]), df_real[label_col], df_real["patient_id"]
    X_syn, y_syn = df_syn.drop(columns=[label_col]), (df_syn[label_col] >= 0.5).astype(int)
    sgkf_r = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    skf_s  = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    mean_fpr = np.linspace(0,1,100)
    tprs, aucs, accs = [], [], []
    y_true_all, y_pred_all, all_imps = [], [], []
    (out_dir/"results.txt").write_text(f"--------5-FOLD: TRAIN ON REAL / TEST ON SYNTHETIC (NON-OVERLAPPING)--------\nSynthetic target distribution:\n{y_syn.value_counts()}\n\n")
    for fold, ((r_tr, _), (_, s_val)) in enumerate(zip(sgkf_r.split(X_real, y_real, groups=groups), skf_s.split(X_syn, y_syn)), 1):
        Xtr, ytr, Xte, yte = X_real.iloc[r_tr], y_real.iloc[r_tr], X_syn.iloc[s_val], y_syn.iloc[s_val]
        clf = _svm(fold).fit(Xtr, ytr)
        proba = clf.predict_proba(Xte)[:,1]
        pred = (proba >= 0.5).astype(int)
        y_true_all.extend(yte)
        y_pred_all.extend(pred)
        auc = roc_auc_score(yte, proba)
        acc = accuracy_score(yte, pred)
        prec = precision_score(yte, pred)
        rec = recall_score(yte, pred)
        f1 = f1_score(yte, pred)
        aucs.append(auc)
        accs.append(acc)
        _append(out_dir/"results.txt", f"Fold {fold}  AUC={auc:.4f}  Acc={acc:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  F1={f1:.3f}\n")
        fpr, tpr, _ = roc_curve(yte, proba)
        tprs.append(_interp(mean_fpr, fpr, tpr))
        perm = permutation_importance(clf, Xte, yte, scoring="accuracy", n_repeats=5, n_jobs=-1, random_state=fold)
        all_imps.append(pd.DataFrame({"feature": Xtr.columns, "importance": perm.importances_mean, "fold": fold}))
    _append(out_dir/"results.txt", f"\nCross-validation summary\nMean AUC : {np.mean(aucs):.4f} ± {np.std(aucs):.4f}\nMean Acc : {np.mean(accs):.3f} ± {np.std(accs):.3f}\n")
    _save_roc(tprs, mean_fpr, aucs, "ROC – Train Real, Test Synth", out_dir/"roc_real2synth.png")
    _save_perm_importance(all_imps, out_dir, "SVM – Real2Synth")
    plot_confusion_matrix(y_true_all, y_pred_all, labels=["No Diabetes", "Type 2 Diabetes"], title="Confusion Matrix – Real Train / Synth Test", out_path=out_dir / "confusion_matrix.png")
    return np.mean(aucs), np.mean(accs)


def train_on_hold_test_on_synth(df_hold, df_syn, out_dir, random_state, label_col="Type 2 Diabetes", n_splits=5):
    out_dir = _ensure_dir(out_dir)
    Xh, yh = df_hold.drop(columns=[label_col]), df_hold[label_col].astype(int)
    Xs, ys = df_syn.drop(columns=[label_col]), (df_syn[label_col] >= 0.5).astype(int)
    skf_h = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    skf_s = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    mean_fpr = np.linspace(0,1,100)
    tprs, aucs, accs = [], [], []
    y_true_all, y_pred_all, all_imps = [], [], []
    (out_dir/"results.txt").write_text(f"--------5-FOLD: TRAIN ON HOLD / TEST ON SYNTHETIC--------\nSynthetic target distribution:\n{ys.value_counts()}\n\n")
    for fold, ((h_tr, _), (_, s_val)) in enumerate(zip(skf_h.split(Xh, yh), skf_s.split(Xs, ys)), 1):
        Xtr, ytr, Xte, yte = Xh.iloc[h_tr], yh.iloc[h_tr], Xs.iloc[s_val], ys.iloc[s_val]
        clf = _svm(random_state + fold).fit(Xtr, ytr)
        proba = clf.predict_proba(Xte)[:,1]
        pred = (proba >= 0.5).astype(int)
        y_true_all.extend(yte)
        y_pred_all.extend(pred)
        auc = roc_auc_score(yte, proba)
        acc = accuracy_score(yte, pred)
        prec = precision_score(yte, pred)
        rec = recall_score(yte, pred)
        f1 = f1_score(yte, pred)
        aucs.append(auc)
        accs.append(acc)
        _append(out_dir/"results.txt", f"Fold {fold}  AUC={auc:.4f}  Acc={acc:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  F1={f1:.3f}\n")
        fpr, tpr, _ = roc_curve(yte, proba)
        tprs.append(_interp(mean_fpr, fpr, tpr))
        perm = permutation_importance(clf, Xte, yte, scoring="accuracy", n_repeats=5, n_jobs=-1, random_state=random_state + fold)
        all_imps.append(pd.DataFrame({"feature": Xtr.columns, "importance": perm.importances_mean, "fold": fold}))
    _append(out_dir/"results.txt", f"\nCross-validation summary\nMean AUC : {np.mean(aucs):.4f} ± {np.std(aucs):.4f}\nMean Acc : {np.mean(accs):.3f} ± {np.std(accs):.3f}\n")
    _save_roc(tprs, mean_fpr, aucs, "ROC – Train Hold, Test Synth", out_dir/"roc_hold2synth.png")
    _save_perm_importance(all_imps, out_dir, "SVM – Hold2Synth")
    plot_confusion_matrix(y_true_all, y_pred_all, labels=["No Diabetes", "Type 2 Diabetes"], title="Confusion Matrix – Hold Train / Synth Test", out_path=out_dir / "confusion_matrix.png")
    return np.mean(aucs), np.mean(accs)

def train_on_synth_test_on_hold(df_syn, df_hold, out_dir, random_state, label_col="Type 2 Diabetes", n_splits=5):
    out_dir = _ensure_dir(out_dir)
    Xtr, ytr = df_syn.drop(columns=[label_col]), (df_syn[label_col] >= 0.5).astype(int)
    Xh, yh = df_hold.drop(columns=[label_col]), df_hold[label_col].astype(int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    mean_fpr = np.linspace(0,1,100)
    tprs, aucs, accs = [], [], []
    y_true_all, y_pred_all, all_imps = [], [], []
    (out_dir/"results.txt").write_text(f"--------5-FOLD: TRAIN ON SYNTHETIC / TEST ON HOLD--------\nTarget distribution (hold):\n{yh.value_counts()}\n\n")
    for fold, (_, vidx) in enumerate(skf.split(Xh, yh), 1):
        Xv, yv = Xh.iloc[vidx], yh.iloc[vidx]
        clf = _svm(random_state + fold).fit(Xtr, ytr)
        proba = clf.predict_proba(Xv)[:,1]
        pred = (proba >= 0.5).astype(int)
        y_true_all.extend(yv)
        y_pred_all.extend(pred)
        auc = roc_auc_score(yv, proba)
        acc = accuracy_score(yv, pred)
        prec = precision_score(yv, pred)
        rec = recall_score(yv, pred)
        f1 = f1_score(yv, pred)
        aucs.append(auc)
        accs.append(acc)
        _append(out_dir/"results.txt", f"Fold {fold}  AUC={auc:.4f}  Acc={acc:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  F1={f1:.3f}\n")
        fpr, tpr, _ = roc_curve(yv, proba)
        tprs.append(_interp(mean_fpr, fpr, tpr))
        perm = permutation_importance(clf, Xv, yv, scoring="accuracy", n_repeats=5, n_jobs=-1, random_state=random_state + fold)
        all_imps.append(pd.DataFrame({"feature": Xtr.columns, "importance": perm.importances_mean, "fold": fold}))
    _append(out_dir/"results.txt", "\nCross-validation summary\nMean AUC : %.4f ± %.4f\nMean Acc : %.3f ± %.3f\n" % (np.mean(aucs), np.std(aucs), np.mean(accs), np.std(accs)))
    _save_roc(tprs, mean_fpr, aucs, "ROC – Train Synth, Test Hold", out_dir/"roc_synth2hold.png")
    _save_perm_importance(all_imps, out_dir, "SVM – Synth2Hold")
    plot_confusion_matrix(y_true_all, y_pred_all, labels=["No Diabetes", "Type 2 Diabetes"], title="Confusion Matrix – Synth Train / Hold Test", out_path=out_dir / "confusion_matrix.png")
    return np.mean(aucs), np.mean(accs)

def train_on_real_test_on_real(out_dir, df_real, seed):
    LABEL = "Type 2 Diabetes"
    out_dir = _ensure_dir(out_dir)
    X, y, groups = df_real.drop(columns=[LABEL, "patient_id"]), df_real[LABEL], df_real["patient_id"]
    (out_dir/"results.txt").write_text(f"-------------------------STANDARDIZED UNBALANCED REAL DATA--------------------- \n \nTarget distribution: {y.value_counts()}\n")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    mean_fpr = np.linspace(0,1,100)
    tprs, aucs, accs = [], [], []
    y_true_all, y_pred_all, all_imps = [], [], []
    for fold, (tr, va) in enumerate(sgkf.split(X, y, groups=groups), 1):
        Xtr, Xv, ytr, yv = X.iloc[tr], X.iloc[va], y.iloc[tr], y.iloc[va]
        clf = _svm(fold).fit(pd.DataFrame(Xtr, columns=Xtr.columns), ytr)
        proba = clf.predict_proba(pd.DataFrame(Xv, columns=Xv.columns))[:,1]
        pred = (proba >= 0.5).astype(int)
        y_true_all.extend(yv)
        y_pred_all.extend(pred)
        auc = roc_auc_score(yv, proba)
        acc = accuracy_score(yv, pred)
        prec = precision_score(yv, pred)
        rec = recall_score(yv, pred)
        f1 = f1_score(yv, pred)
        aucs.append(auc)
        accs.append(acc)
        _append(out_dir/"results.txt", f"Fold {fold}  AUC={auc:.4f}  Acc={acc:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  F1={f1:.3f}\n")
        fpr, tpr, _ = roc_curve(yv, proba)
        tprs.append(_interp(mean_fpr, fpr, tpr))
        perm = permutation_importance(clf, Xv, yv, scoring="accuracy", n_repeats=5, n_jobs=-1, random_state=fold)
        all_imps.append(pd.DataFrame({"feature": Xtr.columns, "importance": perm.importances_mean, "fold": fold}))
    _append(out_dir/"results.txt", f"\nCross-validation summary\nMean AUC : {np.mean(aucs):.4f} ± {np.std(aucs):.4f}\nMean Acc : {np.mean(accs):.3f} ± {np.std(accs):.3f}\n")
    _save_roc(tprs, mean_fpr, aucs, "5-fold ROC – Binary Diabetes (SVM)", out_dir/"5foldROC.png")
    _save_perm_importance(all_imps, out_dir, "SVM – Binary Diabetes")
    plot_confusion_matrix(y_true_all, y_pred_all, ["No Diabetes","Type 2 Diabetes"], "Confusion Matrix – Real Overall (5-Fold)", out_dir/"confusion_matrix.png")
    return np.mean(aucs), np.mean(accs)

def classify_real_vs_syn(df_all, out_dir, seed):
    out_dir = _ensure_dir(out_dir)
    X, y = df_all.drop(columns=["is_syn"]), df_all["is_syn"]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    aucs, accs = [], []
    y_true_all, y_pred_all, all_imps = [], [], []
    (out_dir/"results.txt").write_text(f"CLASSIFY REAL VS SYN\nSynthetic target distribution:\n{y.value_counts()}\n\n")
    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
        clf = _svm(fold).fit(Xtr, ytr)
        proba = clf.predict_proba(Xte)[:,1]
        pred = clf.predict(Xte)
        y_true_all.extend(yte)
        y_pred_all.extend(pred)
        auc, acc = roc_auc_score(yte, proba), accuracy_score(yte, pred)
        aucs.append(auc)
        accs.append(acc)
        perm = permutation_importance(clf, Xte, yte, scoring="accuracy", n_repeats=5, n_jobs=-1, random_state=fold)
        all_imps.append(pd.DataFrame({"feature": Xtr.columns, "importance": perm.importances_mean, "fold": fold}))
        _append(out_dir/"results.txt", f"Fold {fold}: AUC={auc:.4f}  Acc={acc:.3f}\n")
    _append(out_dir/"results.txt", f"Mean AUC : {np.mean(aucs):.4f} ± {np.std(aucs):.4f}\nMean Acc : {np.mean(accs):.3f} ± {np.std(accs):.3f}\n")
    _save_perm_importance(all_imps, out_dir, "SVM – Real vs Synth")
    df_cm = pd.DataFrame(confusion_matrix(y_true_all, y_pred_all), index=["Real","Synth"], columns=["Predicted Real","Predicted Synth"])
    plt.figure(figsize=(5,4))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_dir/"confusion_matrix.png")
    plt.close()
    return np.mean(aucs), np.mean(accs)

#------------------------------------------------------------PRIVACY ANALYSIS------------------------------------------------------------# 

#helpers 
def find_replicant(real, fake):
    #a = (square every elemnt in synthetic matrix and sum across features into col vector) + (square and sum real into row vector )
    #results in (n_fake x n_real) matrix where every (i, j) holds fake_i^2 + real_j^2
    a = np.sum(fake ** 2, axis=1).reshape(fake.shape[0], 1) + np.sum(real.T ** 2, axis=0)
    #each entry (i, j) is 2 (fake I, realIj)
    b = np.dot(fake, real.T) * 2
    #| x - y | ^2 = |x|^2 | y|^2 (full squared euclidean distance matrix between every fake real pair)
    distance_matrix = a - b
    #for every real row j, find the closest fake sample (the min over i), take the sqrt to get euclidean distance --> 1D (n_real)
    #the function returns, for every real record, the distance to its nearest synthetic replicant 
    return np.sqrt(np.min(distance_matrix, axis=0))

def each_group(model, batchsize, n_train, n_test, n_cont_col, model_id, train, test, fake, theta):
    distance_train = np.zeros(n_train)
    distance_test = np.zeros(n_test)
    #for the synethic: for each batch slice of data, compute nearest-distance from that real batch to all fake records
    if model_id != 'real':
        steps = np.ceil(n_train / batchsize)
        for i in range(int(steps)):
            distance_train[i * batchsize:(i + 1) * batchsize] = find_replicant(train[i * batchsize:(i + 1) * batchsize], fake)
    #do the same for test 
    steps = np.ceil(n_test / batchsize)
    for i in range(int(steps)):
        distance_test[i * batchsize:(i + 1) * batchsize] = find_replicant(test[i * batchsize:(i + 1) * batchsize], fake)
    #true positives: real-train rows whose nearest synthetic neighbour is within the radius theta 
    n_tp = np.sum(distance_train <= theta) 
    #false negatives: real-train rows not captured inside theta 
    n_fn = n_train - n_tp
    #false positive counts: hold out rows that also fall within theta (attacker thought they were in training)
    n_fp = np.sum(distance_test <= theta) 
    #F1 score
    f1 = n_tp / (n_tp + (n_fp + n_fn) / 2)  
    return f1, n_tp, n_fn, n_fp 

def _append(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f: f.write(text)

def _batched_min_dists(X, fake, batchsize):
    d = np.empty(len(X))
    for i in range(0, len(X), batchsize):
        d[i:i+batchsize] = find_replicant(X[i:i+batchsize], fake)
    return d

def _plot_rates(theta, tpr, fpr, title, path):
    plt.figure(figsize=(5, 4))
    plt.plot(theta, tpr, label="TPR (recall)")
    plt.plot(theta, fpr, label="FPR")
    plt.xlabel("θ (distance threshold)"); plt.ylabel("Rate"); plt.title(title)
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()

def _plot_roc(fpr, tpr, title_tpl, label, path):
    order = np.argsort(fpr); auc_val = metrics.auc(fpr[order], tpr[order])
    plt.figure(figsize=(4.5, 4.5))
    plt.plot(fpr, tpr, label=f"{label} (AUC {auc_val:.3f})")
    plt.plot([0,1],[0,1],"--", lw=0.8, label="random guess")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(title_tpl.format(auc=auc_val))
    plt.legend(loc="lower right"); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()
    return auc_val

def mem_risk(train_df, test_df, synth_df, CAT_COLS, NUM_COLS, OUT_DIR, GLOBAL_SEED): 
    OUT_DIR = Path(OUT_DIR); OUT_DIR.mkdir(parents=True, exist_ok=True)
    BAL_DIR = OUT_DIR / "balanced"; BAL_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "auc.txt").write_text("Mem risk AUCS\n")
    (OUT_DIR / "mem_risk.txt").write_text("Mem risk results\n")
    #just double check all binary cols 0 or 1 
    for df in (train_df, test_df, synth_df):
        df[CAT_COLS] = (df[CAT_COLS] >= 0.5).astype(float)
    ret = {}
    for model_id in ["real", "synth"]:
        _append(OUT_DIR / "mem_risk.txt", f"Model {model_id} results\n")
        thetas = np.round(np.linspace(0.05, 4.7, 94), 2)
        tpr_raw, fpr_raw, tpr_bal, fpr_bal, theta_list = [], [], [], [], []
        train, test = train_df.values, test_df.values
        fake = train.copy() if model_id == "real" else synth_df.values
        n_train, n_test = len(train), len(test)
        n_cont_col, batchsize = len(NUM_COLS), 1000
        for theta in thetas:
            risk, n_tp, n_fn, n_fp = each_group(model_id.lower(), batchsize, n_train, n_test, n_cont_col, model_id, train, test, fake, float(theta))
            adv = (n_tp / n_train) - (n_fp / n_test)
            _append(OUT_DIR / "mem_risk.txt", f"model_id={model_id}   θ={theta:.2f}   Membership-risk F1 = {risk:.3f} TP: {n_tp} FN: {n_fn} FP {n_fp} Advantage: {adv:.3f} \n")
            tpr_raw.append(n_tp / n_train)
            fpr_raw.append(n_fp / n_test)
            theta_list.append(theta)
            n_bal = min(n_train, n_test)
            #balance 
            train_bal = (train_df.sample(n=n_bal, random_state=GLOBAL_SEED) if n_train > n_bal else train_df).values
            test_bal = (test_df .sample(n=n_bal, random_state=GLOBAL_SEED) if n_test  > n_bal else test_df ).values
            d_train = _batched_min_dists(train_bal, fake, batchsize)
            d_test = _batched_min_dists(test_bal,  fake, batchsize)
            tpb = (d_train <= theta).sum()
            fnb = n_bal - tpb
            fpb = (d_test <= theta).sum()
            tpr_bal.append(tpb / n_bal); fpr_bal.append(fpb / n_bal)
            f1b = tpb / (tpb + (fpb + fnb) / 2)
            _append(BAL_DIR / "mem_risk_balanced.txt", f"model_id={model_id} θ={theta:.2f} F1: {f1b:.3f} TP_bal:{tpb} FN_bal:{fnb} FP_bal:{fpb}\n")
        #raw metrics 
        tpr_raw, fpr_raw, th = np.asarray(tpr_raw), np.asarray(fpr_raw), np.asarray(theta_list)
        _plot_rates(th, tpr_raw, fpr_raw, f"TPR / FPR vs θ – {model_id}", OUT_DIR / f"tpr_fpr_vs_theta_{model_id}.png")
        auc_raw = _plot_roc(fpr_raw, tpr_raw, "Membership-Inference ROC (AUC = {auc:.4f})", model_id, OUT_DIR / f"roc_{model_id}.png")
        print(f"[{model_id}] ROC-AUC = {auc_raw:.4f}\n"); _append(OUT_DIR / "auc.txt", f"[{model_id}] ROC-AUC = {auc_raw:.4f}\n")
        ret[f"{model_id}_roc_auc"] = auc_raw
        #balanced metrix 
        tpr_bal, fpr_bal = np.asarray(tpr_bal), np.asarray(fpr_bal)
        _plot_rates(th, tpr_bal, fpr_bal, f"TPR / FPR vs θ BAL – {model_id}", BAL_DIR / f"tpr_fpr_vs_theta_{model_id}.png")
        auc_bal = _plot_roc(fpr_bal, tpr_bal, "Membership-Inference ROC BAL (AUC = {auc:.4f})", model_id, BAL_DIR / f"roc_{model_id}.png")
        print(f"[{model_id}] ROC-AUC BAL = {auc_bal:.4f}"); _append(OUT_DIR / "auc.txt", f"[{model_id}] ROC-AUC BAL = {auc_bal:.4f}\n")
        ret[f"{model_id}_roc_auc_bal"] = auc_bal
        np.savez(BAL_DIR / f"tpr_fpr_vs_theta_{model_id}.npz", theta=th, TPR=tpr_bal, FPR=fpr_bal)
        #alpha  
        ADV_DIR = OUT_DIR / "advantages_at_alpha"; ADV_DIR.mkdir(exist_ok=True)
        for a in range(1, 101):
            alpha = a * 0.01
            idx = np.where(fpr_raw <= alpha)[0]
            if idx.size:
                best = idx[np.argmax(th[idx])]
                adv = tpr_raw[best] - fpr_raw[best]
                ppv = tpr_raw[best] / (tpr_raw[best] + fpr_raw[best])
                _append(ADV_DIR / f"advantage_{model_id}.txt", f"α={alpha:.2%}, θ*={th[best]:.2f}, TPR={tpr_raw[best]:.4f}, FPR={fpr_raw[best]:.4f}, PPV={ppv:.4f}, Advantage={adv:.4f}\n")
            else:
                _append(ADV_DIR / f"advantage_{model_id}.txt", f"No θ achieves FPR ≤ {alpha:.2%}\n")
    return ret

def log_result(RESULTS_CSV, tag, iters, data_size, seed, cwc, 
               r2r_auc, r2r_acc, s2h_auc, s2h_acc, s2r_auc, s2r_acc, r2s_auc, r2s_acc, rvs_auc, rvs_acc, real_mem_auc, real_mem_auc_bal, synth_mem_auc, synth_mem_auc_bal, elapsed_time):
    p = Path(RESULTS_CSV)
    p.parent.mkdir(parents=True, exist_ok=True)
    row = {"tag": tag, "iters": iters, "data_size" : data_size, "seed" : seed,
        "cwc": cwc, 
        "r2r_auc" : r2r_auc, "r2r_acc" : r2r_acc, "s2h_auc" : s2h_auc, "s2h_acc" : s2h_acc,
        "s2r_auc" : s2r_auc, "s2r_acc" : s2r_acc, "r2s_auc" : r2s_auc, "r2s_acc" : r2s_acc,
        "rvs_auc" : rvs_auc, "rvs_acc" : rvs_acc, 
        "real_mem_auc" : real_mem_auc, "real_mem_auc_bal" : real_mem_auc_bal, "synth_mem_auc" : synth_mem_auc, "synth_mem_auc_bal" : synth_mem_auc_bal,
        "elapsed_time" : elapsed_time,
    }
    write_header = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)

def evaluate_model_aireadi(df_real, df_hold, df_train, df_syn, df_hold_norm, df_train_norm, df_syn_norm, df_real_with_patients_norm, OUT_DIR, RESULT_CSV, RUN_NAME, ITERS, DATA_SIZE, SEED, NUM_COLS, CAT_COLS, elapsed_time): 
    #plot losses
    #-----------------FIDELITY-----------------#
    #examine value statistics per feature
    print_stats(f"{OUT_DIR}/value_stats.txt", df_real, df_syn, df_hold, df_train, NUM_COLS) 
    #get single feature histograms
    #get_histograms(df_real, df_syn, f"{OUT_DIR}/histograms")
    #PCA analysis 
    latent_cluster_analysis(df_real_with_patients_norm.drop(columns=['patient_id']), [df_syn_norm], f"{OUT_DIR}/PCA", SEED)
    df_real_with_patients_norm = df_real_with_patients_norm.drop(columns=["total_steps", "total_kcal"])
    df_syn_norm = df_syn_norm.drop(columns=["total_steps", "total_kcal"])
    latent_cluster_analysis(df_real_with_patients_norm.drop(columns=['patient_id']), [df_syn_norm], f"{OUT_DIR}/PCA_no_steps_kcal", SEED)
    #column wise correlations 
    #cwc = get_column_wise_correlations(df_real, df_syn, f"{OUT_DIR}/correlations") 
    #distributional scores by value stats 
    #overall_score_real_num, overall_score_hold_num, overall_score_real_cat, overall_score_hold_cat, overall_score_real, overall_score_hold = compute_value_stats(df_real, df_hold, df_syn, f"{OUT_DIR}/value_stat_analysis.txt", NUM_COLS, CAT_COLS)  
    #-----------------UTILITY-----------------#
    #get hold out (keep same size no matter the train - test split)
    ten_percent = 490
    if len(df_hold_norm) > ten_percent: 
        hold_fraction = ten_percent / len(df_hold_norm)   
        df_hold_norm = df_hold_norm.sample(frac=hold_fraction, random_state=SEED).reset_index(drop=True)
    #synthetic to hold out 

    s2h_auc, s2h_acc = train_on_synth_test_on_hold(df_syn_norm,  df_hold_norm, f"{OUT_DIR}/synth_to_hold2", SEED)
    #real to synthetic 
    r2s_auc, r2s_acc = train_on_real_test_on_synth(df_real_with_patients_norm, df_syn_norm, f"{OUT_DIR}/real_to_synth2", SEED)
    #synthetic to real 
    s2r_auc, s2r_acc = train_on_synth_test_on_real(df_syn_norm, df_real_with_patients_norm, f"{OUT_DIR}/synth_to_real", SEED)
    #real to real 
    r2r_auc, r2r_acc = train_on_real_test_on_real(f"{OUT_DIR}/real_to_real", df_real_with_patients_norm, SEED)
    #classify real vs syn 
    df_real = df_real_with_patients_norm.copy().drop(columns=["patient_id"])
    df_real["is_syn"] = 0
    df_syn = df_syn_norm.copy()
    df_syn["is_syn"] = 1 
    fraction = len(df_real) / len(df_syn)
    if fraction < 1: 
        df_syn = df_syn.sample(frac=fraction, random_state=SEED).reset_index(drop=True)
    elif fraction > 1: 
        new_frac = len(df_syn) / len(df_real) 
        df_real =  df_real.sample(frac=new_frac, random_state=SEED).reset_index(drop=True)
    df_all = pd.concat([df_real, df_syn], ignore_index=True)
    rvs_auc, rvs_acc = classify_real_vs_syn(df_all, f"{OUT_DIR}/real_vs_syn", SEED)
    
    #-----------------PRIVACY---------------------#
    mem_aucs = mem_risk(df_train_norm, df_hold_norm, df_syn_norm, CAT_COLS, NUM_COLS, f"{OUT_DIR}/mem_risk", SEED) 
    #-----------------LOG RESULTS-----------------#
    with open(f"{OUT_DIR}/eval.txt", "a") as f:
        f.write(f"CWC: {cwc}\n"
            f"r2r_auc: {r2r_auc}\nr2r_acc: {r2r_acc}\nr2s_auc: {r2s_auc}\nr2s_acc: {r2s_acc}\ns2r_auc: {s2r_auc}\ns2r_acc: {s2r_acc}\ns2h_auc: {s2h_auc}\ns2h_acc: {s2h_acc}\nrvs_auc: {rvs_auc}\nrvs_acc: {rvs_acc}\n"
            f"real_mem_auc_: {mem_aucs['real_roc_auc']}\nreal_mem_auc_bal: {mem_aucs['real_roc_auc_bal']}\nsynth_mem_auc: {mem_aucs['synth_roc_auc']}\nsynth_mem_auc_bal: {mem_aucs['synth_roc_auc_bal']}\nelapsed_time: {elapsed_time}")
    log_result(RESULT_CSV, RUN_NAME, ITERS, DATA_SIZE, SEED, cwc, r2r_auc, r2r_acc, s2h_auc, s2h_acc, s2r_auc, s2r_acc, r2s_auc, r2s_acc, rvs_auc, rvs_acc, mem_aucs["real_roc_auc"], mem_aucs["real_roc_auc_bal"], mem_aucs["synth_roc_auc"], mem_aucs["synth_roc_auc_bal"], elapsed_time)

if __name__ == '__main__':
    
