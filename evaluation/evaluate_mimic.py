import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from matplotlib import cm
from matplotlib.colors import ListedColormap
from scipy.spatial import distance
from scipy.stats import wasserstein_distance
from itertools import chain
from numpy import linalg as LA
import sklearn.preprocessing as preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics
import lightgbm as lgb
import math
import os, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, confusion_matrix
import argparse 
import random
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedGroupKFold
from numpy.linalg import norm
from sklearn.metrics import (roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score)
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import csv
import seaborn as sns
from numpy.linalg import norm
from sklearn.metrics import silhouette_score
import copy
from sklearn.model_selection import cross_val_score, cross_validate, KFold
import joblib
import shap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, confusion_matrix
import scipy.stats as ss
import os 


#------------------------------------------------------------MISC FUNCTIONS------------------------------------------------------------# 
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

def log_result_RL_search(RESULTS_CSV, tag, iters, data_size, seed, cwc, r2s_auc, r2s_prauc, r2s_acc, s2h_auc, s2h_prauc, s2h_acc): 
    p = Path(RESULTS_CSV)
    p.parent.mkdir(parents=True, exist_ok=True)
    row = {"tag": tag, "iters": iters, "data_size" : data_size, "seed" : seed,
        "cwc": cwc, "s2h_auc" : s2h_auc, "s2h_prauc" : s2h_prauc, "s2h_acc" : s2h_acc, "r2s_auc" : r2s_auc, "r2s_prauc" : r2s_prauc, "r2s_acc" : r2s_acc,
    }
    write_header = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)

def log_result_RL_MIMIC(RESULTS_CSV, tag, iters, data_size, seed,cwc, ad2d, continuous_w_d, latent_cluster_analysis, mca_dist, mca_tvd_dist, combined_clinical_violations, s2h_auc, s2h_prauc, s2h_acc, r2s_auc, r2s_prauc, r2s_acc, r2r_auc, r2r_prauc, r2r_acc, mem_aucs, elapsed_time):
    p = Path(RESULTS_CSV)
    p.parent.mkdir(parents=True, exist_ok=True)
    row = {"tag": tag, "iters": iters, "data_size" : data_size, "seed" : seed,
        "cwc": cwc, "ad2d": ad2d, "continuous_w_d": continuous_w_d, "latent_cluster_analysis": latent_cluster_analysis, "mca_dist": mca_dist, "mca_tvd_dist": mca_tvd_dist, "combined_clinical_violations": combined_clinical_violations,
        "r2r_auc" : r2r_auc, "r2r_prauc": r2r_prauc, "r2r_acc" : r2r_acc, 
        "s2h_auc" : s2h_auc, "s2h_prauc": s2h_prauc, "s2h_acc" : s2h_acc,
        "r2s_auc" : r2s_auc, "r2s_prauc": r2s_prauc, "r2s_acc" : r2s_acc,
        "mem_auc_real": mem_aucs['real_roc_auc'], "mem_auc_synth": mem_aucs['synth_roc_auc'],
        "elapsed_time" : elapsed_time,
    }
    write_header = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)
#------------------------------------------------------------FIDELITY ANALYSIS------------------------------------------------------------# 

def compute_value_stats(df_train, df_syn, OUT_DIR): 
    os.makedirs(OUT_DIR, exist_ok=True)
    race_cols = ["WHITE", "BLACK", "ASIAN", "HISPANIC", "UN", "OTHER"]
    target_col = "DIE_1y"
    n_real = len(df_train)
    n_syn  = len(df_syn)
    def cnt_pct(df, col, total):
        n = int(df[col].sum())                
        pct = (n / total * 100) if total else 0.0
        return n, pct
    name_w = max(len(c) for c in race_cols)

    with open(f"{OUT_DIR}/value_stat_analysis.txt", "w") as f:
        f.write("DATASET SIZES\n")
        f.write(f"- REAL: {n_real}\n- SYN : {n_syn}\n\n")
        f.write("RACE COLUMNS\n")
        for col in race_cols:
            r_n, r_p = cnt_pct(df_train, col, n_real)
            s_n, s_p = cnt_pct(df_syn,   col, n_syn)
            f.write(f"[{col:{name_w}}]  REAL: {r_n:>6} ({r_p:5.1f}%)   SYN: {s_n:>6} ({s_p:5.1f}%)\n")
        f.write("\nTARGET COLUMN\n")
        pos_r, pos_r_p = cnt_pct(df_train, target_col, n_real)
        neg_r, neg_r_p = n_real - pos_r, 100 - pos_r_p
        f.write(f"REAL  pos: {pos_r:>6} ({pos_r_p:5.1f}%)  |  neg: {neg_r:>6} ({neg_r_p:5.1f}%)\n")
        pos_s, pos_s_p = cnt_pct(df_syn, target_col, n_syn)
        neg_s, neg_s_p = n_syn - pos_s, 100 - pos_s_p
        f.write(f"SYN   pos: {pos_s:>6} ({pos_s_p:5.1f}%)  |  neg: {neg_s:>6} ({neg_s_p:5.1f}%)\n")

def get_column_wise_correlationsM(df_real, df_syn, out_dir, get_matrices):
    nonzero_var_columns = df_real.columns[df_real.std() !=0 ].values
    df_real = df_real[nonzero_var_columns]
    df_syn = df_syn[nonzero_var_columns]

    real_cor = np.corrcoef(np.transpose(df_real.values ))
    noise_matrix = (np.random.rand(len(df_syn.values),len(df_syn.values[0])) - 1) / 100000000    
    syn_cor = np.corrcoef(np.transpose(df_syn.values + noise_matrix))
    n = df_real.shape[1]
    cwc = norm((real_cor - syn_cor), "fro") / (n ** 2) * 1e6
    print("cwc", cwc)

    if get_matrices: 
        os.makedirs(out_dir, exist_ok=True)
        def plot(df_real, df_syn, col, name, size, use_vminmax): 
            df_real = df_real[col]
            df_syn = df_syn[col] 
            real_cor = np.corrcoef(np.transpose(df_real.values ))
            noise_matrix = (np.random.rand(len(df_syn.values),len(df_syn.values[0])) - 1) / 100000000    
            syn_cor = np.corrcoef(np.transpose(df_syn.values + noise_matrix))
            n = df_real.shape[1]

            fig, axes = plt.subplots(1, 2, figsize=size, constrained_layout=True)
            for ax, corr, title in zip(axes, (real_cor, syn_cor), ("Real correlation", "Synthetic correlation")):
                if use_vminmax: 
                    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
                else: 
                    im = ax.imshow(corr, cmap="coolwarm")
                ax.set_title(title, fontsize=11, pad=6)
                ax.set_xticks(range(n))
                ax.set_yticks(range(n))
                ax.set_xticklabels(df_real.columns, rotation=45, ha="right", fontsize=7)
                ax.set_yticklabels(df_real.columns, fontsize=7)
            fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.02, pad=0.02, label="Pearson r").ax.tick_params(labelsize=9)
            fig.savefig(f"{out_dir}/{name}.png", dpi=300)
            plt.close(fig)

        col = ['AGE', 'BMI', 'DIASTOLIC', 'SYSTOLIC', 'WHITE', 'BLACK', 'ASIAN', 'HISPANIC', 'UN', 'OTHER',  'GENDER', 'DIE_1y',]
        plot(df_real, df_syn, col, "corr_continuous_race_gender", (25, 25), True)
        plot(df_real, df_syn, list(df_real.columns), "corr_all",  (50, 50), False)
        for i in range(100, 1500, 75): 
            plot(df_real, df_syn, list(df_real.columns)[i-75:i], f"corr_{i-75}_{i}",  (50, 50), True)
        
        plot(df_real, df_syn, list(df_real.columns)[1400:], f"corr_{1400}_end",  (50, 50), True)
    return cwc 
    

def compute_dimension_wide_distribution(df_train, df_syn, OUT_DIR): 
    os.makedirs(OUT_DIR, exist_ok=True)
    binary_d2d_average = {}
    binary_d2d_sum = {}
    binary_perc_real = []
    condition_columns = list(df_train.columns)[8:-4]
    FEATURE_COUNT = len(df_syn.columns) - 4
    train_data = df_train.values
    for c in range(0,FEATURE_COUNT):
        if c != 6:
            binary_perc_real.append(np.sum(train_data[:,c])/train_data.shape[0])
    print(len(binary_perc_real))

    fig, axs = plt.subplots(figsize = (20,5.5))
    plt.setp(axs, xticks=[0, 1.0, 1], yticks=[0, 1.0, 1])
    # plt.xticks(fontsize=20)
    sd2d_result = []
    ad2d_result = []
    binary_perc_syn = []
    for c in range(0,FEATURE_COUNT):
        if c != 6:
            binary_perc_syn.append(np.sum(df_syn.iloc[:,c])/df_syn.shape[0])
    # define categories for features
    cluster_list= np.array([0]*6 + [2] + [1]*len(condition_columns))
    # calculate sd2d for each synthetic data
    diff = np.abs(np.array(binary_perc_syn) - np.array(binary_perc_real))
    sd2d = diff.sum()
    ad2d = diff.mean() * 1000.0

    for index in [2,1,0]:
        if index == 0:
            marker = "s"
            color = 'black'
            label = 'Race'
        elif index == 1:
            marker = "o"
            color = 'pink'
            label = 'Diagnosis'
        else:
            marker = "d"
            color = 'green'
            label = 'Gender'

        axs.scatter(np.array(binary_perc_real)[cluster_list==index], np.array(binary_perc_syn)[cluster_list==index], color = color, s = 48, label=label,marker = marker,alpha=0.9)
    axs.set_xlabel("Real", fontsize = 20)
    axs.set_ylabel("Synthetic", fontsize = 20)
    axs.set_xlim(0, 1)
    axs.set_ylim(0, 1)
    axs.tick_params(labelsize=20)
    #axs[i,0-1].set_title(f'{syn_list[i]}_{0}', fontsize = 8)
    axs.plot([0, 1], [0, 1], ls="--", c=".1")
    axs.text(0.4, 0.10, f'APD = {ad2d:.2f}', fontsize = 20)
    sd2d_result.append(sd2d)
    ad2d_result.append(ad2d)

    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=15, frameon=False)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=20, frameon=False)
    #fig.savefig(f"{OUT_DIR}/dimension_wide_distribution.eps", bbox_inches='tight',format='eps') 
    fig.savefig(f"{OUT_DIR}/binary_dimension_wide_distribution.png", bbox_inches='tight',format='png') 
    plt.close() 
    #plot continuous 
    continuous_w_d = {}
    continuous_columns = ['AGE', 'BMI', 'DIASTOLIC', 'SYSTOLIC']

    syn_data_list = [df_syn ]
    RUN_list = [1]
    for column in continuous_columns:
        distances_to_real = []

        model_list_figure = ['Real']
        real_values = list(df_train[column])
        data_list = [real_values]
        color_list = ['maroon']
        for run in RUN_list:
            emrwgan_df = pd.DataFrame(syn_data_list[run-1], columns = list(df_train.columns))
            data_list.append(list(emrwgan_df[column]))
            color_list.append('blue')
            model_list_figure.append('EMR-WGAN')
        
        fig = plt.figure(figsize=(20, 3))
        
        print('\n %s' % column)
        for subplot_count in range(len(data_list)):
            if subplot_count > 0:
                distances_to_real.append(wasserstein_distance(real_values, data_list[subplot_count]))
            plt.subplot(1, 6, subplot_count+1)
            # fig = plt.figure(figsize = (3, 5))
            plt.hist(data_list[subplot_count], color = color_list[subplot_count])
            plt.xlabel("Value")
            plt.title(model_list_figure[subplot_count])
            plt.grid(color = 'blue', linestyle = '--', axis = 'y')
        plt.savefig(f"{OUT_DIR}/continuous_dimension_distribution_{column}.png", bbox_inches='tight',format='png')
        plt.close()
        print("EMR-WGAN : mean: %.4f, std: %.4f" % (np.mean(distances_to_real), np.std(distances_to_real)))
        continuous_w_d[column] = distances_to_real
    
    print(continuous_w_d)
    return ad2d, continuous_w_d

def latent_cluster_analysisM(df_train, df_syn, OUT_DIR): 
    os.makedirs(OUT_DIR, exist_ok=True)
    min_max_scaler = preprocessing.MinMaxScaler()
    train_data_tmp = copy.deepcopy(df_train.values)
    train_data_df = pd.DataFrame(train_data_tmp)
    train_data_df.loc[:,[0, 1, 2, 3]] = min_max_scaler.fit_transform(train_data_df[[0, 1, 2, 3]].values)
    NUM_C = 5 ## the number of clusters which has been optimized
    syn_data_list = [df_syn.values]
    log_cluster_score_list = []
    print("Latent deviation: \n")
    for matrix in syn_data_list:
        matrix_df = pd.DataFrame(copy.deepcopy(matrix))
        matrix_df.loc[:,[0, 1, 2, 3]] = min_max_scaler.fit_transform(matrix_df[[0, 1, 2, 3]].values)
        mixed_data = np.concatenate((train_data_df.values,matrix_df.values), axis = 0)
        pca = PCA()
        pca_result = pca.fit_transform(mixed_data)
        sum_diag = np.sum(pca.explained_variance_ratio_)
        i = 1
        while  np.sum(pca.explained_variance_ratio_[:i]) < 0.8:
            i += 1
    #     print(i, np.sum(pca.explained_variance_ratio_[:i])/np.sum(pca.explained_variance_ratio_))
        pca = PCA(n_components=i)
        pca_result = pca.fit_transform(mixed_data)
        
        kmeans_model = KMeans(n_clusters=NUM_C).fit(pca_result)
        cluster_aff = kmeans_model.labels_.tolist()
        real_syn_label = [1]*len(train_data_df.values) + [0]*len(matrix_df.values)
        
        cluster_score_sum = 0
        for label in range(NUM_C):
            indices_label = [i for i in range(len(cluster_aff)) if cluster_aff[i] == label]
            real_syn_for_label = [real_syn_label[i] for i in indices_label]
            ratio = np.sum(real_syn_for_label)/len(real_syn_for_label)
            cluster_score_sum += (ratio - 0.5)**2
        log_cluster_score_list.append(math.log2(cluster_score_sum/NUM_C))
    print(log_cluster_score_list)
    return log_cluster_score_list

def run_PCA(df_real, syn_data_list, out_dir, random_state, var_threshold=0.80):
    os.makedirs(out_dir, exist_ok=True)
    X_real = df_real.to_numpy()
    mixdev, silh = [], []
    for k in range(2, 4, 2):
        kdir = f"{out_dir}/num_clusters{k}"
        os.makedirs(kdir, exist_ok=True)
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
                fig.savefig(f"{out_dir}/variance_explained.png")
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
                fig.savefig(f'{kdir}/pca_clusters_set{sid}.png')
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
                    fig.savefig(f"{out_dir}/pca_real_syn_set{sid}.png")
                plt.close(fig)
            if sid == 0 and k > 1: silh.append((k, silhouette_score(Z, labels)))
            pca_for_load = pca_full

        pd.DataFrame(scores).to_csv(f'{kdir}/latent_cluster_scores.csv', index=False)
        if first:
            load = pd.DataFrame(pca_for_load.components_.T[:, :2], index=df_real.columns, columns=['PC1', 'PC2'])
            #with (f'{kdir}/cluster_scores.txt').open('a') as fh:
            for pc in ['PC1', 'PC2']:
                top = load[pc].abs().nlargest(10)
                #fh.write(f"\nTop 10 features for {pc}:\n")
                #fh.write(top.to_string(float_format='%.4f') + "\n")
                fig = plt.figure(figsize=(6, 4))
                top.sort_values().plot(kind='barh')
                plt.title(f'{pc} loadings (top 10)')
                plt.xlabel('|loading|')
                plt.tight_layout()
                fig.savefig(f'{out_dir}/{pc}_loadings.png')
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
                fig.savefig(f'{out_dir}/biplot.png')
                plt.close(fig)
        mixdev.append((k, np.mean([s['raw_dev'] for s in scores])))
    ''' 
    ks, devs = map(np.array, zip(*mixdev))
    fig = plt.figure()
    plt.plot(ks, devs, marker='o')
    plt.xlabel('K')
    plt.ylabel('mix dev')
    plt.title('Mix-deviation vs K')
    plt.tight_layout()
    fig.savefig(f'{out_dir}/mixdev_vs_k.png')
    plt.close(fig)
    if silh:
        a, b = map(np.array, zip(*silh))
        fig = plt.figure()
        plt.plot(a, b, marker='o')
        plt.xlabel('K')
        plt.ylabel('silhouette')
        plt.title('Silhouette vs K')
        plt.tight_layout()
        fig.savefig(f'{out_dir}/silhouette_vs_k.png')
        plt.close(fig)
    '''    

def medical_concept_abundance(df_train, df_syn, CAT_COLS, out_dir): 
    mca_train_data_df = df_train[CAT_COLS]
    mca_train_data = np.sum(mca_train_data_df.values, axis=1)
    n, bins, patches = plt.hist(x=mca_train_data, bins=20, color='#0504aa', alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    mca_syn_data_list = []
    syn_data_list = [df_syn]
    for matrix in syn_data_list:
        syn_data_df = pd.DataFrame(data = matrix, columns = list(df_train.columns))
        mca_syn_data_df = syn_data_df[CAT_COLS]
        mca_syn_data = np.sum(mca_syn_data_df.values, axis=1).astype(int)
        count_in_bins = {}

        bin_counts = [0]*len(n) 
        for data_point in mca_syn_data:
            bin_number = data_point // (bins[1]-bins[0])
            if bin_number >= len(n):
                bin_counts[-1] += 1
            else:
                bin_counts[int(bin_number)] += 1

        mca_syn_data_list.append(np.sum(np.abs(np.array(bin_counts)-n))*0.5/len(mca_train_data))

    print("Medical concept abundance distances:", mca_syn_data_list)
    tvd = medical_concept_abundance_plot(df_train, df_syn, CAT_COLS, out_dir) 
    return  mca_syn_data_list, tvd 


def medical_concept_abundance_plot(df_train, df_syn, CAT_COLS, out_dir, bins=20, density=False):
    # 0) Align schemas just in case
    os.makedirs(out_dir, exist_ok=True)

    mca_train = df_train[CAT_COLS].sum(axis=1).astype(int).values
    mca_syn   = df_syn[CAT_COLS].sum(axis=1).astype(int).values

    # integer bins covering both datasets; bin width = 1
    max_val = int(max(mca_train.max(initial=0), mca_syn.max(initial=0)))
    bins = np.arange(0, max_val + 2)  # e.g., [0,1,2,...,max_val+1]

    # normalized histograms (PMFs)
    train_counts, _ = np.histogram(mca_train, bins=bins)
    syn_counts,   _ = np.histogram(mca_syn,   bins=bins)
    p_train = train_counts / train_counts.sum() if train_counts.sum() else train_counts
    p_syn   = syn_counts   / syn_counts.sum()   if syn_counts.sum()   else syn_counts

    # Total Variation Distance (0..1)
    tvd = 0.5 * np.abs(p_train - p_syn).sum()

    # Plot as normalized bars
    centers = (bins[:-1] + bins[1:]) / 2
    width = 0.45
    fig, ax = plt.subplots(figsize=(9,5))
    ax.bar(centers - width/2, p_train, width=width, alpha=0.8, label="Real")
    ax.bar(centers + width/2, p_syn,   width=width, alpha=0.8, label="Synthetic")

    ax.set_xlabel("Number of medical concepts per record")
    ax.set_ylabel("Probability")
    ax.set_title("Medical concept abundance (size invariant)")
    ax.legend()
    ax.text(0.98, 0.98, f"TVD = {tvd:.4f}", transform=ax.transAxes, ha="right", va="top")

    plt.tight_layout()
    fig.savefig(f"{out_dir}/medical_concept_abundance.png")
    print(f"Medical concept abundance distance (normalized): {tvd:.6f}")
    return tvd 


def clinical_knowledge_violation(df_train, df_syn, CAT_COLS, out_dir): 
    os.makedirs(out_dir, exist_ok=True)
    disease_male_dict = {}
    disease_female_dict = {}

    gender_column = df_train['GENDER'].tolist()
    for column in CAT_COLS:
        disease_column = df_train[column].tolist()
        patient_positive = [index for index in range(len(disease_column)) if disease_column[index] == 1]
        gender_positive_patient = [gender_column[index] for index in patient_positive]
        if np.sum(gender_positive_patient) == 0: # male
            disease_male_dict[column] = len(patient_positive)
        if np.sum(gender_positive_patient) == len(gender_positive_patient): # female
            disease_female_dict[column] = len(patient_positive)

    sorted_disease_female_dict = sorted(disease_female_dict.items(), key=lambda kv: kv[1], reverse=True)
    sorted_disease_male_dict = sorted(disease_male_dict.items(), key=lambda kv: kv[1], reverse=True)

    # male: 600: Hyperplasia of prostate
    # male: 185: Cancer of prostate
    # male: 605: Erectile dysfunction [ED]

    # female: 649: Other conditions or status of the mother complicating pregnancy, childbirth, or the puerperium
    # female: 655: Known or suspected fetal abnormality affecting management of mother
    # female: 646: Other complications of pregnancy NEC

    print('    Male codes: ')
    with open(f"{out_dir}/clinical_knowledge_violation.txt", "w") as f: 
        f.write(f"Male codes:\n")
        for code in ['600', '185', '605']:
            disease_column = df_train[code].tolist()
            patient_positive = [index for index in range(len(disease_column)) if disease_column[index] == 1]
            gender_column = df_train['GENDER'].tolist()
            gender_positive_patient = [gender_column[index] for index in patient_positive]
            f.write(f"{code}: # total patients = {str(np.sum(disease_column))}; male percentage: {str((len(gender_positive_patient) - np.sum(gender_positive_patient))/len(gender_positive_patient))}; female percentage: {str(np.sum(gender_positive_patient)/len(gender_positive_patient))}\n")
        f.write(f"Female codes:\n")
        for code in ['649', '655', '646']:
            disease_column = df_train[code].tolist()
            patient_positive = [index for index in range(len(disease_column)) if disease_column[index] == 1]
            gender_column = df_train['GENDER'].tolist()
            gender_positive_patient = [gender_column[index] for index in patient_positive]
            f.write(f"code: # total patients: {str(np.sum(disease_column))}; male percentage: {str((len(gender_positive_patient) - np.sum(gender_positive_patient))/len(gender_positive_patient))}; female percentage: {str(np.sum(gender_positive_patient)/len(gender_positive_patient))}\n")

        male_code_violation = []
        female_code_violation = []
        syn_data_list = [df_syn]
        for matrix in syn_data_list:
            syn_data_df = pd.DataFrame(data = matrix, columns = list(df_train.columns))
            f.write('Synthetic data:\n')
            f.write('Male codes: \n')
            male_viol_sum = 0
            for code in ['600', '185', '605']:
                disease_column = syn_data_df[code].tolist()
                patient_positive = [index for index in range(len(disease_column)) if disease_column[index] == 1]
                gender_column = syn_data_df['GENDER'].tolist()
                gender_positive_patient = [gender_column[index] for index in patient_positive]
                f.write(f"{code}: # total patients: {str(np.sum(disease_column))}; male percentage: {str((len(gender_positive_patient) - np.sum(gender_positive_patient))/len(gender_positive_patient))}; female percentage: {str(np.sum(gender_positive_patient)/len(gender_positive_patient))}\n")
                male_viol_sum += np.sum(gender_positive_patient)/len(gender_positive_patient)
            male_code_violation.append(male_viol_sum/3)
            f.write('Female codes: ')
            female_viol_sum = 0
            for code in ['649', '655', '646']:
                disease_column = syn_data_df[code].tolist()
                patient_positive = [index for index in range(len(disease_column)) if disease_column[index] == 1]
                gender_column = syn_data_df['GENDER'].tolist()
                gender_positive_patient = [gender_column[index] for index in patient_positive]
                f.write(f"{code}: # total patients: {str(np.sum(disease_column))}; male percentage: {str((len(gender_positive_patient) - np.sum(gender_positive_patient))/len(gender_positive_patient))}; female percentage: {str(np.sum(gender_positive_patient)/len(gender_positive_patient))}\n")                
                female_viol_sum += (len(gender_positive_patient) - np.sum(gender_positive_patient))/len(gender_positive_patient)
            female_code_violation.append(female_viol_sum/3)
        f.write(f"Clinical knowledge violation: male code violation {male_code_violation}, female code violation {female_code_violation}\n")
        f.write(f"Combined violation: {np.array(male_code_violation) + np.array(female_code_violation)}\n")

        
    print('Clinical knowledge violation: male code violation ', male_code_violation, 'female code violation ', female_code_violation)
    print('Combined violation: ', np.array(male_code_violation) + np.array(female_code_violation))
    
    #the average proportion of wrong-sex patients among those assigned sex-specific diagnoses
    return np.array(male_code_violation) + np.array(female_code_violation)



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
 

def train_and_test_classification(df_train, df_test, out_dir, seed, cat_cols, n_splits=5, label_col="DIE_1y", recall_threshold=0.60):
    os.makedirs(out_dir, exist_ok=True)
    #split
    y_train = df_train[label_col].astype(int)
    y_test  = df_test[label_col].astype(int)
    X_train = df_train.drop(columns=[label_col]).copy()
    X_test = df_test.drop(columns=[label_col]).copy()

    for c in cat_cols:
        X_train[c] = X_train[c].astype("category")
        X_test[c]  = X_test[c].astype("category")

    def make_model():
        return lgb.LGBMClassifier(boosting_type='gbdt', objective='binary', learning_rate=0.01, metric='auc', n_jobs=20, n_estimators=500, colsample_bytree=0.9,
            max_depth=15, num_leaves=50, reg_alpha=1.3, min_split_gain=0.3, subsample=0.9, subsample_freq=40, 
        )

    #KFold CV on df_train 
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cv_auroc, cv_ap = [], []
    for tr_idx, va_idx in kf.split(X_train, y_train):
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]
        model = make_model()
        model.fit(X_tr, y_tr, categorical_feature=cat_cols)
        va_scores = model.predict_proba(X_va)[:, 1]
        cv_auroc.append(roc_auc_score(y_va, va_scores))
        cv_ap.append(average_precision_score(y_va, va_scores))

    #final model on full train -> evaluate on df_test
    final_model = make_model()
    final_model.fit(X_train, y_train, categorical_feature=cat_cols)

    y_scores = final_model.predict_proba(X_test)[:, 1]
    auroc = roc_auc_score(y_test, y_scores)
    ap = average_precision_score(y_test, y_scores)

    #threshold at target recall (with safe fallback)
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    meets = tpr >= recall_threshold
    thres = thresholds[np.argmax(meets)] if np.any(meets) else thresholds[np.argmax(tpr)]
    y_pred = (y_scores >= thres).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0,1]).ravel()
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spes = tn / (tn + fp) if (tn + fp) else 0.0
    acc  = (tn + tp) / max((tn + fp + fn + tp), 1)

    # Write summary
    with open(f"{out_dir}/results.txt", "w") as f:
        f.write("TRAINING DATA:\n")
        f.write(f" Pos rate: {y_train.mean():.3f}\n")
        f.write(f" Num feats: {X_train.shape[1]}, Num rows: {X_train.shape[0]}\n")
        f.write("TESTING DATA:\n")
        f.write(f" Pos rate: {y_test.mean():.3f}\n")
        f.write(f" Num feats: {X_test.shape[1]}, Num rows: {X_test.shape[0]}\n\n")
        f.write(f"KFold (n={n_splits}) CV AUROC: mean={np.mean(cv_auroc):.4f} ± {np.std(cv_auroc):.4f}\n")
        f.write(f"KFold (n={n_splits}) CV PR-AUC: mean={np.mean(cv_ap):.4f} ± {np.std(cv_ap):.4f}\n\n")
        f.write(f"Final threshold @ recall≥{recall_threshold:.2f}: {thres:.4f}\n")
        f.write("TEST METRICS:\n")
        f.write(f" AUROC: {auroc:.4f}\n PRAUC: {ap:.4f}\n ACC: {acc:.4f}\n PPV: {ppv:.4f}\n NPV: {npv:.4f}\n Sensitivity: {sens:.4f}\n Specificity: {spes:.4f}\n")

    def plot_confusion(y_true, y_pred, out_path, title="Confusion Matrix"):
        disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=["No DIE_1y", "DIE_1y"], cmap="Blues", values_format=".0f")
        plt.title(title)
        plt.savefig(out_path, bbox_inches="tight", dpi=200)
        plt.close()

    plot_confusion(y_test, y_pred, out_path=f"{out_dir}/confusion_matrix.png", title=f"Confusion Matrix")
    return { "cv_auroc": cv_auroc,"cv_ap": cv_ap,
        "test": dict(auroc=auroc, prauc=ap, acc=acc, ppv=ppv, npv=npv, sens=sens, spes=spes, thres=thres),
    }



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

def mem_risk_MIMIC(train_df, test_df, synth_df, CAT_COLS, NUM_COLS, OUT_DIR, GLOBAL_SEED): 
    OUT_DIR = Path(OUT_DIR); OUT_DIR.mkdir(parents=True, exist_ok=True)
    BAL_DIR = OUT_DIR / "balanced"; BAL_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "auc.txt").write_text("Mem risk AUCS\n")
    (OUT_DIR / "mem_risk.txt").write_text("Mem risk results\n")
    #just double check all binary cols 0 or 1 
    for df in (train_df, test_df, synth_df):
        df[CAT_COLS] = (df[CAT_COLS] >= 0.5).astype(float)
    ret = {}
    for model_id in ["synth", "real"]: 
        _append(OUT_DIR / "mem_risk.txt", f"Model {model_id} results\n")
        thetas = np.round(np.linspace(0.05, 10.5, 210), 2)
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
            #balanced
            ''' 
            n_bal = min(n_train, n_test) 
            train_bal = (train_df.sample(n=n_bal, random_state=GLOBAL_SEED) if n_train > n_bal else train_df).values
            test_bal = (test_df .sample(n=n_bal, random_state=GLOBAL_SEED) if n_test  > n_bal else test_df ).values
            d_train = _batched_min_dists(train_bal, fake, batchsize)
            d_test = _batched_min_dists(test_bal,  fake, batchsize)
            tpb = (d_train <= theta).sum()
            fnb = n_bal - tpb
            fpb = (d_test <= theta).sum()
            tpr_bal.append(tpb / n_bal)
            fpr_bal.append(fpb / n_bal)
            f1b = tpb / (tpb + (fpb + fnb) / 2)
            
            _append(BAL_DIR / "mem_risk_balanced.txt", f"model_id={model_id} θ={theta:.2f} F1: {f1b:.3f} TP_bal:{tpb} FN_bal:{fnb} FP_bal:{fpb}\n")
            '''
        #raw metrics 
        tpr_raw, fpr_raw, th = np.asarray(tpr_raw), np.asarray(fpr_raw), np.asarray(theta_list)
        _plot_rates(th, tpr_raw, fpr_raw, f"TPR / FPR vs θ – {model_id}", OUT_DIR / f"tpr_fpr_vs_theta_{model_id}.png")
        auc_raw = _plot_roc(fpr_raw, tpr_raw, "Membership-Inference ROC (AUC = {auc:.4f})", model_id, OUT_DIR / f"roc_{model_id}.png")
        print(f"[{model_id}] ROC-AUC = {auc_raw:.4f}\n"); _append(OUT_DIR / "auc.txt", f"[{model_id}] ROC-AUC = {auc_raw:.4f}\n")
        ret[f"{model_id}_roc_auc"] = auc_raw
        #balanced metrix 
        '''
        tpr_bal, fpr_bal = np.asarray(tpr_bal), np.asarray(fpr_bal)
        _plot_rates(th, tpr_bal, fpr_bal, f"TPR / FPR vs θ BAL – {model_id}", BAL_DIR / f"tpr_fpr_vs_theta_{model_id}.png")
        auc_bal = _plot_roc(fpr_bal, tpr_bal, "Membership-Inference ROC BAL (AUC = {auc:.4f})", model_id, BAL_DIR / f"roc_{model_id}.png")
        print(f"[{model_id}] ROC-AUC BAL = {auc_bal:.4f}"); _append(OUT_DIR / "auc.txt", f"[{model_id}] ROC-AUC BAL = {auc_bal:.4f}\n")
        ret[f"{model_id}_roc_auc_bal"] = auc_bal
        np.savez(BAL_DIR / f"tpr_fpr_vs_theta_{model_id}.npz", theta=th, TPR=tpr_bal, FPR=fpr_bal)
        '''
        #alpha  
        ''' 
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
        '''
    return ret


def mem_risk_2only(train_df, test_df, synth_df, CAT_COLS, NUM_COLS, OUT_DIR, GLOBAL_SEED): 
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
        for theta in [2]:
            risk, n_tp, n_fn, n_fp = each_group(model_id.lower(), batchsize, n_train, n_test, n_cont_col, model_id, train, test, fake, float(theta))
            adv = (n_tp / n_train) - (n_fp / n_test)
            _append(OUT_DIR / "mem_risk.txt", f"model_id={model_id}   θ={theta:.2f}   Membership-risk F1 = {risk:.3f} TP: {n_tp} FN: {n_fn} FP {n_fp} Advantage: {adv:.3f} \n")
            tpr_raw.append(n_tp / n_train)
            fpr_raw.append(n_fp / n_test)
            theta_list.append(theta)
            #balanced
            n_bal = min(n_train, n_test) 
            train_bal = (train_df.sample(n=n_bal, random_state=GLOBAL_SEED) if n_train > n_bal else train_df).values
            test_bal = (test_df .sample(n=n_bal, random_state=GLOBAL_SEED) if n_test  > n_bal else test_df ).values
            d_train = _batched_min_dists(train_bal, fake, batchsize)
            d_test = _batched_min_dists(test_bal,  fake, batchsize)
            tpb = (d_train <= theta).sum()
            fnb = n_bal - tpb
            fpb = (d_test <= theta).sum()
            tpr_bal.append(tpb / n_bal)
            fpr_bal.append(fpb / n_bal)
            f1b = tpb / (tpb + (fpb + fnb) / 2)
            _append(BAL_DIR / "mem_risk_balanced.txt", f"model_id={model_id} θ={theta:.2f} F1: {f1b:.3f} TP_bal:{tpb} FN_bal:{fnb} FP_bal:{fpb}\n")
        #raw metrics 
        #tpr_raw, fpr_raw, th = np.asarray(tpr_raw), np.asarray(fpr_raw), np.asarray(theta_list)
        #_plot_rates(th, tpr_raw, fpr_raw, f"TPR / FPR vs θ – {model_id}", OUT_DIR / f"tpr_fpr_vs_theta_{model_id}.png")
        #auc_raw = _plot_roc(fpr_raw, tpr_raw, "Membership-Inference ROC (AUC = {auc:.4f})", model_id, OUT_DIR / f"roc_{model_id}.png")
        #print(f"[{model_id}] ROC-AUC = {auc_raw:.4f}\n"); _append(OUT_DIR / "auc.txt", f"[{model_id}] ROC-AUC = {auc_raw:.4f}\n")
        #ret[f"{model_id}_roc_auc"] = auc_raw
        #balanced metrix 
        tpr_bal, fpr_bal = np.asarray(tpr_bal), np.asarray(fpr_bal)
        #_plot_rates(th, tpr_bal, fpr_bal, f"TPR / FPR vs θ BAL – {model_id}", BAL_DIR / f"tpr_fpr_vs_theta_{model_id}.png")
        #auc_bal = _plot_roc(fpr_bal, tpr_bal, "Membership-Inference ROC BAL (AUC = {auc_bal:.4f})", model_id, BAL_DIR / f"roc_{model_id}.png")
        #print(f"[{model_id}] ROC-AUC BAL = {auc_bal:.4f}"); _append(OUT_DIR / "auc.txt", f"[{model_id}] ROC-AUC BAL = {auc_bal:.4f}\n")
        #ret[f"{model_id}_roc_auc_bal"] = auc_bal
        #np.savez(BAL_DIR / f"tpr_fpr_vs_theta_{model_id}.npz", theta=th, TPR=tpr_bal, FPR=fpr_bal)
        ''' 
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
        '''
    #return ret



def evaluate_model_MIMIC(df_train, df_real, df_syn, df_train_norm, df_hold_norm, df_real_norm, df_syn_norm, H, elapsed_time): 
    #-----------------FIDELITY-----------------#
    #column wise correlations 
    cwc = get_column_wise_correlationsM(df_real, df_syn, f"{H.OUT_DIR}/correlations", True) 
    ad2d, continuous_w_d = compute_dimension_wide_distribution(df_real, df_syn, f"{H.OUT_DIR}/dimension_wide_distributions")
    latent_cluster_analysis = latent_cluster_analysisM(df_real, df_syn, f"{H.OUT_DIR}/PCA")
    run_PCA(df_real, [df_syn], f"{H.OUT_DIR}/PCA", H.SEED)
    mca_dist, mca_tvd_dist = medical_concept_abundance(df_real, df_syn, H.CAT_COLS, f"{H.OUT_DIR}/medical_abundance")
    combined_clinical_violations = clinical_knowledge_violation(df_train, df_syn, H.CAT_COLS, f"{H.OUT_DIR}/clinical_knowledge_violation")
    EXCLUDE_COLS = ['WHITE', 'BLACK', 'ASIAN', 'HISPANIC', 'UN', 'OTHER', 'DIE_1y']
    cat_cols = [c for c in H.CAT_COLS if c not in EXCLUDE_COLS]
    r2s_results = train_and_test_classification(df_real_norm, df_syn_norm, f"{H.OUT_DIR}/real_to_synthetic", H.SEED, cat_cols)
    r2s_auc = r2s_results["test"]["auroc"]
    r2s_prauc = r2s_results["test"]["prauc"]
    r2s_acc  = r2s_results["test"]["acc"]
    #-----------------UTILITY-----------------#
    s2h_results = train_and_test_classification(df_syn_norm, df_hold_norm, f"{H.OUT_DIR}/synthetic_to_hold", H.SEED,  cat_cols)
    s2h_auc = s2h_results["test"]["auroc"]
    s2h_prauc = s2h_results["test"]["prauc"]
    s2h_acc  = s2h_results["test"]["acc"]
    r2r_results = train_and_test_classification(df_train_norm, df_hold_norm, f"{H.OUT_DIR}/real_to_real", H.SEED, cat_cols)
    r2r_auc = r2r_results["test"]["auroc"]
    r2r_prauc = r2r_results["test"]["prauc"]
    r2r_acc  = r2r_results["test"]["acc"]
    #-----------------PRIVACY-----------------#
    df_train_norm_bal = df_train_norm.sample(n=30000, random_state=H.SEED)[H.NUM_COLS + H.CAT_COLS]
    df_hold_norm_bal = df_hold_norm.sample(n=30000, random_state=H.SEED)[H.NUM_COLS + H.CAT_COLS]
    #df_train_norm_unbal = df_train_norm.sample(n=21000, random_state=H.SEED)[H.NUM_COLS + H.CAT_COLS]
    #df_hold_norm_unbal = df_hold_norm.sample(n=9000, random_state=H.SEED)[H.NUM_COLS + H.CAT_COLS]
    mem_aucs = mem_risk_MIMIC(df_train_norm_bal, df_hold_norm_bal, df_syn_norm, H.CAT_COLS, H.NUM_COLS, f"{H.OUT_DIR}/mem_risk", H.SEED)
    #-----------------LOG RESULTS-----------------#
    with open(f"{H.OUT_DIR}/eval.txt", "a") as f:
        f.write(f"CWC: {cwc}\nr2s_auc: {r2s_auc}\nr2s_prauc: {r2s_prauc}\nr2s_acc: {r2s_acc}\ns2h_auc: {s2h_auc}\ns2h_prauc: {s2h_prauc}\ns2h_acc: {s2h_acc}\n"
                f"r2r_auc: {r2r_auc}\nr2r_prauc: {r2r_prauc}\nr2r_acc: {r2r_acc}\n"
                f"ad2d: {ad2d}\ncontinuous_w_d: {continuous_w_d}\nlatent_cluster_analysis: {latent_cluster_analysis}\nmca_dist: {mca_dist}\nmca_tvd_dist: {mca_tvd_dist}\ncombined_clinical_violations: {combined_clinical_violations}\nmem_auc_real: {mem_aucs['real_roc_auc']}\nmem_auc_synth: {mem_aucs['synth_roc_auc']}\n elapsed_time: {elapsed_time}\n")
    log_result_RL_MIMIC(H.RESULT_CSV, H.RUN_NAME, H.ITERS, H.DATA_SIZE, H.SEED, cwc, ad2d, continuous_w_d, latent_cluster_analysis, mca_dist, mca_tvd_dist, combined_clinical_violations, s2h_auc, s2h_prauc, s2h_acc, r2s_auc, r2s_prauc, r2s_acc, r2r_auc, r2r_prauc, r2r_acc, mem_aucs, elapsed_time)
    return s2h_auc, r2s_auc 


def make_logger(model_id, trial_dir):
    logfile  = Path(f"{trial_dir}/{model_id}_outs.txt")
    fh = open(logfile, "a", buffering=1)   
    def log(msg, end="\n"): 
        fh.write(msg + end)   
        fh.flush()
    return log, fh 



if __name__ == '__main__':
   