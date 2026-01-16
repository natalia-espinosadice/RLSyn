import torch, torch.nn as nn, torch.nn.functional as F
import os 
import optuna
import gc 
from optuna.visualization import (plot_optimization_history, plot_parallel_coordinate, plot_pareto_front, plot_param_importances)
from evaluation.evaluate_mimic import (log_result_RL_search,  train_and_test_classification,) #get_column_wise_correlations)
from RL.hyperparams import HyperParams_MIMIC
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from RL.train import train, set_seed
import os 
from pathlib import Path
import numpy as np 
import random
import argparse

def evaluate_model(df_hold_norm, df_syn_norm, df_real_norm, H): 
    #-----------------FIDELITY-----------------#
    #column wise correlations 
    #cwc = get_column_wise_correlations(df_real, df_syn, f"{H.OUT_DIR}/correlations", False) 
    #-----------------UTILITY-----------------#
    EXCLUDE_COLS = ['WHITE', 'BLACK', 'ASIAN', 'HISPANIC', 'UN', 'OTHER', 'DIE_1y']
    cat_cols = [c for c in H.CAT_COLS if c not in EXCLUDE_COLS]
    r2s_results = train_and_test_classification(df_real_norm, df_syn_norm, f"{H.OUT_DIR}/real_to_synthetic", H.SEED, cat_cols)
    r2s_auc = r2s_results["test"]["auroc"]
    r2s_prauc = r2s_results["test"]["prauc"]
    r2s_acc  = r2s_results["test"]["acc"]
    s2h_results = train_and_test_classification(df_syn_norm, df_hold_norm, f"{H.OUT_DIR}/synthetic_to_hold", H.SEED,  cat_cols)
    s2h_auc = s2h_results["test"]["auroc"]
    s2h_prauc = s2h_results["test"]["prauc"]
    s2h_acc  = s2h_results["test"]["acc"]
    #-----------------LOG RESULTS-----------------#
    with open(f"{H.OUT_DIR}/eval.txt", "a") as f:
        f.write(f"r2s_auc: {r2s_auc}\nr2s_prauc: {r2s_prauc}\nr2s_acc: {r2s_acc}\ns2h_auc: {s2h_auc}\ns2h_prauc: {s2h_prauc}\ns2h_acc: {s2h_acc}\n\n")
        f.write(f"BATCH: {H.BATCH}, NOISE_DIM: {H.NOISE_DIM}, N_CRITIC: {H.DISC_STEPS}, GP_COEFF: {H.GRADIENT_PENALTY}, DISC_LR: {H.D_LR}, GEN LR: {H.G_LR}, G_HIDDEN_DIM: {H.G_H}, D_HIDDEN_DIM: {H.D_H}")
    log_result_RL_search(H.RESULT_CSV, H.RUN_NAME, H.ITERS, H.DATA_SIZE, H.SEED, np.nan, s2h_auc, s2h_prauc, s2h_acc, r2s_auc, r2s_prauc, r2s_acc,)
    return s2h_auc, r2s_auc 

def objective(trial:optuna.Trial, study_name, data_path): 
    #define parameters 
    BATCH = trial.suggest_int("batch", 512, 4096, step=256) 
    NOISE_DIM = trial.suggest_int("noise_dim", 64, 256, step=32)  
    PPO_EPOCHS = trial.suggest_int("ppo_epochs", 1, 5) 
    DISC_STEPS = trial.suggest_int("disc_steps", 1, 5) 
    MEAN_PENALTY_SCALE = trial.suggest_float("mean_penalty", 0, 0.2, step=0.1) 
    GRADIENT_PENALTY = trial.suggest_categorical("gradient_penalty", [1, 5, 10]) 
    USE_TANH = trial.suggest_categorical("use_tanh", [True, False])
    G_LR = trial.suggest_categorical("g_lr", [1e-6, 5e-5, 1e-5, 1e-4, 5e-4]) 
    D_LR = trial.suggest_categorical("d_lr", [1e-6, 1e-5, 3e-5, 5e-5, 5e-4, 1e-4]) 
    G_H = trial.suggest_int("g_h", 128, 512, step=128) 
    D_H = trial.suggest_int("d_h", 128, 512, step=128) 
    #define save paths 
    
    H = HyperParams_MIMIC()
    H = H.override(
        OUT_DIR = f"{study_name}/trial_{trial.number}", 
        DATA_PATH = f"{data_path}", 
        RESULT_CSV = f"{study_name}/results.csv", 
        RUN_NAME = f"trial_{trial.number}", 
        DATASET = "MIMIC", 
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu', 
        ITERS = 30000, 
        NPY_PATH = f"{data_path}/min_max_log.npy", 
        NUM_SAMPLES = 30000, 
        BATCH = BATCH, 
        NOISE_DIM = NOISE_DIM, 
        GRADIENT_PENALTY = GRADIENT_PENALTY, 
        G_LR = G_LR, 
        G_H = G_H, 
        D_LR = D_LR, 
        D_H = D_H, 
        DISC_STEPS = DISC_STEPS, 
        USE_TANH = USE_TANH, 
        PPO_EPOCHS = PPO_EPOCHS, 
        MEAN_PENALTY_SCALE = MEAN_PENALTY_SCALE, 
        VF_COEF = 0.5, 
        CLIP_EPS = 0.1, 
        ENT_BETA = 1e-3, 
        EPS  = 1e-6, 
    ) 
    os.makedirs(H.OUT_DIR, exist_ok=True)
    df_train = pd.read_csv(f"{H.DATA_PATH}/normalized_training_data.csv")[H.NUM_COLS + H.CAT_COLS]
    real = torch.tensor(df_train.values, dtype=torch.float32)
    loader = DataLoader(TensorDataset(real), batch_size=H.BATCH, shuffle=True, num_workers=0) 
    df_syn, elapsed_time = train(df_train, real, loader, H)
    df_hold_norm = pd.read_csv(f"{H.DATA_PATH}/normalized_testing_data.csv")[H.NUM_COLS+ H.CAT_COLS]
    df_syn_norm = pd.read_csv(f"{H.OUT_DIR}/synthetic.csv")[H.NUM_COLS+H.CAT_COLS]
    df_real_norm =  pd.concat([df_train, df_hold_norm])[H.NUM_COLS + H.CAT_COLS]
    try:
        s2h_auc, r2s_auc = evaluate_model(df_hold_norm, df_syn_norm, df_real_norm, H)          
        return s2h_auc, r2s_auc
    except Exception as e:
        raise optuna.TrialPruned(f"Pruned due to evaluation failure")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--STUDY_NAME", type=str, required=True )
    p.add_argument("--DATA_PATH", type=str, required=True )
    #add more here if you want to override other hps. 
    return vars(p.parse_args())

def main(): 
    args = parse_args() 
   study_name = args["STUDY_NAME"]
    data_path = args["DATA_PATH"]
    set_seed(0)
    base_dir = f"{study_name}"
    os.makedirs(base_dir, exist_ok=True) 
    study = optuna.create_study(
        study_name=f"{study_name}",
        storage=f"sqlite:///{study_name}.db",       
        load_if_exists=True,
        directions= ("maximize", "maximize")
    )
    objectives = ["r2s_auc", "s2r_auc"]
    study.optimize(lambda trial: objective(trial, study_name, data_path), n_trials = 25)
    #best trial 
    best_save = f"{base_dir}/best_trial_summary.txt"
    with open(best_save, "w") as f:
        pareto = study.best_trials
        f.write(f"Pareto front has {len(pareto)} trial(s)\n")
        for t in pareto: 
            f.write(f"{t.number},{t.values}, {t.params}\n")
    #pareto front
    fig = plot_pareto_front(study, target_names=objectives) 
    fig.write_html(f"{base_dir}/pareto_front.html")
    # 1D optimization history 
    for i, name in enumerate(objectives):
        fig = plot_optimization_history(study, target = lambda t, idx=i: t.values[idx], target_name = name)
        fig.write_html(f"{base_dir}/opt_history_{name}.html")
    #parameter importance plots per objective
    for i, name in enumerate(objectives):
        fig = plot_param_importances(study, target = lambda t, idx=i: t.values[idx], target_name = name)
        fig.write_html(f"{base_dir}/param_importance_{name}.html")
    
if __name__ == '__main__': 
    main() 
