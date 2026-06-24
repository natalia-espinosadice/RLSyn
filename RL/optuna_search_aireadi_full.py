import torch, torch.nn as nn, torch.nn.functional as F
import os 
import optuna
import gc 
from optuna.visualization import (plot_optimization_history, plot_pareto_front, plot_param_importances)
from evaluation.evaluate_aireadi import (log_result_RL_search,  train_on_synth_test_on_hold, mem_risk, get_column_wise_correlations, train_on_real_test_on_synth,)
from RL.hyperparams import HyperParams_AIREADI 
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from RL.train import train, set_seed
import os 
import argparse 
from pathlib import Path
import numpy as np 
import random

def evaluate_model(df_real, df_hold, df_syn, df_train_norm, df_hold_norm, df_syn_norm, df_real_with_patients_norm, H): 
    #-----------------FIDELITY-----------------#
    #column wise correlations 
    cwc = get_column_wise_correlations(df_real, df_syn, f"{H.OUT_DIR}/correlations") 
    #-----------------UTILITY-----------------#
    #get hold out (keep same size no matter the train - test split)
    ten_percent = 490
    if len(df_hold_norm) > ten_percent: 
        hold_fraction = ten_percent / len(df_hold_norm)   
        df_hold_norm = df_hold_norm.sample(frac=hold_fraction, random_state=H.SEED).reset_index(drop=True)
    #synthetic to hold out 
    s2h_auc, s2h_acc = train_on_synth_test_on_hold(df_syn_norm,  df_hold_norm, f"{H.OUT_DIR}/synth_to_hold", H.SEED)
    #real to synthetic 
    r2s_auc, r2s_acc = train_on_real_test_on_synth(df_real_with_patients_norm, df_syn_norm, f"{H.OUT_DIR}/real_to_synth", H.SEED)
    mem_aucs = mem_risk(df_train_norm, df_hold_norm, df_syn_norm, H.CAT_COLS, H.NUM_COLS, f"{H.OUT_DIR}/mem_risk", H.SEED) 
    synth_mem_auc = mem_aucs["synth_roc_auc"]
    #-----------------LOG RESULTS-----------------#
    with open(f"{H.OUT_DIR}/eval.txt", "a") as f:
        f.write(f"CWC: {cwc}\nsynth_mem_auc: {synth_mem_auc}\n"
            f"r2s_auc: {r2s_auc}\nr2s_acc: {r2s_acc}\ns2h_auc: {s2h_auc}\ns2h_acc: {s2h_acc}\n")
        f.write(f"BATCH: {H.BATCH}, NOISE_DIM: {H.NOISE_DIM}, N_CRITIC: {H.DISC_STEPS}, GP_COEFF: {H.GRADIENT_PENALTY}, DISC_LR: {H.D_LR}, GEN LR: {H.G_LR}, G_HIDDEN_DIM: {H.G_H}, D_HIDDEN_DIM: {H.D_H}")
    log_result_RL_search(H.RESULT_CSV, H.RUN_NAME, H.ITERS, H.DATA_SIZE, H.SEED, cwc, synth_mem_auc, s2h_auc, s2h_acc, r2s_auc, r2s_acc)
    return s2h_auc, r2s_auc, synth_mem_auc 

def objective(trial:optuna.Trial, study_name, data_path): 
    #define parameters 
    BATCH = trial.suggest_int("batch", 128, 384, step=128) 
    NOISE_DIM = trial.suggest_int("noise_dim", 16, 112, step=48) 
    PPO_EPOCHS = trial.suggest_int("ppo_epochs", 1, 5, step = 2) 
    DISC_STEPS = trial.suggest_int("disc_steps", 1, 5, step =2) 
    MEAN_PENALTY_SCALE = trial.suggest_float("mean_penalty", 0, 0.2, step=0.2) 
    GRADIENT_PENALTY = trial.suggest_int("gradient_penalty", 5, 10, step = 5) 
    USE_TANH = True # trial.suggest_categorical("use_tanh", [True, False])
    G_LR = trial.suggest_categorical("g_lr", [5e-5, 1e-4, 2e-4]) 
    D_LR = trial.suggest_categorical("d_lr", [1e-5, 3e-5, 5e-5, 5e-4]) 
    G_H = 64 # trial.suggest_int("g_h", 64, 128, step=64) 
    D_H = 64 # trial.suggest_int("d_h", 64, 128, step=64) 
    #define save paths 
    
    H = HyperParams_AIREADI()
    H = H.override(
        OUT_DIR = f"{study_name}/trial_{trial.number}", 
        DATA_PATH = f"{data_path}", 
        RESULT_CSV = f"{study_name}/results.csv", 
        RUN_NAME = f"trial_{trial.number}", 
        DATASET = "AIREADI", 
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu', 
        ITERS = 30000, 
        NPY_PATH = f"{data_path}/min_max_log.npy", 
        BATCH = BATCH, 
        NOISE_DIM = NOISE_DIM, 
        GRADIENT_PENALTY = GRADIENT_PENALTY, 
        G_LR = G_LR, 
        G_H = G_H, 
        D_LR = D_LR, 
        D_H = D_H, 
        DISC_STEPS = DISC_STEPS, 
        NUM_SAMPLES = 5000, 
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
    #get raw to use for cwc, value stat analysis, histograms etc. 
    df_hold = pd.read_csv(f"{H.DATA_PATH}/original_testing_data.csv")[H.NUM_COLS+ H.CAT_COLS]
    df_real = pd.read_csv(f"{H.DATA_PATH}/original_data_with_patients.csv").drop(columns=['patient_id'])[H.NUM_COLS+H.CAT_COLS]
    df_syn = pd.read_csv(f"{H.OUT_DIR}/synthetic_rescaled.csv")[H.NUM_COLS+H.CAT_COLS]
    #get normalized data to use for classifications (need patients to split real data without leakage)
    df_train_norm = pd.read_csv(f"{H.DATA_PATH}/normalized_training_data.csv")[H.NUM_COLS+ H.CAT_COLS]
    df_hold_norm = pd.read_csv(f"{H.DATA_PATH}/normalized_testing_data.csv")[H.NUM_COLS+ H.CAT_COLS]
    df_real_with_patients_norm =  pd.read_csv(f"{H.DATA_PATH}/preprocessed_data_with_patients.csv")[H.NUM_COLS + H.CAT_COLS +["patient_id"]]
    df_syn_norm = pd.read_csv(f"{H.OUT_DIR}/synthetic.csv")[H.NUM_COLS+H.CAT_COLS]
    try:
        s2h_auc, r2s_auc, synth_mem_auc = evaluate_model(df_real, df_hold, df_syn, df_train_norm, df_hold_norm, df_syn_norm, df_real_with_patients_norm, H)          
        return s2h_auc, r2s_auc, synth_mem_auc
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
        directions= ("maximize", "maximize", "minimize")
    )
    objectives = ["s2h_auc", "r2s_auc", "synth_mem_auc"]
    study.optimize(lambda trial: objective(trial, study_name, data_path), n_trials = 20)
    #best trial info 
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
