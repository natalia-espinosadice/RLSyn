import tensorflow as tf
import pandas as pd
import numpy as np
import os
from pathlib import Path
import random
from evaluation.evaluate_aireadi import (make_logger, log_result_EMR_search, compute_value_stats, train_on_synth_test_on_hold, get_column_wise_correlations, train_on_real_test_on_synth)
from EMR_WGAN.train import train, gen, set_global_seed
import optuna 
from optuna.trial import TrialState
from EMR_WGAN.hyperparams import HyperParams_AIREADI
from optuna.visualization import (plot_optimization_history, plot_parallel_coordinate, plot_pareto_front, plot_param_importances)
import argparse

def evaluate_model(df_real, df_hold, df_syn, df_hold_norm, df_syn_norm, df_real_with_patients_norm, H, checkpoint): 
    #-----------------FIDELITY-----------------#
    #column wise correlations 
    cwc = get_column_wise_correlations(df_real, df_syn, f"{H.OUT_DIR}/ckpt_{checkpoint}/correlations") 
    #distributional scores by value stats 
    overall_score_real_num, overall_score_hold_num, overall_score_real_cat, overall_score_hold_cat, overall_score_real, overall_score_hold = compute_value_stats(df_real, df_hold, df_syn, f"{H.OUT_DIR}/ckpt_{checkpoint}/value_stat_analysis.txt", H.NUM_COLS, H.CAT_COLS)  
    #-----------------UTILITY-----------------#
    #get hold out (keep same size no matter the train - test split)
    ten_percent = 490
    if len(df_hold_norm) > ten_percent: 
        hold_fraction = ten_percent / len(df_hold_norm)   
        df_hold_norm = df_hold_norm.sample(frac=hold_fraction, random_state=H.SEED).reset_index(drop=True)
    #synthetic to hold out 
    s2h_auc, s2h_acc = train_on_synth_test_on_hold(df_syn_norm,  df_hold_norm, f"{H.OUT_DIR}/ckpt_{checkpoint}/synth_to_hold", H.SEED)
    #real to synthetic 
    r2s_auc, r2s_acc = train_on_real_test_on_synth(df_real_with_patients_norm, df_syn_norm, f"{H.OUT_DIR}/ckpt_{checkpoint}/real_to_synth", H.SEED)
    #-----------------LOG RESULTS-----------------#
    with open(f"{H.OUT_DIR}/ckpt_{checkpoint}/eval.txt", "a") as f:
        f.write(f"CWC: {cwc}\noverall_score_hold : {overall_score_hold}\noverall_score_real: {overall_score_real}\noverall_score_real_num: {overall_score_real_num}\noverall_score_hold_num: {overall_score_hold_num}\noverall_score_real_cat: {overall_score_real_cat}\noverall_score_hold_cat:{overall_score_hold_cat}\n"
            f"r2s_auc: {r2s_auc}\nr2s_acc: {r2s_acc}\ns2h_auc: {s2h_auc}\ns2h_acc: {s2h_acc}\n")
        f.write(f"BATCH: {H.BATCH}, NOISE_DIM: {H.NOISE_DIM}, N_CRITIC: {H.DISC_STEPS}, GP_COEFF: {H.GRADIENT_PENALTY}, DISC_LR: {H.D_LR}, GEN LR: {H.G_LR}, G_HIDDEN_DIM: {H.G_H}, D_HIDDEN_DIM: {H.D_H}, G_LAYERS: {H.G_HLAYERS}, D_LAYERS: {H.D_HLAYERS} ")
    log_result_EMR_search(H.RESULT_CSV, H.RUN_NAME, H.EPOCHS, H.DATA_SIZE, H.SEED, checkpoint, cwc, overall_score_real_num, overall_score_hold_num,  overall_score_real_cat, overall_score_hold_cat, overall_score_real, overall_score_hold, 
            s2h_auc, s2h_acc, r2s_auc, r2s_acc)
    return s2h_auc, r2s_auc 

def objective(trial: optuna.Trial, study_name, data_path):
    #search params 
    batch_size = trial.suggest_int("batch_size", 256, 1024, step=256) #1024 
    z_dim = trial.suggest_int("z_dim", 16, 128, step=16)
    n_critic = trial.suggest_int("n_critic", 1, 5, step=2)
    GP_coeff = trial.suggest_categorical("GP_coeff", [0.1, 0.5, 1, 5, 10])
    disc_LR = trial.suggest_categorical("disc_lr", [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4])
    gen_LR  = trial.suggest_categorical("gen_lr",  [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4])
    g_hidden_dim = trial.suggest_int("g_hdim", 64, 512, step=64)
    d_hidden_dim = trial.suggest_int("d_hdim", 64, 512, step=64)
    g_layers =  trial.suggest_int("g_layers", 2, 4)
    d_layers = trial.suggest_int("d_layers", 2, 4)
    H = HyperParams_AIREADI()
    H = H.override(
        OUT_DIR = f"{study_name}/trial_{trial.number}", 
        DATA_PATH = f"{data_path}", 
        RESULT_CSV = f"{study_name}/results.csv", 
        RUN_NAME = f"trial_{trial.number}", 
        DATASET = "AIREADI", 
        NPY_PATH = f"{data_path}/min_max_log.npy", 
        MODEL_ID = f"{study_name}_trial_{trial.number}", 
        CHECKPOINT_DIRECTORY = f"{study_name}_training_checkpoints/trial_{trial.number}", 
        BATCH = batch_size, 
        NOISE_DIM = z_dim, 
        GRADIENT_PENALTY = GP_coeff, 
        G_LR = gen_LR, 
        G_H = g_hidden_dim, 
        G_HLAYERS = g_layers, 
        D_LR = disc_LR, 
        D_H = d_hidden_dim, 
        D_HLAYERS = d_layers, 
        DISC_STEPS = n_critic
    ) 
    os.makedirs(H.OUT_DIR, exist_ok=True)
    os.makedirs(H.CHECKPOINT_DIRECTORY, exist_ok=True)
    #train and log 
    log, fh = make_logger(H.MODEL_ID, H.OUT_DIR)
    try: 
        elapsed_time, best_epoch = train(H, log)
        print("training complete")
    finally: 
        fh.close() 

    #generate/evaluate from the 'best' checkpoint (lowest loss) AND the last checkpoint (last epoch)
    checkpoints = [str(best_epoch), str(H.EPOCHS-10)]
    checkpoint_success, checkpoint_metrics = [], [] 
    final_s2h_auc, final_r2s_auc = 0, 0  
    for checkpoint in checkpoints: 
        gen_success = gen(H, checkpoint)
        checkpoint_success.append(gen_success)
        if gen_success != -1: 
            #get raw to use for cwc, value stat analysis, histograms etc. 
            df_hold = pd.read_csv(f"{H.DATA_PATH}/original_testing_data.csv")[H.NUM_COLS+ H.CAT_COLS]
            df_real = pd.read_csv(f"{H.DATA_PATH}/original_data_with_patients.csv").drop(columns=['patient_id'])[H.NUM_COLS+H.CAT_COLS]
            df_syn = pd.read_csv(f"{H.OUT_DIR}/ckpt_{checkpoint}/synthetic_rescaled.csv")[H.NUM_COLS+H.CAT_COLS]
            #get normalized data to use for classifications (need patients to split real data without leakage)
            df_hold_norm = pd.read_csv(f"{H.DATA_PATH}/normalized_testing_data.csv")[H.NUM_COLS+ H.CAT_COLS]
            df_real_with_patients_norm =  pd.read_csv(f"{H.DATA_PATH}/preprocessed_data_with_patients.csv")[H.NUM_COLS + H.CAT_COLS +["patient_id"]]
            df_syn_norm = pd.read_csv(f"{H.OUT_DIR}/ckpt_{checkpoint}/synthetic.csv")[H.NUM_COLS+H.CAT_COLS]
            #evaluate 
            s2h_auc, r2s_auc = evaluate_model(df_real, df_hold, df_syn, df_hold_norm, df_syn_norm, df_real_with_patients_norm, H, checkpoint) 
            checkpoint_metrics.append(s2h_auc)
            checkpoint_metrics.append(r2s_auc)
    #logic to report to Optuna the better checkpoint
    #failed generation --> prune trial 
    if checkpoint_success[0] == -1 and checkpoint_success[1] == -1: 
        tf.keras.backend.clear_session()
        raise optuna.exceptions.TrialPruned()
    #first failed - take second 
    elif checkpoint_success[0] == -1: 
        final_s2h_auc, final_r2s_auc = checkpoint_metrics[0], checkpoint_metrics[1] #this will be it 
    #second failed - take first 
    elif checkpoint_success[1] == -1: 
        final_s2h_auc, final_r2s_auc = checkpoint_metrics[0], checkpoint_metrics[1]
    #both succeeded, take best 
    else: 
        #take first
        if checkpoint_metrics[0] > checkpoint_metrics[2] and checkpoint_metrics[1] > checkpoint_metrics[3]: 
            final_s2h_auc, final_r2s_auc = checkpoint_metrics[0], checkpoint_metrics[1] 
        #take second 
        elif checkpoint_metrics[2] > checkpoint_metrics[0] and checkpoint_metrics[3] > checkpoint_metrics[1]: 
            final_s2h_auc, final_r2s_auc = checkpoint_metrics[2], checkpoint_metrics[3]
        #if not both greater, then focus on s2h_auc 
        #take first 
        elif checkpoint_metrics[0] > checkpoint_metrics[2]: 
            final_s2h_auc, final_r2s_auc = checkpoint_metrics[0], checkpoint_metrics[1]
        #take second 
        else: 
            final_s2h_auc, final_r2s_auc = checkpoint_metrics[1], checkpoint_metrics[2]
    tf.keras.backend.clear_session()
    return final_s2h_auc, final_r2s_auc

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--STUDY_NAME", type=str, required=True )
    p.add_argument("--DATA_PATH", type=str, required=True )
    #add more here if you want to override other hps. 
    return vars(p.parse_args())

def main(): 
    #gpu set up 
    args = parse_args() 
    study_name = args["STUDY_NAME"]
    data_path = args["DATA_PATH"]
    set_global_seed(0)
    study = optuna.create_study(
        study_name=f"{study_name}", 
        storage=f"sqlite:///{study_name}.db",       
        load_if_exists=True,
        directions= ["maximize", "maximize"]
    )
    base_dir = f"{study_name}"
    os.makedirs(base_dir, exist_ok=True) 
    study.optimize(lambda trial: objective(trial, study_name, data_path), n_trials=20)
    #save best trial / info 
    save_path = f"{base_dir}/best_trial_summary.txt"
    with open(save_path, "w") as f:
        pareto = study.best_trials
        f.write(f"Pareto front has {len(pareto)} trial(s)\n")
        for t in pareto: 
            f.write(f"{t.number},{t.values}, {t.params}\n")
    #pareto front
    objectives = ["r2s_auc", "s2h_auc"]
    fig = plot_pareto_front(study, target_names=objectives) 
    fig.write_html(f"{base_dir}/pareto_front.html")
    # 1D optimization history 
    for i, name in enumerate(objectives):
        fig = plot_optimization_history(study, target = lambda t, idx=i: t.values[idx], target_name = name)
        fig.write_html(f"{base_dir}/opt_history_{name}.html")
    # parameter importance plots per objective
    for i, name in enumerate(objectives):
        fig = plot_param_importances(study, target = lambda t, idx=i: t.values[idx], target_name = name)
        fig.write_html(f"{base_dir}/param_importance_{name}.html")

if __name__ == '__main__':
    main() 