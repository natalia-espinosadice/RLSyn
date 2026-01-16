import torch, torch.nn as nn, torch.nn.functional as F
import os, socket 
import optuna
from optuna.visualization import (plot_optimization_history, plot_parallel_coordinate, plot_pareto_front, plot_param_importances)
from evaluation.evaluate_aireadi import (train_on_synth_test_on_hold,  train_on_real_test_on_synth)
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import time 
from pathlib import Path
import numpy as np 
import sys, subprocess, csv, yaml
from pathlib import Path
import argparse 

NUM_COLS= [
        'heart_rate_mean', 'blood_glucose_mean', 'resp_rate_mean', 'stress_mean', 'blood_glucose_std', 'total_steps',  'resting_heart_rate',
        'total_kcal', 'sleep_light_hrs', 'sleep_deep_hrs', 'sleep_rem_hrs', 'sleep_awake_hrs', 'act_generic_hrs', 'act_running_hrs','act_walking_hrs', 'act_sedentary_hrs', 
        'A/G Ratio_value', 'ALT (IU/L)_value (IU/L)', 'AST (IU/L)_value (IU/L)', 'Albumin (g/dL)_value (g/dL)', 'Alkaline Phosphatase (IU/L)_value (IU/L)', 'BUN (mg/dL)_value (mg/dL)', 'BUN/Creatinine ratio_value', "Bilirubin Total (mg/dL)_value (mg/dL)", 'C-Peptide (ng/mL)_value (ng/mL)', 
        'CRP - HS (mg/L)_value (mg/L)', 'Calcium (mg/dL)_value (mg/dL)', 'Carbon Dioxide, Total (mEq/L)_value (mEq/L)', 'Chloride (mEq/L)_value (mEq/L)', 'Creatinine (mg/dL)_value (md/dL)', 'Globulin, Total (g/dL)_value (g/dL)', 'Glucose (mg/dL)_value (mg/dL)', 'HDL Cholesterol (mg/dL)_value (mg/dL)',
        'INSULIN (ng/mL)_value (ng/mL)', 'LDL Cholesterol Calculation (mg/dL)_value (mg/dL)',  'Potassium (mEq/L)_value (mEq/L)', 'Protein, Total (g/dL)_value (g/dL)', 'Sodium (mEq/L)_value (mEq/L)', 'Total Cholesterol (mg/dL)_value (mg/dL)', 'Triglycerides (mg/dL)_value (mg/dL)',  'Urine Albumin (mg/dL)_value (mg/DL)', 'Urine Creatinine (mg/dL)_value (mg/DL)', 
        'clock_visuospatial_executive_time_value', 'delayed_recall_with_no_clue_time_value', 'digitspan_time_value', 'memory_trial1_time_value', 'memory_trial2_time_value', 'moca_abstraction_time_value', 'moca_orientation_time_value', 'moca_total_score_value', 'naming_time_value', 'repetition_time_value', 'subtraction_time_value',  
        'cube_visuospatial_executive_time_value', 'lettera_time_value', 'trails_visuospatial_executive_time_value','moca_combined_mis_score_value', 'memory_trial1_value', 'memory_trial2_value', 'moca_abstraction_value', 'moca_orientation_value', 'naming_value', 'repetition_value', 'subtraction_value', 'Rate', 'PR', 'QRSD', 'QT', 'QTc', 'P', 'QRS', 'T',     
]
CAT_COLS = ['Age-related macular degeneration', 'Arthritis', 'Cancer', 'Cataracts (1+ eyes)', 'Chronic pullmonary problems', 'Circulation problems', 'Diabetic retinopathy (1+)', 'Digestive problems', 'Marijuana user', 'Dry eye (1+)', 'Glaucoma (1+)', 'Hearing impairment',
        'Heart attack', 'High blood cholesterol', 'High blood pressure', 'Kidney problems', 'Low blood pressure', 'Mild cognitive impairmen', 'Multiple sclerosis', 'Obesity', 'Osteoporosis','Other heart issues (pacemaker)', 'Other neurological conditions',
        "Parkinson's disease", 'Retinal vascular occlusion', 'Stroke', 'Type 2 Diabetes', 'Urinary problems', 
        'ABNORMAL', 'BORDERLINE', 'NORMAL', 'OTHERWISE NORMAL', 'cube_visuospatial_executive_value', 'fluency_language_value', 'lettera_value', 'trails_visuospatial_executive_value'
]

REPO_ROOT = Path(__file__).resolve().parent  
DATA_ROOT = REPO_ROOT.parent / "AI-READI-FULL" / "preprocessed_data0.9_seed0"
CONFIG_TRAIN  = REPO_ROOT / "configs/our_aireadi/train_edm.yaml"
CONFIG_SAMPLE = REPO_ROOT / "configs/our_aireadi/sample_edm.yaml"

def get_free_port():
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]
port  = get_free_port() 

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def dump_config(cfg, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

def override_cfg(cfg, overrides: dict):
    def rec(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = rec(d.get(k, {}), v)
            else:
                d[k] = v
        return d
    return rec(cfg, overrides)

def log_result(RESULTS_CSV, tag, seed, batch, time_dim, learning_rate, s2h_auc, s2h_prauc, s2h_acc, r2s_auc, r2s_prauc, r2s_acc, elapsed_time, n_epochs):
    p = Path(RESULTS_CSV)
    p.parent.mkdir(parents=True, exist_ok=True)
    row = {"tag": tag, "seed" : seed, "epochs": n_epochs, 
        "batch": batch, "time_dim": time_dim, "lr": learning_rate, 
        "s2h_auc" : s2h_auc, "s2h_prauc": s2h_prauc, "s2h_acc" : s2h_acc,
        "r2s_auc" : r2s_auc, "r2s_prauc": r2s_prauc, "r2s_acc" : r2s_acc,
        "elapsed_time" : elapsed_time,
    }
    write_header = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)

def evaluate_model(df_hold_norm, df_real_with_patients_norm, df_syn_norm, OUT_DIR, seed, result_csv, batch, time_dim, learning_rate, trial, elapsed_time, n_epochs): 
    #-----------------UTILITY-----------------#
    #get hold out (keep same size no matter the train - test split)
    ten_percent = 490
    if len(df_hold_norm) > ten_percent: 
        hold_fraction = ten_percent / len(df_hold_norm)   
        df_hold_norm = df_hold_norm.sample(frac=hold_fraction, random_state=seed).reset_index(drop=True)
    #synthetic to hold out 
    s2h_auc, s2h_acc = train_on_synth_test_on_hold(df_syn_norm,  df_hold_norm, f"{OUT_DIR}/synth_to_hold", seed)
    #real to synthetic 
    r2s_auc, r2s_acc = train_on_real_test_on_synth(df_real_with_patients_norm, df_syn_norm, f"{OUT_DIR}/real_to_synth", seed)
    #-----------------LOG RESULTS-----------------#
    with open(f"{OUT_DIR}/eval.txt", "a") as f:
        f.write(f"r2s_auc: {r2s_auc}\nr2s_acc: {r2s_acc}\ns2h_auc: {s2h_auc}\ns2h_acc: {s2h_acc}\n")
        f.write(f"BATCH: {batch}, TIME_DIM: {time_dim}, LR: {learning_rate}, ELAPSED TIME: {elapsed_time}")
    log_result(result_csv, trial, seed, batch, time_dim, learning_rate, s2h_auc, np.nan, s2h_acc, r2s_auc, np.nan, r2s_acc, elapsed_time, n_epochs)
    return s2h_auc, r2s_auc 


def run_to_logs(cmd, cwd, trial_dir, name, env=None):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env)
    (trial_dir / f"{name}.stdout.txt").write_text(p.stdout)
    (trial_dir / f"{name}.stderr.txt").write_text(p.stderr)
    return p

def objective(trial:optuna.Trial, study_name, data_path): 
    #define parameters 
    batch = trial.suggest_int("batch_size", 128, 512, step=128)
    lr = trial.suggest_float("learning_rate", 3e-5, 3e-3, log=True)
    time_dim = trial.suggest_categorical("time_dim", [128, 192, 256])
    n_epochs = trial.suggest_int("n_epochs", 500, 2500, step=500)

    #set up paths 
    base_dir  = REPO_ROOT / study_name
    trial_dir = base_dir / f"trial_{trial.number}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    train_cfg_out = trial_dir / "train_edm.yaml"
    sample_cfg_out =  trial_dir / "sample_edm.yaml"
    ckpt_path = trial_dir / "checkpoints" / "final_checkpoint.pth"
    npy_path = DATA_ROOT / "min_max_log.npy"
    synth_path = trial_dir / "samples" / "all_x.npy"
    save_syn = trial_dir / "synthetic.csv"
    save_syn_rescaled = trial_dir / "synthetic_rescaled.csv"
    seed = 2023 
    result_csv = base_dir / "results.csv"
    train_data = DATA_ROOT / "normalized_training_data.csv"
    train_array_npy = (DATA_ROOT / "normalized_training_data.array.npy").resolve()
    df = pd.read_csv(train_data)[CAT_COLS + NUM_COLS]
    print(len(df.columns))
    arr = df.values.astype(np.float32)
    print(arr.shape[0], arr.shape[1])
    np.save(train_array_npy, arr)
    raw_data = np.load(train_array_npy)
    print(raw_data.shape[1])
    
    #load override and save config file 
    train_cfg  = load_config(CONFIG_TRAIN)
    sample_cfg = load_config(CONFIG_SAMPLE)
    train_overrides = {"setup": {"master_address": "127.0.0.1", "master_port": port}, "data": {"path": str(train_array_npy)}, "train": {"batch_size": batch, "n_epochs": n_epochs}, "model": {"network": {"time_dim": time_dim, "unit_dims": [1024, time_dim, time_dim, time_dim, 1024]}}, 
        "optim": {"params": {"lr": lr}}, 
    }
    sample_overrides = {"setup": {"master_address": "127.0.0.1", "master_port": port}, "model": {"ckpt": str(ckpt_path), "network": {"time_dim": time_dim, "unit_dims": [1024, time_dim, time_dim, time_dim, 1024], }}}
    train_cfg  = override_cfg(train_cfg,  train_overrides)
    sample_cfg = override_cfg(sample_cfg, sample_overrides)
    dump_config(train_cfg,  train_cfg_out)
    dump_config(sample_cfg, sample_cfg_out)
    sample_npy_path = trial_dir / "samples" / "sample.npy"

    #train 
    start_time = time.time() 
    cmd = [sys.executable, "main.py", "--mode", "train",  "--workdir", str(trial_dir), "--config", str(train_cfg_out)]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH','')}"
    proc  = run_to_logs(cmd,      cwd=REPO_ROOT, trial_dir=trial_dir, name="train", env=env)
    if proc.returncode != 0:
        raise optuna.TrialPruned(f"TRAIN failed (rc={proc.returncode}).\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    #sample 
    cmd_eval = [sys.executable, "main.py", "--mode", "eval", "--workdir", str(trial_dir), "--config", str(sample_cfg_out)]
    proc2 = run_to_logs(cmd_eval, cwd=REPO_ROOT, trial_dir=trial_dir, name="eval",  env=env)
    elapsed_time = (time.time() - start_time) / 60 

    #get synthetic 
    synthetic = np.load(synth_path, allow_pickle=True)
    df_syn = pd.DataFrame(synthetic, columns=CAT_COLS + NUM_COLS)
    df_syn.to_csv(save_syn, index=False)
    feature_range = np.load(npy_path, allow_pickle=True).item()
    for col in NUM_COLS:
        xmin, xmax = feature_range[col]
        df_syn[col] = (1.0 - df_syn[col]) * xmin + df_syn[col] * xmax
    df_syn.to_csv(save_syn_rescaled)

    #evaluation 
    train_csv = DATA_ROOT / "normalized_training_data.csv"
    test_csv = DATA_ROOT / "normalized_testing_data.csv"
    real_csv = DATA_ROOT / "preprocessed_data_with_patients.csv"
    #df_train_norm = pd.read_csv(train_csv)[NUM_COLS + CAT_COLS]
    df_hold_norm =  pd.read_csv(test_csv)[NUM_COLS + CAT_COLS]
    df_real_norm_with_patients = pd.read_csv(real_csv)[NUM_COLS + CAT_COLS + ['patient_id']]
    df_syn_norm = pd.read_csv(save_syn)[NUM_COLS + CAT_COLS]
    try:
        s2h_auc, r2s_auc = evaluate_model(df_hold_norm, df_real_norm_with_patients, df_syn_norm, trial_dir, seed, result_csv, batch, time_dim, lr, str(trial.number), elapsed_time, n_epochs)          
        if sample_npy_path.exists():
            os.remove(sample_npy_path)
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
    base_dir = (REPO_ROOT / study_name)
    base_dir.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=f"{study_name}",
        storage=f"sqlite:///{study_name}.db",       
        load_if_exists=True,
        directions= ("maximize", "maximize")
    )
    objectives = ["s2h_auc", "r2s_auc"]
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
   