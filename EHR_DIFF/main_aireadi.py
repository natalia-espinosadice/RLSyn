import torch, torch.nn as nn, torch.nn.functional as F
import os, socket
from evaluation.evaluate_aireadi import (get_histograms, train_on_synth_test_on_hold, mem_risk, train_on_synth_test_on_real,  train_on_real_test_on_real, train_on_real_test_on_synth, latent_cluster_analysis, get_column_wise_correlations)
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import os 
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
        'INSULIN (ng/mL)_value (ng/mL)', 'LDL Cholesterol Calculation (mg/dL)_value (mg/dL)',  'Potassium (mEq/L)_value (mEq/L)', 'Protein, Total (g/dL)_value (g/dL)', 'Sodium (mEq/L)_value (mEq/L)', 'Total Cholesterol (mg/dL)_value (mg/dL)', 'Triglycerides (mg/dL)_value (mg/dL)',  
        'Urine Albumin (mg/dL)_value (mg/DL)', 'Urine Creatinine (mg/dL)_value (mg/DL)', 'clock_visuospatial_executive_time_value', 'delayed_recall_with_no_clue_time_value', 'digitspan_time_value', 'memory_trial1_time_value', 'memory_trial2_time_value', 'moca_abstraction_time_value', 
        'moca_orientation_time_value', 'moca_total_score_value', 'naming_time_value', 'repetition_time_value', 'subtraction_time_value',  'cube_visuospatial_executive_time_value', 'lettera_time_value', 'trails_visuospatial_executive_time_value', 'moca_combined_mis_score_value', 'memory_trial1_value', 
        'memory_trial2_value', 'moca_abstraction_value', 'moca_orientation_value', 'naming_value', 'repetition_value',  'subtraction_value', 'Rate', 'PR', 'QRSD', 'QT', 'QTc', 'P', 'QRS', 'T',     
]
CAT_COLS = ['Age-related macular degeneration', 'Arthritis', 'Cancer', 'Cataracts (1+ eyes)', 'Chronic pullmonary problems', 'Circulation problems', 'Diabetic retinopathy (1+)', 'Digestive problems', 'Marijuana user', 'Dry eye (1+)', 'Glaucoma (1+)', 'Hearing impairment',
        'Heart attack', 'High blood cholesterol', 'High blood pressure', 'Kidney problems', 'Low blood pressure', 'Mild cognitive impairmen', 'Multiple sclerosis', 'Obesity', 'Osteoporosis','Other heart issues (pacemaker)', 'Other neurological conditions',
        "Parkinson's disease", 'Retinal vascular occlusion', 'Stroke', 'Type 2 Diabetes', 'Urinary problems', 
        'ABNORMAL', 'BORDERLINE', 'NORMAL', 'OTHERWISE NORMAL', 'cube_visuospatial_executive_value', 'fluency_language_value', 'lettera_value', 'trails_visuospatial_executive_value'
]

REPO_ROOT = Path(__file__).resolve().parent  
DATA_ROOT = REPO_ROOT.parent / "AI-READI-FULL" 
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
    #by nested dict keys. Ex: "data": {"path": "/data/EHR"},
    def rec(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = rec(d.get(k, {}), v)
            else:
                d[k] = v
        return d
    return rec(cfg, overrides)


def log_result(RESULTS_CSV, tag, iters, data_size, seed, cwc, 
               r2r_auc, r2r_acc, s2h_auc, s2h_acc, s2r_auc, s2r_acc, r2s_auc, r2s_acc, rvs_auc, rvs_acc, real_mem_auc, real_mem_auc_bal, synth_mem_auc, synth_mem_auc_bal, elapsed_time):
    p = Path(RESULTS_CSV)
    p.parent.mkdir(parents=True, exist_ok=True)
    row = {"tag": tag, "iters": iters, "seed" : seed,
        "cwc": cwc, "r2r_auc" : r2r_auc, "r2r_acc" : r2r_acc, "s2h_auc" : s2h_auc, "s2h_acc" : s2h_acc,
        "s2r_auc" : s2r_auc, "s2r_acc" : s2r_acc, "r2s_auc" : r2s_auc, "r2s_acc" : r2s_acc,
        "real_mem_auc" : real_mem_auc, "real_mem_auc_bal" : real_mem_auc_bal, "synth_mem_auc" : synth_mem_auc, "synth_mem_auc_bal" : synth_mem_auc_bal, "elapsed_time" : elapsed_time,
    }
    write_header = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)

def evaluate_model_aireadi(df_real, df_hold, df_train, df_syn, df_hold_norm, df_train_norm, df_syn_norm, df_real_with_patients_norm, OUT_DIR, RESULT_CSV, RUN_NAME, ITERS, SEED, NUM_COLS, CAT_COLS, elapsed_time): 
    #-----------------FIDELITY-----------------#
    #examine value statistics per feature
    #print_stats(f"{OUT_DIR}/value_stats.txt", df_real, df_syn, df_hold, df_train, NUM_COLS) 
    #get single feature histograms
    get_histograms(df_real, df_syn, f"{OUT_DIR}/histograms")
    #PCA analysis 
    latent_cluster_analysis(df_real_with_patients_norm.drop(columns=['patient_id']), [df_syn_norm], f"{OUT_DIR}/PCA", SEED)
    #column wise correlations 
    cwc = get_column_wise_correlations(df_real, df_syn, f"{OUT_DIR}/correlations") 
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
    #-----------------PRIVACY---------------------#
    mem_aucs = mem_risk(df_train_norm, df_hold_norm, df_syn_norm, CAT_COLS, NUM_COLS, f"{OUT_DIR}/mem_risk", SEED) 
    #-----------------LOG RESULTS-----------------#
    with open(f"{OUT_DIR}/eval.txt", "a") as f:
        f.write(f"CWC: {cwc}\n"
            f"r2r_auc: {r2r_auc}\nr2r_acc: {r2r_acc}\nr2s_auc: {r2s_auc}\nr2s_acc: {r2s_acc}\ns2r_auc: {s2r_auc}\ns2r_acc: {s2r_acc}\ns2h_auc: {s2h_auc}\ns2h_acc: {s2h_acc}\n"
            f"real_mem_auc_: {mem_aucs['real_roc_auc']}\nreal_mem_auc_bal: {mem_aucs['real_roc_auc_bal']}\nsynth_mem_auc: {mem_aucs['synth_roc_auc']}\nsynth_mem_auc_bal: {mem_aucs['synth_roc_auc_bal']}\nelapsed_time: {elapsed_time}")
    log_result(RESULT_CSV, RUN_NAME, ITERS, SEED, cwc, r2r_auc, r2r_acc, s2h_auc, s2h_acc, s2r_auc, s2r_acc, r2s_auc, r2s_acc, mem_aucs["real_roc_auc"], mem_aucs["real_roc_auc_bal"], mem_aucs["synth_roc_auc"], mem_aucs["synth_roc_auc_bal"], elapsed_time)


def run_to_logs(cmd, cwd, trial_dir, name, env=None):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env)
    (trial_dir / f"{name}.stdout.txt").write_text(p.stdout)
    (trial_dir / f"{name}.stderr.txt").write_text(p.stderr)
    return p

def train(data_path, seed, base_dir, npy_path, run_name): 
    trial_dir = base_dir / f"{run_name}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    train_cfg_out = trial_dir / "train_edm.yaml"
    sample_cfg_out =  trial_dir / "sample_edm.yaml"
    ckpt_path = trial_dir / "checkpoints" / "final_checkpoint.pth"
    synth_path = trial_dir / "samples" / "all_x.npy"
    save_syn = trial_dir / "synthetic.csv"
    save_syn_rescaled = trial_dir / "synthetic_rescaled.csv"
    result_csv = base_dir / "results.csv"
    train_data = f"{data_path}/normalized_training_data.csv"
    train_array_npy = Path(data_path) / f"normalized_training_data.array.npy"

    df = pd.read_csv(train_data)[CAT_COLS + NUM_COLS] #FIXED 
    print(len(df.columns))
    arr = df.values.astype(np.float32)
    print(arr.shape[0], arr.shape[1])
    np.save(train_array_npy, arr)
    raw_data = np.load(train_array_npy)
    print(raw_data.shape[1])
    
    #load override and save config file 
    train_cfg  = load_config(CONFIG_TRAIN)
    sample_cfg = load_config(CONFIG_SAMPLE)
    train_overrides = {"setup": {"master_address": "127.0.0.1", "master_port": port}, "data": {"path": str(train_array_npy)}, "train": {"seed": seed}}
    sample_overrides = {"setup": {"master_address": "127.0.0.1", "master_port": port}, "model": {"ckpt": str(ckpt_path)}, "test": {"seed": seed}}
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
    #sample 
    cmd_eval = [sys.executable, "main.py", "--mode", "eval", "--workdir", str(trial_dir), "--config", str(sample_cfg_out)]
    proc2 = run_to_logs(cmd_eval, cwd=REPO_ROOT, trial_dir=trial_dir, name="eval",  env=env)
    elapsed_time = (time.time() - start_time) / 60 

    #get synthetic 
    synthetic = np.load(synth_path, allow_pickle=True)
    df_syn = pd.DataFrame(synthetic, columns=CAT_COLS + NUM_COLS) #FIXED
    df_syn.to_csv(save_syn, index=False)
    feature_range = np.load(npy_path, allow_pickle=True).item()
    for col in NUM_COLS:
        xmin, xmax = feature_range[col]
        df_syn[col] = (1.0 - df_syn[col]) * xmin + df_syn[col] * xmax
    df_syn.to_csv(save_syn_rescaled)
    #evaluation 
    df_train_norm = pd.read_csv(f"{data_path}/normalized_training_data.csv")[NUM_COLS + CAT_COLS]
    df_hold_norm =  pd.read_csv(f"{data_path}/normalized_testing_data.csv")[NUM_COLS + CAT_COLS]
    df_real_with_patients_norm = pd.read_csv(f"{data_path}/preprocessed_data_with_patients.csv")[NUM_COLS + CAT_COLS + ['patient_id']]
    df_train = pd.read_csv(f"{data_path}/original_training_data.csv")[NUM_COLS + CAT_COLS]
    df_hold =  pd.read_csv(f"{data_path}/original_testing_data.csv")[NUM_COLS + CAT_COLS]
    df_real = pd.concat([df_train, df_hold])[NUM_COLS + CAT_COLS]
    df_syn = pd.read_csv(save_syn_rescaled)[NUM_COLS + CAT_COLS]
    df_syn_norm = pd.read_csv(save_syn)[NUM_COLS + CAT_COLS]
    evaluate_model_aireadi(df_real, df_hold, df_train, df_syn, df_hold_norm, df_train_norm, df_syn_norm, df_real_with_patients_norm, trial_dir, result_csv, run_name, seed, NUM_COLS, CAT_COLS, elapsed_time)
    if sample_npy_path.exists():
        os.remove(sample_npy_path)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--SEED", type=int, required=True )
    p.add_argument("--DATA_PATH", type=str, required=True ) #"/PATH/TO/DATA/"
    p.add_argument("--NPY_PATH", type=str, required=True ) #"/PATH/TO/DATA/min_max_log.npy"
    p.add_argument("--BASE_DIR", type=str, required=True )
    p.add_argument("--RUN_NAME", type=str, required=True ) 
    return vars(p.parse_args())

def main(): 
    args = parse_args() 
    seed = int(args["SEED"]) 
    #MODIFY
    base_dir = (REPO_ROOT / f"{args['BASE_DIR']}")   
    os.makedirs(base_dir, exist_ok=True)
    run_name = args['RUN_NAME']
    data_path = DATA_ROOT / f"{args['DATA_PATH']}"
    npy_path = DATA_ROOT /  f"{args['NPY_PATH']}" 
    train(data_path, seed, base_dir, npy_path, run_name)
    

if __name__ == '__main__':
    main() 

