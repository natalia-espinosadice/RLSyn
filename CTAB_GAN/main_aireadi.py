import os
import csv
import time
import optuna
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from CTAB_GAN.model.ctabgan import CTABGAN
from evaluation.evaluate_aireadi import (get_histograms, train_on_synth_test_on_hold, mem_risk, train_on_synth_test_on_real,  train_on_real_test_on_real, train_on_real_test_on_synth, latent_cluster_analysis, get_column_wise_correlations)

NUM_COLS = [
    'heart_rate_mean', 'blood_glucose_mean', 'resp_rate_mean', 'stress_mean', 'blood_glucose_std', 'total_steps', 'resting_heart_rate',
    'total_kcal', 'sleep_light_hrs', 'sleep_deep_hrs', 'sleep_rem_hrs', 'sleep_awake_hrs', 'act_generic_hrs', 'act_running_hrs', 'act_walking_hrs', 'act_sedentary_hrs',
    'A/G Ratio_value', 'ALT (IU/L)_value (IU/L)', 'AST (IU/L)_value (IU/L)', 'Albumin (g/dL)_value (g/dL)', 'Alkaline Phosphatase (IU/L)_value (IU/L)', 'BUN (mg/dL)_value (mg/dL)', 'BUN/Creatinine ratio_value', "Bilirubin Total (mg/dL)_value (mg/dL)", 'C-Peptide (ng/mL)_value (ng/mL)',
    'CRP - HS (mg/L)_value (mg/L)', 'Calcium (mg/dL)_value (mg/dL)', 'Carbon Dioxide, Total (mEq/L)_value (mEq/L)', 'Chloride (mEq/L)_value (mEq/L)', 'Creatinine (mg/dL)_value (md/dL)', 'Globulin, Total (g/dL)_value (g/dL)', 'Glucose (mg/dL)_value (mg/dL)', 'HDL Cholesterol (mg/dL)_value (mg/dL)',
    'INSULIN (ng/mL)_value (ng/mL)', 'LDL Cholesterol Calculation (mg/dL)_value (mg/dL)', 'Potassium (mEq/L)_value (mEq/L)', 'Protein, Total (g/dL)_value (g/dL)', 'Sodium (mEq/L)_value (mEq/L)', 'Total Cholesterol (mg/dL)_value (mg/dL)', 'Triglycerides (mg/dL)_value (mg/dL)', 'Urine Albumin (mg/dL)_value (mg/DL)', 'Urine Creatinine (mg/dL)_value (mg/DL)',
    'clock_visuospatial_executive_time_value', 'delayed_recall_with_no_clue_time_value', 'digitspan_time_value', 'memory_trial1_time_value', 'memory_trial2_time_value', 'moca_abstraction_time_value', 'moca_orientation_time_value', 'moca_total_score_value', 'naming_time_value', 'repetition_time_value', 'subtraction_time_value',
    'cube_visuospatial_executive_time_value', 'lettera_time_value', 'trails_visuospatial_executive_time_value', 'moca_combined_mis_score_value', 'memory_trial1_value', 'memory_trial2_value', 'moca_abstraction_value', 'moca_orientation_value', 'naming_value', 'repetition_value', 'subtraction_value', 'Rate', 'PR', 'QRSD', 'QT', 'QTc', 'P', 'QRS', 'T',
]
CAT_COLS = [
    'Age-related macular degeneration', 'Arthritis', 'Cancer', 'Cataracts (1+ eyes)', 'Chronic pullmonary problems', 'Circulation problems', 'Diabetic retinopathy (1+)', 'Digestive problems', 'Marijuana user', 'Dry eye (1+)', 'Glaucoma (1+)', 'Hearing impairment',
    'Heart attack', 'High blood cholesterol', 'High blood pressure', 'Kidney problems', 'Low blood pressure', 'Mild cognitive impairmen', 'Multiple sclerosis', 'Obesity', 'Osteoporosis', 'Other heart issues (pacemaker)', 'Other neurological conditions',
    "Parkinson's disease", 'Retinal vascular occlusion', 'Stroke', 'Type 2 Diabetes', 'Urinary problems',
    'ABNORMAL', 'BORDERLINE', 'NORMAL', 'OTHERWISE NORMAL', 'cube_visuospatial_executive_value', 'fluency_language_value', 'lettera_value', 'trails_visuospatial_executive_value'
]

REPO_ROOT = Path(__file__).resolve().parent
DATA_ROOT = REPO_ROOT.parent / "AIREADI_DATA"  


def log_result(RESULTS_CSV, tag, seed, cwc, 
               r2r_auc, r2r_acc, s2h_auc, s2h_acc, s2r_auc, s2r_acc, r2s_auc, r2s_acc, real_mem_auc, real_mem_auc_bal, synth_mem_auc, synth_mem_auc_bal, elapsed_time):
    p = Path(RESULTS_CSV)
    p.parent.mkdir(parents=True, exist_ok=True)
    row = {"tag": tag, "seed" : seed,
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

def evaluate_model_aireadi(df_real, df_hold, df_train, df_syn, df_hold_norm, df_train_norm, df_syn_norm, df_real_with_patients_norm, OUT_DIR, RESULT_CSV, RUN_NAME, SEED, NUM_COLS, CAT_COLS, elapsed_time): 
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
    log_result(RESULT_CSV, RUN_NAME, SEED, cwc, r2r_auc, r2r_acc, s2h_auc, s2h_acc, s2r_auc, s2r_acc, r2s_auc, r2s_acc, mem_aucs["real_roc_auc"], mem_aucs["real_roc_auc_bal"], mem_aucs["synth_roc_auc"], mem_aucs["synth_roc_auc_bal"], elapsed_time)




def train(base_dir, seed, epochs, batch_size, num_channels, random_dim ):
    data_path = DATA_ROOT / f"preprocessed_data0.9_seed{seed}"
    trial_dir = base_dir /  "trial17" / f"seed{seed}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    tag = f"trial17_seed{seed}"
    save_syn  = trial_dir / "synthetic.csv"
    npy_path = data_path / "min_max_log.npy"
    results_csv = base_dir / "results.csv"
    save_syn_rescaled = trial_dir / "synthetic_rescaled.csv"
    train_csv = data_path / "normalized_training_data.csv"
    test_csv  = data_path / "normalized_testing_data.csv"
    real_csv  = data_path / "preprocessed_data_with_patients.csv"
    raw_csv = train_csv 
    start_time = time.time()
    synthesizer = CTABGAN(
        raw_csv_path=str(raw_csv),
        test_ratio=0.01,                  # already using the train split
        categorical_columns=CAT_COLS,
        log_columns=[],
        mixed_columns={},
        general_columns=[],
        non_categorical_columns=[],
        integer_columns=[],
        problem_type={"Classification": "Type 2 Diabetes"}
    )
    synthesizer.synthesizer.epochs      = epochs
    synthesizer.synthesizer.batch_size  = batch_size
    synthesizer.synthesizer.num_channels = num_channels
    synthesizer.synthesizer.random_dim  = random_dim
    synthesizer.synthesizer.private     = False

    synthesizer.fit()
    df_syn = synthesizer.generate_samples()
    df_syn = df_syn[CAT_COLS + NUM_COLS]
    df_syn.to_csv(save_syn, index=False)
    feature_range = np.load(npy_path, allow_pickle=True).item()
    for col in NUM_COLS:
        xmin, xmax = feature_range[col]
        df_syn[col] = (1.0 - df_syn[col]) * xmin + df_syn[col] * xmax
    df_syn.to_csv(save_syn_rescaled)

    elapsed_time = (time.time() - start_time) / 60
    # --- evaluate ---
    df_train_norm = pd.read_csv(f"{data_path}/normalized_training_data.csv")[NUM_COLS + CAT_COLS]
    df_hold_norm =  pd.read_csv(f"{data_path}/normalized_testing_data.csv")[NUM_COLS + CAT_COLS]
    df_real_with_patients_norm = pd.read_csv(f"{data_path}/preprocessed_data_with_patients.csv")[NUM_COLS + CAT_COLS + ['patient_id']]
    df_train = pd.read_csv(f"{data_path}/original_training_data.csv")[NUM_COLS + CAT_COLS]
    df_hold =  pd.read_csv(f"{data_path}/original_testing_data.csv")[NUM_COLS + CAT_COLS]
    df_real = pd.concat([df_train, df_hold])[NUM_COLS + CAT_COLS]
    df_syn = pd.read_csv(save_syn_rescaled)[NUM_COLS + CAT_COLS]
    df_syn_norm = pd.read_csv(save_syn)[NUM_COLS + CAT_COLS]
    evaluate_model_aireadi(df_real, df_hold, df_train, df_syn, df_hold_norm, df_train_norm, df_syn_norm, df_real_with_patients_norm, trial_dir, results_csv, tag, seed, NUM_COLS, CAT_COLS, elapsed_time)
    


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=str, required=True)
    return vars(p.parse_args())


def main():
    args = parse_args()
    seed = int(args["seed"])
    base_dir = REPO_ROOT / "CTAB_GAN_aireadi" # / "trial17" / f"seed{seed}"
    base_dir.mkdir(parents=True, exist_ok=True)
    epochs = 150 
    batch_size = 200
    num_channels =128 
    random_dim = 128 
    train(base_dir, seed, epochs, batch_size, num_channels, random_dim )
    

    #17, [0.8622476035868892, 0.6341229954038928], {'epochs': 150, 'batch_size': 200, 'num_channels': 128, 'random_dim': 128}
    #19, [0.8020232941661514, 0.6487673993059703], {'epochs': 200, 'batch_size': 200, 'num_channels': 64, 'random_dim': 64}


  
if __name__ == "__main__":
    main()