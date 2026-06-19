import os
import csv
import time
import optuna
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from optuna.visualization import (plot_optimization_history, plot_param_importances, plot_pareto_front,)
from model.ctabgan import CTABGAN
from evaluation.evaluate_aireadi import train_on_synth_test_on_hold, train_on_real_test_on_synth

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
DATA_ROOT = REPO_ROOT.parent / "AIREADI_DATA" / "preprocessed_data0.9_seed0"


def log_result(results_csv, tag, seed, epochs, batch_size, num_channels, random_dim,
               s2h_auc, s2h_acc, r2s_auc, r2s_acc, elapsed_time):
    p = Path(results_csv)
    p.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "tag": tag, "seed": seed, "epochs": epochs,
        "batch_size": batch_size, "num_channels": num_channels, "random_dim": random_dim,
        "s2h_auc": s2h_auc, "s2h_acc": s2h_acc,
        "r2s_auc": r2s_auc, "r2s_acc": r2s_acc,
        "elapsed_time": elapsed_time,
    }
    write_header = not os.path.exists(results_csv)
    with open(results_csv, "a", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def evaluate_model(df_hold_norm, df_real_with_patients_norm, df_syn_norm,
                   out_dir, seed, results_csv, epochs, batch_size, num_channels,
                   random_dim, trial_tag, elapsed_time):
    ten_percent = 490
    if len(df_hold_norm) > ten_percent:
        df_hold_norm = df_hold_norm.sample(
            frac=ten_percent / len(df_hold_norm), random_state=seed
        ).reset_index(drop=True)

    s2h_auc, s2h_acc = train_on_synth_test_on_hold(df_syn_norm, df_hold_norm, f"{out_dir}/synth_to_hold", seed)
    r2s_auc, r2s_acc = train_on_real_test_on_synth(df_real_with_patients_norm, df_syn_norm, f"{out_dir}/real_to_synth", seed)

    with open(f"{out_dir}/eval.txt", "a") as f:
        f.write(f"s2h_auc: {s2h_auc}\ns2h_acc: {s2h_acc}\nr2s_auc: {r2s_auc}\nr2s_acc: {r2s_acc}\n")
        f.write(f"epochs: {epochs}, batch_size: {batch_size}, num_channels: {num_channels}, random_dim: {random_dim}, elapsed: {elapsed_time:.2f}min\n")

    log_result(results_csv, trial_tag, seed, epochs, batch_size, num_channels, random_dim,
               s2h_auc, s2h_acc, r2s_auc, r2s_acc, elapsed_time)

    return s2h_auc, r2s_auc


def objective(trial: optuna.Trial, study_name: str):
    epochs       = trial.suggest_int("epochs",       100, 200, step=50)
    batch_size   = trial.suggest_categorical("batch_size",   [200, 500, 1000])
    num_channels = trial.suggest_categorical("num_channels", [32, 64, 128])
    random_dim   = trial.suggest_categorical("random_dim",   [64, 100, 128])
    base_dir  = REPO_ROOT / study_name
    trial_dir = base_dir / f"trial_{trial.number}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    save_syn  = trial_dir / "synthetic.csv"
    results_csv = base_dir / "results.csv"
    seed = 2023
    train_csv = DATA_ROOT / "normalized_training_data.csv"
    test_csv  = DATA_ROOT / "normalized_testing_data.csv"
    real_csv  = DATA_ROOT / "preprocessed_data_with_patients.csv"
    raw_csv = train_csv  # CTAB-GAN reads from a csv path directly
    # --- train ---
    start_time = time.time()
    try:
        synthesizer = CTABGAN(
            raw_csv_path=str(raw_csv),
            test_ratio=0.0,                  # already using the train split
            categorical_columns=CAT_COLS,
            log_columns=[],
            mixed_columns={},
            general_columns=[],
            non_categorical_columns=[],
            integer_columns=[],
            problem_type={}
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
    except Exception as e:
        raise optuna.TrialPruned(f"Training failed: {e}")

    elapsed_time = (time.time() - start_time) / 60

    # --- evaluate ---
    df_hold_norm              = pd.read_csv(test_csv)[NUM_COLS + CAT_COLS]
    df_real_norm_with_patients = pd.read_csv(real_csv)[NUM_COLS + CAT_COLS + ['patient_id']]
    df_syn_norm               = pd.read_csv(save_syn)[NUM_COLS + CAT_COLS]

    try:
        s2h_auc, r2s_auc = evaluate_model(
            df_hold_norm, df_real_norm_with_patients, df_syn_norm,
            trial_dir, seed, results_csv,
            epochs, batch_size, num_channels, random_dim,
            str(trial.number), elapsed_time
        )
        return s2h_auc, r2s_auc
    except Exception as e:
        raise optuna.TrialPruned(f"Evaluation failed: {e}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--STUDY_NAME", type=str, required=True)
    return vars(p.parse_args())


def main():
    args = parse_args()
    study_name = args["STUDY_NAME"]
    base_dir = REPO_ROOT / study_name
    base_dir.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{study_name}.db",
        load_if_exists=True,
        directions=("maximize", "maximize"),
    )
    study.optimize(
        lambda trial: objective(trial, study_name),
        n_trials=20,
    )

    # --- save best trials ---
    best_save = base_dir / "best_trial_summary.txt"
    with open(best_save, "w") as f:
        pareto = study.best_trials
        f.write(f"Pareto front has {len(pareto)} trial(s)\n")
        for t in pareto:
            f.write(f"{t.number}, {t.values}, {t.params}\n")

    objectives = ["s2h_auc", "r2s_auc"]

    fig = plot_pareto_front(study, target_names=objectives)
    fig.write_html(str(base_dir / "pareto_front.html"))

    for i, name in enumerate(objectives):
        fig = plot_optimization_history(study, target=lambda t, idx=i: t.values[idx], target_name=name)
        fig.write_html(str(base_dir / f"opt_history_{name}.html"))

        fig = plot_param_importances(study, target=lambda t, idx=i: t.values[idx], target_name=name)
        fig.write_html(str(base_dir / f"param_importance_{name}.html"))


if __name__ == "__main__":
    main()