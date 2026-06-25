#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KernelDensity

# =============================================================================
# CONFIG -- edit these; the script otherwise takes no arguments.
# =============================================================================
_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parent.parent
OUT_DIR = REPO_ROOT / "results" / "domias"

# DOMIAS attack (van Breugel et al., 2023). The score is the density ratio
# p_G(x) / p_R(x): a record's density under the synthetic data (p_G) over its density
# under a reference of real data (p_R). Densities are estimated with a Gaussian KDE
# (sklearn.neighbors.KernelDensity, "automated adjusted bandwidth")

# Members are real training records,
# non-members are held-out test records, and p_R is fit on the remaining TRAINING records 
# (disjoint from the members). The point of this is to use a larger reference than using 
# half of the test set as we found small references led to an unstable attack (this is also noted by the paper)
BANDWIDTH = "scott"      # automated bandwidth rule (sklearn KernelDensity param)
N_SYN = 10000            # max synthetic records used to fit p_G
N_REF = 10000            # max real records used to fit p_R
EVAL_PER_SIDE = 5000     # members and non-members per seed (balanced); AI-READI uses all it has
QUERY_BATCH = 4096       # rows per chunk when scoring log-density

# Split it by individuals for AI-READI (MIMIC doesn't need these details) 
TEST_SIZE = 0.1                    # AI-READI patient hold-out fraction (preprocessing 0.9 split)
AIREADI_LABEL = "Type 2 Diabetes"  # stratification label used by the preprocessing split

# NOTE: I know this is a mess, but its useful as a strategy to run this over the directories we used 
#       with the best hyperparams per model
#
# How to find every stored run. Each rule globs synthetic.csv files, parses the
# seed from the path, and maps it to the matching real-split directory. Runs whose
# files are absent are silently skipped, so it is safe to list models that have not
# landed yet. Optional per-rule keys:
#   * split_suffix : suffix on the real split filenames (MIMIC stores per-seed
#                    files like normalized_training_data_3.csv; "{seed}" is filled in).
#   * latest_ckpt  : if True, when several checkpoints exist per seed
#                    (.../seedN/ckpt_*/synthetic.csv) keep only the highest ckpt_NNNNN.
#
# The AI-READI trial/checkpoint picks below are the canonical Table 1 models:
# RLSyn=trial_16, EHRDiff=trial_0 (2500 epochs, matches the appendix hyperparameter
# table), EMR-WGAN=trial_9 latest checkpoint, CTAB-GAN+=trial17.
DISCOVERY_RULES: List[Dict] = [
    # ---------------------------- AI-READI ----------------------------
    # Real splits: AI-READI-FULL/preprocessed_data0.9_seed{seed}/normalized_*_data.csv
    {"model": "RLSyn", "dataset": "AI-READI",
     "synthetic_glob": "results/AIREADI/RL_AIREADI/trial_16/seed*/synthetic.csv",
     "seed_regex": r"seed(\d+)",
     "data_dir_template": "AI-READI-FULL/preprocessed_data0.9_seed{seed}"},
    {"model": "EHRDiff", "dataset": "AI-READI",
     "synthetic_glob": "results/AIREADI/EHR_DIFF/trial_0_2500_epochs/*seed*/synthetic.csv",
     "seed_regex": r"seed(\d+)",
     "data_dir_template": "AI-READI-FULL/preprocessed_data0.9_seed{seed}"},
    {"model": "EMR-WGAN", "dataset": "AI-READI",
     "synthetic_glob": "results/AIREADI/EMR_WGAN_AIREADI/trial_9/seed*/ckpt_*/synthetic.csv",
     "seed_regex": r"seed(\d+)", "latest_ckpt": True,
     "data_dir_template": "AI-READI-FULL/preprocessed_data0.9_seed{seed}"},
    {"model": "CTAB-GAN+", "dataset": "AI-READI",
     "synthetic_glob": "CTAB_GAN/CTAB_GAN_aireadi/trial17/seed*/synthetic.csv",
     "seed_regex": r"seed(\d+)",
     "data_dir_template": "AI-READI-FULL/preprocessed_data0.9_seed{seed}"},
    # ---------------------------- MIMIC-IV ----------------------------
    # Real splits: results/MIMIC/MIMIC_DATA/seeds/seed{seed}/normalized_*_data_{seed}.csv
    {"model": "RLSyn", "dataset": "MIMIC-IV",
     "synthetic_glob": "results/MIMIC/RL_MIMIC/seed*/synthetic.csv",
     "seed_regex": r"seed(\d+)", "split_suffix": "_{seed}",
     "data_dir_template": "results/MIMIC/MIMIC_DATA/seeds/seed{seed}"},
    {"model": "EHRDiff", "dataset": "MIMIC-IV",
     "synthetic_glob": "results/MIMIC/EHR_DIFF_MIMIC/seed*/synthetic.csv",
     "seed_regex": r"seed(\d+)", "split_suffix": "_{seed}",
     "data_dir_template": "results/MIMIC/MIMIC_DATA/seeds/seed{seed}"},
    {"model": "EMR-WGAN", "dataset": "MIMIC-IV",
     "synthetic_glob": "results/MIMIC/EMR_WGAN_MIMIC/seed*/ckpt_*/synthetic.csv",
     "seed_regex": r"seed(\d+)", "split_suffix": "_{seed}", "latest_ckpt": True,
     "data_dir_template": "results/MIMIC/MIMIC_DATA/seeds/seed{seed}"},
    {"model": "CTAB-GAN+", "dataset": "MIMIC-IV",
     "synthetic_glob": "results/MIMIC/CTAB_GAN_reduced/seed*/synthetic.csv",
     "seed_regex": r"seed(\d+)", "split_suffix": "_{seed}",
     "data_dir_template": "results/MIMIC/MIMIC_DATA/seeds/seed{seed}"},
]

# AI-READI feature columns (copied from CTAB_GAN/main_aireadi.py). For other
# datasets (e.g. MIMIC) every column of the real training file is used.
AIREADI_COLS: List[str] = [
    'heart_rate_mean', 'blood_glucose_mean', 'resp_rate_mean', 'stress_mean', 'blood_glucose_std', 'total_steps', 'resting_heart_rate',
    'total_kcal', 'sleep_light_hrs', 'sleep_deep_hrs', 'sleep_rem_hrs', 'sleep_awake_hrs', 'act_generic_hrs', 'act_running_hrs', 'act_walking_hrs', 'act_sedentary_hrs',
    'A/G Ratio_value', 'ALT (IU/L)_value (IU/L)', 'AST (IU/L)_value (IU/L)', 'Albumin (g/dL)_value (g/dL)', 'Alkaline Phosphatase (IU/L)_value (IU/L)', 'BUN (mg/dL)_value (mg/dL)', 'BUN/Creatinine ratio_value', "Bilirubin Total (mg/dL)_value (mg/dL)", 'C-Peptide (ng/mL)_value (ng/mL)',
    'CRP - HS (mg/L)_value (mg/L)', 'Calcium (mg/dL)_value (mg/dL)', 'Carbon Dioxide, Total (mEq/L)_value (mEq/L)', 'Chloride (mEq/L)_value (mEq/L)', 'Creatinine (mg/dL)_value (md/dL)', 'Globulin, Total (g/dL)_value (g/dL)', 'Glucose (mg/dL)_value (mg/dL)', 'HDL Cholesterol (mg/dL)_value (mg/dL)',
    'INSULIN (ng/mL)_value (ng/mL)', 'LDL Cholesterol Calculation (mg/dL)_value (mg/dL)', 'Potassium (mEq/L)_value (mEq/L)', 'Protein, Total (g/dL)_value (g/dL)', 'Sodium (mEq/L)_value (mEq/L)', 'Total Cholesterol (mg/dL)_value (mg/dL)', 'Triglycerides (mg/dL)_value (mg/dL)', 'Urine Albumin (mg/dL)_value (mg/DL)', 'Urine Creatinine (mg/dL)_value (mg/DL)',
    'clock_visuospatial_executive_time_value', 'delayed_recall_with_no_clue_time_value', 'digitspan_time_value', 'memory_trial1_time_value', 'memory_trial2_time_value', 'moca_abstraction_time_value', 'moca_orientation_time_value', 'moca_total_score_value', 'naming_time_value', 'repetition_time_value', 'subtraction_time_value',
    'cube_visuospatial_executive_time_value', 'lettera_time_value', 'trails_visuospatial_executive_time_value', 'moca_combined_mis_score_value', 'memory_trial1_value', 'memory_trial2_value', 'moca_abstraction_value', 'moca_orientation_value', 'naming_value', 'repetition_value', 'subtraction_value', 'Rate', 'PR', 'QRSD', 'QT', 'QTc', 'P', 'QRS', 'T',
    'Age-related macular degeneration', 'Arthritis', 'Cancer', 'Cataracts (1+ eyes)', 'Chronic pullmonary problems', 'Circulation problems', 'Diabetic retinopathy (1+)', 'Digestive problems', 'Marijuana user', 'Dry eye (1+)', 'Glaucoma (1+)', 'Hearing impairment',
    'Heart attack', 'High blood cholesterol', 'High blood pressure', 'Kidney problems', 'Low blood pressure', 'Mild cognitive impairmen', 'Multiple sclerosis', 'Obesity', 'Osteoporosis', 'Other heart issues (pacemaker)', 'Other neurological conditions',
    "Parkinson's disease", 'Retinal vascular occlusion', 'Stroke', 'Type 2 Diabetes', 'Urinary problems',
    'ABNORMAL', 'BORDERLINE', 'NORMAL', 'OTHERWISE NORMAL', 'cube_visuospatial_executive_value', 'fluency_language_value', 'lettera_value', 'trails_visuospatial_executive_value',
]


def discover_runs() -> List[Dict]:
    """Expand DISCOVERY_RULES into concrete (model, dataset, seed, paths) records."""
    runs: List[Dict] = []
    for rule in DISCOVERY_RULES:
        matches = sorted(REPO_ROOT.glob(rule["synthetic_glob"]))
        if not matches:
            print(f"  [skip] no files for {rule['model']}/{rule['dataset']} ({rule['synthetic_glob']})")
            continue
        # seed -> chosen synthetic.csv. When several checkpoints exist per seed and
        # latest_ckpt is set, keep the one with the highest ckpt_NNNNN.
        by_seed: Dict[int, Path] = {}
        for syn in matches:
            m = re.search(rule["seed_regex"], str(syn))
            if not m:
                continue
            seed = int(m.group(1))
            if rule.get("latest_ckpt") and seed in by_seed:
                ck = lambda p: int((re.search(r"ckpt_(\d+)", str(p)) or [0, 0])[1])
                if ck(syn) <= ck(by_seed[seed]):
                    continue
            by_seed[seed] = syn
        suffix_tmpl = rule.get("split_suffix", "")
        for seed, syn in sorted(by_seed.items()):
            runs.append({
                "model": rule["model"], "dataset": rule["dataset"], "seed": seed,
                "synthetic_csv": syn,
                "data_dir": REPO_ROOT / rule["data_dir_template"].format(seed=seed),
                "split_suffix": suffix_tmpl.format(seed=seed),
            })
    return runs


def _read_features(csv_path: Path, cols: List[str]) -> pd.DataFrame:
    """Read a CSV, drop any leading unnamed index column, and select ``cols`` by name."""
    df = pd.read_csv(csv_path)
    df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed")]]
    return df[[c for c in cols if c in df.columns]].copy()


def _aireadi_test_patients(dfp: pd.DataFrame, seed: int) -> set:
    """Reconstruct the preprocessing patient hold-out (stratified by the T2D label)."""
    pats = dfp.groupby("patient_id")[AIREADI_LABEL].max().rename("label").reset_index()
    test_p = pats.groupby("label", group_keys=False).apply(
        lambda x: x.sample(frac=TEST_SIZE, random_state=seed))
    return set(test_p["patient_id"])


def load_run_arrays(run: Dict) -> Optional[Dict]:
    """Load the training member pool, held-out non-members, and synthetic data for one run.

    Returns float32 arrays ``member_pool`` (training records, later split into members +
    reference), ``non_members`` (held-out test records) and ``synthetic``, plus
    ``member_pids`` (patient id per member-pool row, or None) and ``n_test_patients`` so
    the member/reference split can be done at the patient level for AI-READI. Returns None
    (skip) if a required file is missing.
    """
    data_dir, syn_path = run["data_dir"], run["synthetic_csv"]

    if "readi" in run["dataset"].lower():
        # AI-READI: patient ids are needed so members and reference are patient-disjoint.
        dfp_path = data_dir / "preprocessed_data_with_patients.csv"
        if not dfp_path.exists() or not syn_path.exists():
            print(f"    [skip] missing {dfp_path if not dfp_path.exists() else syn_path}")
            return None
        dfp = pd.read_csv(dfp_path)
        dfp = dfp.loc[:, [c for c in dfp.columns if not str(c).startswith("Unnamed")]]
        cols = [c for c in AIREADI_COLS if c in dfp.columns]
        is_test = dfp["patient_id"].isin(_aireadi_test_patients(dfp, run["seed"]))
        member_pool = dfp.loc[~is_test, cols].to_numpy(dtype=np.float32)
        member_pids = dfp.loc[~is_test, "patient_id"].to_numpy()
        non_members = dfp.loc[is_test, cols].to_numpy(dtype=np.float32)
        n_test_patients = int(dfp.loc[is_test, "patient_id"].nunique())
    else:
        # MIMIC etc.: one record per individual -> row-level split, normalized files.
        suffix = run.get("split_suffix", "")
        train_path = data_dir / f"normalized_training_data{suffix}.csv"
        test_path = data_dir / f"normalized_testing_data{suffix}.csv"
        for p in (syn_path, train_path, test_path):
            if not p.exists():
                print(f"    [skip] missing {p}")
                return None
        train_raw = pd.read_csv(train_path)
        train_raw = train_raw.loc[:, [c for c in train_raw.columns if not str(c).startswith("Unnamed")]]
        cols = list(train_raw.columns)
        member_pool = _read_features(train_path, cols).to_numpy(dtype=np.float32)
        member_pids, n_test_patients = None, None
        non_members = _read_features(test_path, cols).to_numpy(dtype=np.float32)

    synthetic = _read_features(syn_path, cols).reindex(columns=cols)
    if synthetic.isnull().any().any():
        print(f"    [skip] synthetic data missing expected columns for {syn_path}")
        return None
    return {
        "member_pool": member_pool,
        "member_pids": member_pids,
        "n_test_patients": n_test_patients,
        "non_members": non_members,
        "synthetic": synthetic.to_numpy(dtype=np.float32),
    }


def _subsample(arr: np.ndarray, cap: int, rng: np.random.Generator) -> np.ndarray:
    """Randomly subsample rows down to ``cap`` (no-op if already small enough)."""
    return arr if len(arr) <= cap else arr[rng.choice(len(arr), cap, replace=False)]


def _fit_kde(arr: np.ndarray) -> KernelDensity:
    """Fit a Gaussian KDE (sklearn KernelDensity, automated bandwidth), as in the paper."""
    return KernelDensity(kernel="gaussian", bandwidth=BANDWIDTH).fit(arr)


def _logpdf(kde: KernelDensity, query: np.ndarray) -> np.ndarray:
    """Evaluate log-density of each query row, chunked to cap memory."""
    return np.concatenate([kde.score_samples(query[i:i + QUERY_BATCH])
                           for i in range(0, len(query), QUERY_BATCH)])


def domias_auc(arrays: Dict, seed: int) -> float:
    """DOMIAS membership-inference AUC for one run.

    Members are training records, non-members are held-out test records, and p_R is fit
    on the remaining training records (the reference) -- kept disjoint from the members
    at the INDIVIDUAL level (by patient for AI-READI; by row otherwise). p_G is fit on
    the synthetic data. The score is log p_G(x) - log p_R(x) (rank-identical to the DOMIAS
    ratio p_G/p_R, without exp() overflow); we report roc_auc(is_member, score).
    """
    rng = np.random.default_rng(seed)
    pool, pids = arrays["member_pool"], arrays["member_pids"]
    test, syn = arrays["non_members"], arrays["synthetic"]

    n_eval = min(EVAL_PER_SIDE, len(test))
    non = test[rng.choice(len(test), n_eval, replace=False)]        # non-members (label 0)
    if pids is not None:
        # Patient-level: members are whole patients (count-matched to the held-out
        # patients); the remaining patients form the reference -> no shared individuals.
        upats = np.unique(pids)
        k = min(arrays["n_test_patients"] or len(upats), len(upats))
        mem_pats = set(rng.choice(upats, size=k, replace=False))
        mmask = np.isin(pids, list(mem_pats))
        members = _subsample(pool[mmask], n_eval, rng)
        reference = pool[~mmask]
    else:
        perm = rng.permutation(len(pool))
        members = pool[perm[:n_eval]]                               # members (label 1)
        reference = pool[perm[n_eval:]]                            # disjoint reference

    kde_g = _fit_kde(_subsample(syn, N_SYN, rng))                   # p_G from synthetic
    kde_r = _fit_kde(_subsample(reference, N_REF, rng))             # p_R from real reference
    x = np.concatenate([members, non], axis=0)
    y = np.concatenate([np.ones(len(members)), np.zeros(len(non))])
    log_ratio = np.nan_to_num(_logpdf(kde_g, x) - _logpdf(kde_r, x),
                              nan=0.0, posinf=1e30, neginf=-1e30)
    return float(roc_auc_score(y, log_ratio))


SHARD_DIR = OUT_DIR / "_shards"   # per-shard incremental result files live here


def _shard_spec() -> tuple:
    """Parse the optional DOMIAS_SHARD='k/N' env var (default '0/1' = run everything).

    Sharding lets several processes (e.g. one per CPU core) split the run list: shard k of N
    handles runs[k::N]. Each writes its own _shards/per_run_k.csv, and aggregation
    reads every shard file -- so the no-env-var default still runs the whole thing.
    """
    spec = os.environ.get("DOMIAS_SHARD", "0/1")
    k, n = (spec.split("/") + ["1"])[:2]
    return int(k), int(n)


def _append_row(path: Path, row: Dict) -> None:
    """Append one result row to a shard CSV (crash-resilient: progress is saved live)."""
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def aggregate_from_shards() -> None:
    """Merge all _shards/per_run_*.csv into the final per-run CSV, summary, and .tex."""
    files = sorted(SHARD_DIR.glob("per_run_*.csv"))
    if not files:
        print("[done] no shard results to aggregate.")
        return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.drop_duplicates(subset=["model", "dataset", "seed"], keep="first")
    aggregate_and_write(df.to_dict("records"))


def aggregate_and_write(per_run: List[Dict]) -> None:
    """Write the per-run CSV, the mean/std summary, and a drop-in .tex table."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(per_run)
    df.to_csv(OUT_DIR / "domias_per_run.csv", index=False)
    if df.empty:
        print("[done] no successful runs (synthetic data may not be present yet).")
        return

    summary = df.groupby(["dataset", "model"])["domias_auc"].agg(["mean", "std", "count"]).reset_index()
    summary.to_csv(OUT_DIR / "domias_summary.csv", index=False)
    print("\n=== DOMIAS membership-inference AUC (mean +/- std over seeds) ===")
    print(summary.to_string(index=False))

    datasets = sorted(df["dataset"].unique())
    preferred = ["EMR-WGAN", "EHRDiff", "RLSyn", "CTAB-GAN+"]
    models = [m for m in preferred if m in set(df["model"])] + \
             [m for m in sorted(df["model"].unique()) if m not in preferred]

    def cell(model: str, dataset: str) -> str:
        sub = summary[(summary["model"] == model) & (summary["dataset"] == dataset)]
        if sub.empty:
            return "--"
        std = sub["std"].iloc[0]
        return f"{sub['mean'].iloc[0]:.3f} $\\pm$ {0.0 if pd.isna(std) else std:.3f}"

    lines = [
        r"% Auto-generated by evaluation/domias_attack/run_domias.py (DOMIAS attack).",
        r"% Drop-in replacement for paper/tables/privacy_results.tex.",
        r"\begin{table}[h!]", r"    \centering",
        r"    \caption{AUC of DOMIAS membership inference attacks against synthetic data.}",
        rf"    \begin{{tabular}}{{c|{'|'.join(['c'] * len(datasets))}}}",
        r"        \toprule",
        "         & " + " & ".join(f"\\textbf{{{d}}}" for d in datasets) + r" \\",
        r"        \cmidrule{1-" + str(len(datasets) + 1) + "}",
    ]
    for model in models:
        name = r"\modelname" if model == "RLSyn" else model
        lines.append(f"        {name} & " + " & ".join(cell(model, d) for d in datasets) + r" \\")
    lines += [r"        \bottomrule", r"    \end{tabular}",
              r"    \label{tab:grouped_domias}", r"\end{table}"]
    (OUT_DIR / "privacy_results_domias.tex").write_text("\n".join(lines) + "\n")
    print(f"\n[out] {OUT_DIR}/  (domias_per_run.csv, domias_summary.csv, privacy_results_domias.tex)")


def main() -> None:
    k, n = _shard_spec()
    runs = discover_runs()
    mine = runs[k::n]
    print(f"[discover] {len(runs)} runs; shard {k}/{n} -> {len(mine)} runs (KDE)")
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    shard_file = SHARD_DIR / f"per_run_{k}.csv"
    shard_file.unlink(missing_ok=True)  # start this shard fresh; rows appended as computed

    for i, run in enumerate(mine, 1):
        print(f"\n[{k}/{n} {i}/{len(mine)}] {run['model']}/{run['dataset']}/seed{run['seed']}")
        arrays = load_run_arrays(run)
        if arrays is None:
            continue
        try:
            auc = domias_auc(arrays, run["seed"])
        except Exception as exc:
            print(f"    [error] {exc}")
            continue
        print(f"    DOMIAS AUC = {auc:.4f}")
        _append_row(shard_file, {"model": run["model"], "dataset": run["dataset"],
                                 "seed": run["seed"], "domias_auc": auc})

    aggregate_from_shards()


if __name__ == "__main__":
    main()
