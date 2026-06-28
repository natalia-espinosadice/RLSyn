#!/bin/bash
#SBATCH --job-name=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=mig
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=2G
#SBATCH --time=01:59:59
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err
#SBATCH --array=2-9

SEED=$SLURM_ARRAY_TASK_ID

python -m RL.train \
  --OUT_DIR AIREADI_new/cwc_trial25/seed${SEED} \
  --RESULT_CSV AIREADI_new/results.csv \
  --RUN_NAME cwc_trial25_seed${SEED} \
  --DATASET AIREADI \
  --DATA_PATH AIREADI_DATA/preprocessed_data0.9_seed${SEED} \
  --SEED ${SEED} \
  --ABLATION cwc_trial25