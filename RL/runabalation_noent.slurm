#!/bin/bash
#SBATCH --job-name=og0
#SBATCH --partition=gpu
#SBATCH --qos=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=5G
#SBATCH --time=03:59:59
#SBATCH --output=slurm-%A.out    # %A = jobID, %a = array index
#SBATCH --error=slurm-%A.err


python -m RL.train --OUT_DIR ablation/no_entropy/seed0 --RESULT_CSV ablation --RUN_NAME no_entropy_0 --DATASET AIREADI --DATA_PATH AIREADI_DATA/preprocessed_data0.9_seed0 --SEED 0