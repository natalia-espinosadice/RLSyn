#!/bin/bash
#SBATCH --job-name=clip
#SBATCH --partition=gpu
#SBATCH --qos=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=25G
#SBATCH --time=03:59:59
#SBATCH --output=slurm-%A.out    # %A = jobID, %a = array index
#SBATCH --error=slurm-%A.err


python -m RL.train --OUT_DIR ablation/AIREADI/no_clip/seed0 --RESULT_CSV ablation_AIREADI.csv --RUN_NAME no_clip_seed0 --DATASET AIREADI --DATA_PATH AIREADI_DATA/preprocessed_data0.9_seed0 --SEED 0 --ABLATION no_clip
python -m RL.train --OUT_DIR ablation/AIREADI/no_clip/seed1 --RESULT_CSV ablation_AIREADI.csv --RUN_NAME no_clip_seed1 --DATASET AIREADI --DATA_PATH AIREADI_DATA/preprocessed_data0.9_seed1 --SEED 1 --ABLATION no_clip
