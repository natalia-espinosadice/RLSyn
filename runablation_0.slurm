#!/bin/bash
#SBATCH --job-name=ainew6OG
#SBATCH --partition=gpu
#SBATCH --qos=gpu-medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=25G
#SBATCH --time=11:59:59
#SBATCH --output=slurm-%A.out    # %A = jobID, %a = array index
#SBATCH --error=slurm-%A.err

python -m RL.optuna_search_aireadi_full --STUDY_NAME aireadioptunanew_4 --DATA_PATH AIREADI_DATA/preprocessed_data0.9_seed0