#!/bin/bash
#SBATCH --job-name=ai               # Job name
#SBATCH --nodes=1                       # We only need 1 node
#SBATCH --ntasks=1
#SBATCH --partition=mig                      # 1 task
#SBATCH --cpus-per-task=1              # 8 CPU cores
#SBATCH --gres=gpu:1                    # 1 GPU
#SBATCH --mem=2G                       # 16 GB of system RAM
#SBATCH --time=11:59:59              # Max run time (HH:MM:SS)
#SBATCH --output=slurm-%j.out           # Standard output log (%j expands to jobID)
#SBATCH --error=slurm-%j.err            # Standard error log

python -m RL.optuna_search_aireadi_full --STUDY_NAME aireadioptunanew_cwc --DATA_PATH AIREADI_DATA/preprocessed_data0.9_seed0