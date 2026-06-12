#!/bin/bash
#SBATCH --job-name=val
#SBATCH --partition=gpu
#SBATCH --qos=gpu-medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=5G
#SBATCH --time=07:59:59
#SBATCH --output=slurm-%A.out    # %A = jobID, %a = array index
#SBATCH --error=slurm-%A.err

for SEED in 0 1 2 3 4; do
    python -m RL.train --OUT_DIR ablation/no_value/seed${SEED} --RESULT_CSV ablation.csv --RUN_NAME no_value${SEED} --DATASET AIREADI --DATA_PATH AIREADI_DATA/preprocessed_data0.9_seed${SEED} --SEED ${SEED}
done