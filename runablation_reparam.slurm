#!/bin/bash
#SBATCH --job-name=gan
#SBATCH --partition=gpu
#SBATCH --qos=gpu-medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=5G
#SBATCH --time=03:59:59
#SBATCH --output=slurm-%A.out    # %A = jobID, %a = array index
#SBATCH --error=slurm-%A.err

for SEED in 0; do
    python -m RL.train --OUT_DIR ablation/reparam_gan_cat_mimic/seed${SEED} --RESULT_CSV ablation.csv --RUN_NAME reparam_gan_cat_mimic${SEED} --DATASET MIMIC --DATA_PATH MIMIC_DATA/seed${SEED} --SEED ${SEED}
done