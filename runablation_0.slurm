#!/bin/bash
#SBATCH --job-name=seed0
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


python -m RL.train --OUT_DIR ablation_MIMIC/reinforce/seed0 --RESULT_CSV ablation.csv --RUN_NAME reinforce_0 --DATASET MIMIC --DATA_PATH MIMIC_DATA/seed0 --SEED 0 --ABLATION reinforce
python -m RL.train --OUT_DIR ablation_MIMIC/reparam_gan/seed0 --RESULT_CSV ablation.csv --RUN_NAME reparam_gan_0 --DATASET MIMIC --DATA_PATH MIMIC_DATA/seed0 --SEED 0 --ABLATION reparam_gan
python -m RL.train --OUT_DIR ablation_MIMIC/no_entropy/seed0 --RESULT_CSV ablation.csv --RUN_NAME no_entropy_0 --DATASET MIMIC --DATA_PATH MIMIC_DATA/seed0 --SEED 0 --ABLATION no_entropy