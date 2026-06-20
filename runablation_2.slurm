#!/bin/bash
#SBATCH --job-name=seed2
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


python -m RL.train --OUT_DIR ablation_MIMIC/reinforce/seed2 --RESULT_CSV ablation.csv --RUN_NAME reinforce_0 --DATASET MIMIC --DATA_PATH MIMIC_DATA/seed2 --SEED 2 --ABLATION reinforce
python -m RL.train --OUT_DIR ablation_MIMIC/reparam_gan/seed2 --RESULT_CSV ablation.csv --RUN_NAME reparam_gan_0 --DATASET MIMIC --DATA_PATH MIMIC_DATA/seed2 --SEED 2 --ABLATION reparam_gan
python -m RL.train --OUT_DIR ablation_MIMIC/no_entropy/seed2 --RESULT_CSV ablation.csv --RUN_NAME no_entropy_0 --DATASET MIMIC --DATA_PATH MIMIC_DATA/seed2 --SEED 2 --ABLATION no_entropy