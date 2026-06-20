#!/bin/bash
#SBATCH --job-name=seed3
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


python -m RL.train --OUT_DIR ablation_MIMIC/reinforce/seed3 --RESULT_CSV ablation.csv --RUN_NAME reinforce_3 --DATASET MIMIC --DATA_PATH MIMIC_DATA/seed3 --SEED 3 --ABLATION reinforce
python -m RL.train --OUT_DIR ablation_MIMIC/reparam_gan/seed3 --RESULT_CSV ablation.csv --RUN_NAME reparam_gan_3 --DATASET MIMIC --DATA_PATH MIMIC_DATA/seed3 --SEED 3 --ABLATION reparam_gan
python -m RL.train --OUT_DIR ablation_MIMIC/no_entropy/seed3 --RESULT_CSV ablation.csv --RUN_NAME no_entropy_3 --DATASET MIMIC --DATA_PATH MIMIC_DATA/seed3 --SEED 3 --ABLATION no_entropy