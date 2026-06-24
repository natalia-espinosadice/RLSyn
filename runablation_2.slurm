#!/bin/bash
#SBATCH --job-name=seed2
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

python -m RL.train --OUT_DIR ablation/AIREADI/reparamgan_new/seed2 --RESULT_CSV ablation_AIREADI.csv --RUN_NAME reparamgan_newseed2 --DATASET AIREADI --DATA_PATH AIREADI_DATA/preprocessed_data0.9_seed2 --SEED 2 --ABLATION reparam_gan
python -m RL.train --OUT_DIR ablation/AIREADI/reparamgan_new/seed0 --RESULT_CSV ablation_AIREADI.csv --RUN_NAME reparamgan_newseed0 --DATASET AIREADI --DATA_PATH AIREADI_DATA/preprocessed_data0.9_seed0 --SEED 0 --ABLATION reparam_gan
python -m RL.train --OUT_DIR ablation/AIREADI/reparamgan_new/seed1 --RESULT_CSV ablation_AIREADI.csv --RUN_NAME reparamgan_newseed1 --DATASET AIREADI --DATA_PATH AIREADI_DATA/preprocessed_data0.9_seed1 --SEED 1 --ABLATION reparam_gan
python -m RL.train --OUT_DIR ablation/AIREADI/reparamgan_new/seed3 --RESULT_CSV ablation_AIREADI.csv --RUN_NAME reparamgan_newseed3 --DATASET AIREADI --DATA_PATH AIREADI_DATA/preprocessed_data0.9_seed3 --SEED 3 --ABLATION reparam_gan
python -m RL.train --OUT_DIR ablation/AIREADI/reparamgan_new/seed4 --RESULT_CSV ablation_AIREADI.csv --RUN_NAME reparamgan_newseed4 --DATASET AIREADI --DATA_PATH AIREADI_DATA/preprocessed_data0.9_seed4 --SEED 4 --ABLATION reparam_gan
