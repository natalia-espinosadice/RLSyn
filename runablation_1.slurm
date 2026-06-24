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


python -m RL.train --OUT_DIR ablation/AIREADI/det_GAN/seed2 --RESULT_CSV ablation_AIREADI.csv --RUN_NAME det_GANnew_seed2 --DATASET AIREADI --DATA_PATH AIREADI_DATA/preprocessed_data0.9_seed2 --SEED 2 --ABLATION det_GAN
python -m RL.train --OUT_DIR ablation/AIREADI/det_GAN/seed3 --RESULT_CSV ablation_AIREADI.csv --RUN_NAME det_GANnew_seed3 --DATASET AIREADI --DATA_PATH AIREADI_DATA/preprocessed_data0.9_seed3 --SEED 3 --ABLATION det_GAN
python -m RL.train --OUT_DIR ablation/AIREADI/det_GAN/seed4 --RESULT_CSV ablation_AIREADI.csv --RUN_NAME det_GANnew_seed4 --DATASET AIREADI --DATA_PATH AIREADI_DATA/preprocessed_data0.9_seed4 --SEED 4 --ABLATION det_GAN

python -m RL.train --OUT_DIR ablation/AIREADI/det_GAN/seed0 --RESULT_CSV ablation_AIREADI.csv --RUN_NAME det_GANnew_seed0 --DATASET AIREADI --DATA_PATH AIREADI_DATA/preprocessed_data0.9_seed0 --SEED 0 --ABLATION det_GAN
python -m RL.train --OUT_DIR ablation/AIREADI/det_GAN/seed1 --RESULT_CSV ablation_AIREADI.csv --RUN_NAME det_GANnew_seed1 --DATASET AIREADI --DATA_PATH AIREADI_DATA/preprocessed_data0.9_seed1 --SEED 1 --ABLATION det_GAN
