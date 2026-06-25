#!/bin/bash
#SBATCH --job-name=2_25
#SBATCH --partition=gpu
#SBATCH --qos=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G
#SBATCH --time=03:59:59
#SBATCH --output=slurm-%A.out    # %A = jobID, %a = array index
#SBATCH --error=slurm-%A.err

python -m RL.train --OUT_DIR AIREADI_new/optuna2_trial25/seed1 --RESULT_CSV AIREADI_new/results.csv --RUN_NAME optuna2_trial25_seed1 --DATASET AIREADI --DATA_PATH AIREADI_DATA/preprocessed_data0.9_seed1 --SEED 1 --ABLATION optuna2_trial25
python -m RL.train --OUT_DIR AIREADI_new/optuna2_trial25/seed2 --RESULT_CSV AIREADI_new/results.csv --RUN_NAME optuna2_trial25_seed2 --DATASET AIREADI --DATA_PATH AIREADI_DATA/preprocessed_data0.9_seed2 --SEED 2 --ABLATION optuna2_trial25
python -m RL.train --OUT_DIR AIREADI_new/optuna2_trial25/seed3 --RESULT_CSV AIREADI_new/results.csv --RUN_NAME optuna2_trial25_seed3 --DATASET AIREADI --DATA_PATH AIREADI_DATA/preprocessed_data0.9_seed3 --SEED 3 --ABLATION optuna2_trial25
python -m RL.train --OUT_DIR AIREADI_new/optuna2_trial25/seed4 --RESULT_CSV AIREADI_new/results.csv --RUN_NAME optuna2_trial25_seed4 --DATASET AIREADI --DATA_PATH AIREADI_DATA/preprocessed_data0.9_seed4 --SEED 4 --ABLATION optuna2_trial25
python -m RL.train --OUT_DIR AIREADI_new/optuna2_trial25/seed5 --RESULT_CSV AIREADI_new/results.csv --RUN_NAME optuna2_trial25_seed5 --DATASET AIREADI --DATA_PATH AIREADI_DATA/preprocessed_data0.9_seed5 --SEED 5 --ABLATION optuna2_trial25
