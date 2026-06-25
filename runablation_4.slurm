#!/bin/bash
#SBATCH --job-name=2_26
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

python -m RL.train --OUT_DIR AIREADI_new/optuna2_trial26/seed1 --RESULT_CSV AIREADI_new/results.csv --RUN_NAME optuna2_trial26_seed1 --DATASET AIREADI --DATA_PATH AIREADI_DATA/preprocessed_data0.9_seed1 --SEED 1 --ABLATION optuna2_trial26
python -m RL.train --OUT_DIR AIREADI_new/optuna2_trial26/seed2 --RESULT_CSV AIREADI_new/results.csv --RUN_NAME optuna2_trial26_seed2 --DATASET AIREADI --DATA_PATH AIREADI_DATA/preprocessed_data0.9_seed2 --SEED 2 --ABLATION optuna2_trial26
python -m RL.train --OUT_DIR AIREADI_new/optuna2_trial26/seed3 --RESULT_CSV AIREADI_new/results.csv --RUN_NAME optuna2_trial26_seed3 --DATASET AIREADI --DATA_PATH AIREADI_DATA/preprocessed_data0.9_seed3 --SEED 3 --ABLATION optuna2_trial26
python -m RL.train --OUT_DIR AIREADI_new/optuna2_trial26/seed4 --RESULT_CSV AIREADI_new/results.csv --RUN_NAME optuna2_trial26_seed4 --DATASET AIREADI --DATA_PATH AIREADI_DATA/preprocessed_data0.9_seed4 --SEED 4 --ABLATION optuna2_trial26
python -m RL.train --OUT_DIR AIREADI_new/optuna2_trial26/seed5 --RESULT_CSV AIREADI_new/results.csv --RUN_NAME optuna2_trial26_seed5 --DATASET AIREADI --DATA_PATH AIREADI_DATA/preprocessed_data0.9_seed5 --SEED 5 --ABLATION optuna2_trial26
python -m RL.train --OUT_DIR AIREADI_new/optuna2_trial26/seed6 --RESULT_CSV AIREADI_new/results.csv --RUN_NAME optuna2_trial26_seed6 --DATASET AIREADI --DATA_PATH AIREADI_DATA/preprocessed_data0.9_seed6 --SEED 6 --ABLATION optuna2_trial26
python -m RL.train --OUT_DIR AIREADI_new/optuna2_trial26/seed7 --RESULT_CSV AIREADI_new/results.csv --RUN_NAME optuna2_trial26_seed7 --DATASET AIREADI --DATA_PATH AIREADI_DATA/preprocessed_data0.9_seed7 --SEED 7 --ABLATION optuna2_trial26
python -m RL.train --OUT_DIR AIREADI_new/optuna2_trial26/seed8 --RESULT_CSV AIREADI_new/results.csv --RUN_NAME optuna2_trial26_seed8 --DATASET AIREADI --DATA_PATH AIREADI_DATA/preprocessed_data0.9_seed8 --SEED 8 --ABLATION optuna2_trial26
python -m RL.train --OUT_DIR AIREADI_new/optuna2_trial26/seed9 --RESULT_CSV AIREADI_new/results.csv --RUN_NAME optuna2_trial26_seed9 --DATASET AIREADI --DATA_PATH AIREADI_DATA/preprocessed_data0.9_seed9 --SEED 9 --ABLATION optuna2_trial26
