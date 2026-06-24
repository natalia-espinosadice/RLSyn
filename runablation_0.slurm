#!/bin/bash
#SBATCH --job-name=ainew6OG
#SBATCH --partition=gpu
#SBATCH --qos=gpu-medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=25G
#SBATCH --time=03:59:59
#SBATCH --output=slurm-%A.out    # %A = jobID, %a = array index
#SBATCH --error=slurm-%A.err

python -m RL.train --OUT_DIR AIREADI_new/trial6_OG/seed6 --RESULT_CSV AIREADI_new/results.csv --RUN_NAME trial6OG_seed6 --DATASET AIREADI --DATA_PATH AIREADI_DATA/preprocessed_data0.9_seed6 --SEED 6 --ABLATION trial6_ogoptuna
python -m RL.train --OUT_DIR AIREADI_new/trial6_OG/seed7 --RESULT_CSV AIREADI_new/results.csv --RUN_NAME trial6OG_seed7 --DATASET AIREADI --DATA_PATH AIREADI_DATA/preprocessed_data0.9_seed7 --SEED 7 --ABLATION trial6_ogoptuna
python -m RL.train --OUT_DIR AIREADI_new/trial6_OG/seed8 --RESULT_CSV AIREADI_new/results.csv --RUN_NAME trial6OG_seed8 --DATASET AIREADI --DATA_PATH AIREADI_DATA/preprocessed_data0.9_seed8 --SEED 8 --ABLATION trial6_ogoptuna
python -m RL.train --OUT_DIR AIREADI_new/trial6_OG/seed9 --RESULT_CSV AIREADI_new/results.csv --RUN_NAME trial6OG_seed9 --DATASET AIREADI --DATA_PATH AIREADI_DATA/preprocessed_data0.9_seed9 --SEED 9 --ABLATION trial6_ogoptuna
