#!/bin/bash
#SBATCH --job-name=cwc2
#SBATCH --partition=gpu
#SBATCH --qos=gpu-medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G
#SBATCH --time=11:59:59
#SBATCH --output=slurm-%A.out    # %A = jobID, %a = array index
#SBATCH --error=slurm-%A.err

python -m RL.train --OUT_DIR AIREADI_new/cwc_trial2/seed1 --RESULT_CSV AIREADI_new/results.csv --RUN_NAME cwc_trial2_seed1 --DATASET AIREADI --DATA_PATH AIREADI_DATA/preprocessed_data0.9_seed1 --SEED 1 --ABLATION cwc_trial2
python -m RL.train --OUT_DIR AIREADI_new/cwc_trial2/seed2 --RESULT_CSV AIREADI_new/results.csv --RUN_NAME cwc_trial2_seed2 --DATASET AIREADI --DATA_PATH AIREADI_DATA/preprocessed_data0.9_seed2 --SEED 2 --ABLATION cwc_trial2
python -m RL.train --OUT_DIR AIREADI_new/cwc_trial2/seed3 --RESULT_CSV AIREADI_new/results.csv --RUN_NAME cwc_trial2_seed3 --DATASET AIREADI --DATA_PATH AIREADI_DATA/preprocessed_data0.9_seed3 --SEED 3 --ABLATION cwc_trial2
python -m RL.train --OUT_DIR AIREADI_new/cwc_trial2/seed4 --RESULT_CSV AIREADI_new/results.csv --RUN_NAME cwc_trial2_seed4 --DATASET AIREADI --DATA_PATH AIREADI_DATA/preprocessed_data0.9_seed4 --SEED 4 --ABLATION cwc_trial2
python -m RL.train --OUT_DIR AIREADI_new/cwc_trial2/seed5 --RESULT_CSV AIREADI_new/results.csv --RUN_NAME cwc_trial2_seed5 --DATASET AIREADI --DATA_PATH AIREADI_DATA/preprocessed_data0.9_seed5 --SEED 5 --ABLATION cwc_trial2
python -m RL.train --OUT_DIR AIREADI_new/cwc_trial2/seed6 --RESULT_CSV AIREADI_new/results.csv --RUN_NAME cwc_trial2_seed6 --DATASET AIREADI --DATA_PATH AIREADI_DATA/preprocessed_data0.9_seed6 --SEED 6 --ABLATION cwc_trial2
python -m RL.train --OUT_DIR AIREADI_new/cwc_trial2/seed7 --RESULT_CSV AIREADI_new/results.csv --RUN_NAME cwc_trial2_seed7 --DATASET AIREADI --DATA_PATH AIREADI_DATA/preprocessed_data0.9_seed7 --SEED 7 --ABLATION cwc_trial2
python -m RL.train --OUT_DIR AIREADI_new/cwc_trial2/seed8 --RESULT_CSV AIREADI_new/results.csv --RUN_NAME cwc_trial2_seed8 --DATASET AIREADI --DATA_PATH AIREADI_DATA/preprocessed_data0.9_seed8 --SEED 8 --ABLATION cwc_trial2
python -m RL.train --OUT_DIR AIREADI_new/cwc_trial2/seed9 --RESULT_CSV AIREADI_new/results.csv --RUN_NAME cwc_trial2_seed9 --DATASET AIREADI --DATA_PATH AIREADI_DATA/preprocessed_data0.9_seed9 --SEED 9 --ABLATION cwc_trial2
