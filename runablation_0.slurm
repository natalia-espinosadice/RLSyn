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


python -m RL.train --OUT_DIR ablation/AIREADI/no_clip_no_value_ppoepochs1/seed2 --RESULT_CSV ablation_AIREADI.csv --RUN_NAME no_clip_no_value_ppoepochs1_seed2 --DATASET AIREADI --DATA_PATH AIREADI_DATA/preprocessed_data0.9_seed2 --SEED 2 --ABLATION no_clip_no_value_ppoepochs1
python -m RL.train --OUT_DIR ablation/AIREADI/no_clip_no_value_ppoepochs1/seed3 --RESULT_CSV ablation_AIREADI.csv --RUN_NAME no_clip_no_value_ppoepochs1_seed3 --DATASET AIREADI --DATA_PATH AIREADI_DATA/preprocessed_data0.9_seed3 --SEED 3 --ABLATION no_clip_no_value_ppoepochs1
python -m RL.train --OUT_DIR ablation/AIREADI/no_clip_no_value_ppoepochs1/seed4 --RESULT_CSV ablation_AIREADI.csv --RUN_NAME no_clip_no_value_ppoepochs1_seed4 --DATASET AIREADI --DATA_PATH AIREADI_DATA/preprocessed_data0.9_seed4 --SEED 4 --ABLATION no_clip_no_value_ppoepochs1
python -m RL.train --OUT_DIR ablation/AIREADI/no_clip_no_value/seed2 --RESULT_CSV ablation_AIREADI.csv --RUN_NAME no_clip_no_value_seed2 --DATASET AIREADI --DATA_PATH AIREADI_DATA/preprocessed_data0.9_seed2 --SEED 2 --ABLATION no_clip_no_value
python -m RL.train --OUT_DIR ablation/AIREADI/no_clip_no_value/seed3 --RESULT_CSV ablation_AIREADI.csv --RUN_NAME no_clip_no_value_seed3 --DATASET AIREADI --DATA_PATH AIREADI_DATA/preprocessed_data0.9_seed3 --SEED 3 --ABLATION no_clip_no_value
python -m RL.train --OUT_DIR ablation/AIREADI/no_clip_no_value/seed4 --RESULT_CSV ablation_AIREADI.csv --RUN_NAME no_clip_no_value_seed4 --DATASET AIREADI --DATA_PATH AIREADI_DATA/preprocessed_data0.9_seed4 --SEED 4 --ABLATION no_clip_no_value

