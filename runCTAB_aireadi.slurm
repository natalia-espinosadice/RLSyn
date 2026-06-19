#!/bin/bash
#SBATCH --job-name=ai
#SBATCH --partition=gpu
#SBATCH --qos=gpu-medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=5G
#SBATCH --time=07:59:59
#SBATCH --output=slurm-%A.out    # %A = jobID, %a = array index
#SBATCH --error=slurm-%A.err


for SEED in 0 1 2 3 4 5 6 7 8 9 ; do
    python -m CTAB_GAN.main_aireadi --seed ${SEED}
done

