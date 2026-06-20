#!/bin/bash
#SBATCH --job-name=mimic3             # Job name
#SBATCH --nodes=1                       # We only need 1 node
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1              # 8 CPU cores
#SBATCH --gres=gpu:1                    # 1 GPU
#SBATCH --mem=20G                       # 16 GB of system RAM
#SBATCH --time=23:59:59              # Max run time (HH:MM:SS)
#SBATCH --output=slurm-%j.out           # Standard output log (%j expands to jobID)
#SBATCH --error=slurm-%j.err            # Standard error log



python -m CTAB_GAN.main_mimic --seed 3