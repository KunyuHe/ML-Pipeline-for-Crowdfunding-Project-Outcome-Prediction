#!/bin/bash
#SBATCH --job-name=example_sbatch
#SBATCH --time=00:05:00
#SBATCH --partition=broadwl
#SBATCH --nodes=16
#SBATCH --mem-per-cpu=2000

module load Anaconda/2018.12
conda activate mlpp
python train.py --clean_start 0 --verbose 1 --ask_user 0 --plot 0
