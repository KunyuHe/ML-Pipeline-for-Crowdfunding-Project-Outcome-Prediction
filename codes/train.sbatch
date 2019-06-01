#!/bin/bash
#SBATCH --job-name=gbtrain
#SBATCH --output=gbtrain.out
#SBATCH --error=gbtrain.err
#SBATCH --time=72:00:00
#SBATCH --partition=sandyb
#SBATCH --nodes=16
#SBATCH --mem-per-cpu=2000

module load Anaconda/2018.12
conda activate mlpp
python train.py --clean_start 0 --verbose 1 --ask_user 0 --plot 0
