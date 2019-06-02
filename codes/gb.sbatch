#!/bin/bash
#SBATCH --job-name=gb
#SBATCH --output=gb.out
#SBATCH --error=gb.err
#SBATCH --time=36:00:00
#SBATCH --partition=sandyb
#SBATCH --nodes=4
#SBATCH --mem-per-cpu=4000

module load Anaconda3/2018.12
conda activate mlpp
python gb.py --clean_start 1 --verbose 0 --ask_user 0 --plot 1
