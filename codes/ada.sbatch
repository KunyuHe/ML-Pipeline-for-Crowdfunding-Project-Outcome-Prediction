#!/bin/bash
#SBATCH --job-name=ada
#SBATCH --output=ada.out
#SBATCH --error=ada.err
#SBATCH --time=12:00:00
#SBATCH --partition=sandyb
#SBATCH --nodes=8
#SBATCH --mem-per-cpu=2000

module load Anaconda3/2018.12
conda activate mlpp
python ada.py --clean_start 1 --verbose 0 --ask_user 0 --plot 1
