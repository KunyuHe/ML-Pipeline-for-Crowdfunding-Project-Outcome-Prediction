#!/bin/bash
#SBATCH --job-name=ada
#SBATCH --output=ada.out
#SBATCH --error=ada.err
#SBATCH --time=12:00:00
#SBATCH --partition=broadwl
#SBATCH --nodes=8
#SBATCH --mem-per-cpu=2000

module load Anaconda3/2018.12
source activate mlpp
python ada.py --start_clean 1 --verbose 0 --ask 0 --plot 0
