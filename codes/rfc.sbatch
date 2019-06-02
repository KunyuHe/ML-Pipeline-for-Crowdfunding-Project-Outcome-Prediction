#!/bin/bash
#SBATCH --job-name=rfc
#SBATCH --output=rfc.out
#SBATCH --error=rfc.err
#SBATCH --time=24:00:00
#SBATCH --partition=broadwl
#SBATCH --nodes=24
#SBATCH --mem-per-cpu=2000

module load Anaconda3/2018.12
conda activate mlpp
python rfc.py --clean_start 1 --verbose 0 --ask_user 0 --plot 1

