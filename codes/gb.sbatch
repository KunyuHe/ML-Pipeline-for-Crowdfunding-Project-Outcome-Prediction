#!/bin/bash
#SBATCH --job-name=gb
#SBATCH --output=gb.out
#SBATCH --error=gb.err
#SBATCH --time=36:00:00
#SBATCH --partition=sandyb
#SBATCH --nodes=4
#SBATCH --mem-per-cpu=4000

module load Anaconda3/2018.12
source activate mlpp
python gb.py --start_clean 1 --verbose 0 --ask 0 --plot 1