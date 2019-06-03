#!/bin/bash
#SBATCH --job-name=et
#SBATCH --output=et.out
#SBATCH --error=et.err
#SBATCH --time=36:00:00
#SBATCH --partition=sandyb
#SBATCH --nodes=24
#SBATCH --mem-per-cpu=2000

module load Anaconda3/2018.12
source activate mlpp
python et.py --start_clean 0 --verbose 0 --ask 0 --plot 1