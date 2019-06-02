#!/bin/bash
#SBATCH --job-name=bag
#SBATCH --output=bag.out
#SBATCH --error=bag.err
#SBATCH --time=12:00:00
#SBATCH --partition=sandyb
#SBATCH --nodes=8
#SBATCH --mem-per-cpu=2000

module load Anaconda3/2018.12
source activate mlpp
python bag.py --start_clean 1 --verbose 0 --ask 0 --plot 1
