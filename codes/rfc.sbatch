#!/bin/bash
#SBATCH --job-name=rfc
#SBATCH --output=rfc.out
#SBATCH --error=rfc.err
#SBATCH --time=24:00:00
#SBATCH --partition=sandyb
#SBATCH --nodes=24
#SBATCH --mem-per-cpu=16000

module load Anaconda3/2018.12
source activate mlpp
python rfc.py --start_clean 0 --verbose 0 --ask 0 --plot 0
