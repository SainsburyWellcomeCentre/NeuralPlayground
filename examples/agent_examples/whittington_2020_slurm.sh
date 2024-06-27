#!/bin/bash
#SBATCH --job-name=test_TEM
#SBATCH --output=test_TEM.out
#SBATCH --error=test_TEM.err
#
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=8G 
#SBATCH --time=0-6:00
#
source ~/.bashrc
conda activate TEM
python whittington_2020_run.py