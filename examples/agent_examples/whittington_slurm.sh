#!/bin/bash
# Set the job name variable
#SBATCH --job-name=TEM_huge_50K
#SBATCH --mem=50000 # memory pool for all cores
#SBATCH --time=72:00:00 # time
#SBATCH -o TEM_logs/TEM_huge_50K.%N.%j.out # STDOUT
#SBATCH -e TEM_logs/TEM_huge_50K.%N.%j.err # STDERR
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1

source ~/.bashrc

conda activate NPG-env

python whittington_2020_run.py

exit