#!/bin/bash
#SBATCH -p cpu
#SBATCH --nodes 1
#SBATCH --cpus-per-task 4
#SBATCH --mem 16
#SBATCH --time=72:00:00
#SBATCH --job-name=TEM_update
#SBATCH -o TEM_logs/TEM_update.%N.%j.out
#SBATCH -e TEM_logs/TEM_update.%N.%j.err

source ~/.bashrc

conda activate NPG-env

python whittington_2020_run.py

exit
