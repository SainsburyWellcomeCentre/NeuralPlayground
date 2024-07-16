#!/bin/bash
#SBATCH -p cpu
#SBATCH --nodes 1
#SBATCH --cpus-per-task 4
#SBATCH --mem 16
#SBATCH --time=1:00

source ~/.bashrc

conda activate NPG-env

python whittington_2020_run.py

exit
