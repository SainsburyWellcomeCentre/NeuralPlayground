#!/bin/bash
#SBATCH --job-name=TEM_cpu
#SBATCH -o TEM_logs/TEM_cpu.%N.%j.out
#SBATCH -e TEM_logs/TEM_cpu.%N.%j.err
#SBATCH -p cpu
#SBATCH --nodes 1
#SBATCH --cpus-per-task 4
#SBATCH --mem 16
#SBATCH --time=72:00

source ~/.bashrc

conda activate NPG-env

python whittington_2020_run.py

exit
