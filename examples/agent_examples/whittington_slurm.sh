#!/bin/bash
#SBATCH --job-name=TEM_cpu
#SBATCH --mem=20000
#SBATCH --time=72:00:00
#SBATCH -o TEM_logs/TEM_cpu.%N.%j.out
#SBATCH -e TEM_logs/TEM_cpu.%N.%j.err
#SBATCH -p cpu
#SBATCH --ntasks=1

source ~/.bashrc

conda activate NPG-env

python whittington_2020_run.py

exit
