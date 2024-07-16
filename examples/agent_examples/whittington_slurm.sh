#!/bin/bash
#SBATCH --job-name=TEM_update_test
#SBATCH --mem=20000
#SBATCH --time=72:00:00
#SBATCH -o TEM_logs/TEM_update_test.%N.%j.out
#SBATCH -e TEM_logs/TEM_update_test.%N.%j.err
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1

source ~/.bashrc

conda activate NPG-env

python whittington_2020_run.py

exit
