#!/bin/bash

#SBATCH -J TEM_pray # job name
#SBATCH -p gpu # partition (queue)
#SBATCH -N 1   # number of nodes
#SBATCH --mem 16G # memory pool for all cores
#SBATCH -n 4 # number of cores
#SBATCH -t 0-72:00 # time (D-HH:MM)
#SBATCH --gres gpu:1 # request 1 GPU (of any kind)
#SBATCH -o TEM_pray.%x.%N.%j.out # STDOUT
#SBATCH -e TEM_pray.%x.%N.%j.err # STDERR

source ~/.bashrc

conda activate NPG-env

python whittington_2020_run.py

exit
