#!/bin/bash

#SBATCH -J TEM_50G # job name
#SBATCH -p gpu # partition (queue)
#SBATCH -N 1   # number of nodes
#SBATCH --mem 50G # memory pool for all cores
#SBATCH -n 4 # number of cores
#SBATCH -t 0-72:00 # time (D-HH:MM)
#SBATCH --gres gpu:1 # request 1 GPU (of any kind)
#SBATCH -o TEM_50G.%x.%N.%j.out # STDOUT
#SBATCH -e TEM_50G.%x.%N.%j.err # STDERR

source ~/.bashrc

conda activate NPG-env

python whittington_2020_run.py

exit
