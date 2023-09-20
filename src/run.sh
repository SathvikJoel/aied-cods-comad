#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --ntasks-per-node=6
#SBATCH --mem=100000M
#SBATCH --time=11:00:00
#SBATCH --account=def-fard
#SBATCH --mail-user=ksjoe30@gmail.com

source /home/ksjoe30/projects/def-fard/ksjoe30/ubc/bin/activate


python trinaing_SoftmaxLoss.py