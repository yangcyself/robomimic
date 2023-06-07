#!/bin/bash

#SBATCH -n 16
#SBATCH --mem-per-cpu=1024
#SBATCH --time=20:00:00  
#SBATCH --tmp=0                        # per node!!
#SBATCH --job-name=robomimic
#SBATCH -A es_hutter
#SBATCH --gpus=1

python workflow/train.py --logdir_prefix /cluster/scratch/chenyang "$@"

