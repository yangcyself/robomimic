#!/bin/bash

#SBATCH -n 16
#SBATCH --mem-per-cpu=512
#SBATCH --time=20:00:00  # IRIS takes too long
#SBATCH --tmp=0                        # per node!!
#SBATCH --job-name=robomimic
#SBATCH -A es_hutter
#SBATCH --gpus=1

python workflow/train.py --logdir_prefix /cluster/home/chenyang "$@"

