#!/bin/bash

#SBATCH -n 16
#SBATCH --mem-per-cpu=512
#SBATCH --time=8:00:00
#SBATCH --tmp=0                        # per node!!
#SBATCH --job-name=robomimic
#SBATCH -A es_hutter
#SBATCH --gpus=1

python workflow/train.py --dataset /cluster/scratch/chenyang/May29_00-07-25/hdf_dataset.hdf5 "$@"

