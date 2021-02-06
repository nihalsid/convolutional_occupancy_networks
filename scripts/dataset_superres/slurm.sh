#!/bin/bash

#SBATCH --job-name matterport_data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=0
##SBATCH --mem-per-cpu=12G
#SBATCH --exclude=moria,umoja

cd /rhome/ysiddiqui/convolutional_occupancy_networks
python process_dataset.py --outputdir /cluster/gondor/ysiddiqui/conv_occ_superres --num_proc $SLURM_ARRAY_TASK_COUNT --proc $SLURM_ARRAY_TASK_ID