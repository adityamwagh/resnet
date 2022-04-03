#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=2:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu
#SBATCH --job-name=M_funnel_activation
#SBATCH --output=%x.out

module purge

singularity exec --nv \
            --overlay /scratch/vg2097/pytorch-example/my_pytorch.ext3:ro \
            /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
            /bin/bash -c "source /ext3/env.sh; python main.py -en 20 -o sgd -dp -m Funnel ./data -b 2 2 2 2 -c 42 84 168 336"

