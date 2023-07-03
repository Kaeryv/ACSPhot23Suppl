#!/bin/bash
#SBATCH --job-name=unet
#SBATCH --output=logs/%x.%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time={{hours}}:15:00
#SBATCH --mem-per-cpu=1024

source config
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python user/train_unet.py \
    -nthreads $SLURM_CPUS_PER_TASK \
    -epochs {{epochs}} \
    -name {{model_file}} \
    -device cpu -hours {{hours}} \
    -complexity {{complexity}}\
    -augment_angle {{angle}} \
    -wd {{decay}}  \
    -bs {{batch_size}} \
    -lr {{lr}} \
    -data {{dataset:npz.maps}}\
    -validratio {{validratio}}\
    -copy_contour
touch {{touchfile}}
