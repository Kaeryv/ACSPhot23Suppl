#!/bin/bash
#SBATCH --job-name=opt
#SBATCH --output=logs/%x.%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0:40:00
#SBATCH --mem-per-cpu=1024

#{{workdir}}

source config
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python user/optimize_unet_pso.py \
    -fom {{fom:serialize}}\
    -model {{model}} \
    -seed {{seed[]}}\
    -nagents {{nagents}} \
    -fevals {{fevals}} \
    -max-stagnation 10\
    -output {{output:declare_file_output}}\
    -nthreads $SLURM_CPUS_PER_TASK

touch {{touchfile}}
