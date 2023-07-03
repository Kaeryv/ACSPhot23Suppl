#!/bin/bash
#SBATCH --job-name=sim
#SBATCH --output=logs/%x.%j.log
#SBATCH --ntasks=16
#SBATCH --time=00:30:00

source config
#touch {{workdir}}/{{individual_tag[]}}.out.npz
mpirun -n 16 python user/evaluate_simulation.py \
    -npz {{workdir}}/{{individual_tag[]}}.in.npz \
    -type {{type}}\
    -res {{resolution}}

touch {{touchfile}}
