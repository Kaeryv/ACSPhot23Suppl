#!/bin/bash
#SBATCH --job-name=sel.
#SBATCH --output=logs/sel_%x.%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=0:10:00
#SBATCH --mem-per-cpu=4096

#{{workdir}}
source config
python user/perform_selection.py \
    -population {{dataset:npz.maps}} \
    -nclusters {{num_clusters}}\
    -nselected {{selection_size}} \
    -optimization-archives {{optimizer_archive}} \
    -model {{model}} \
    -output {{output:declare_file_output}}

touch {{touchfile}}
