#!/bin/bash

# Run experiments from the paper and reproduce the figures
set -e
green='\e[1;32m%s\e[0m\n'

# if -o or --original is specified, extract the original data
USE_ORIGINAL_DATA=0
for arg in "$@"; do
    case $arg in
        -o|--original)
            USE_ORIGINAL_DATA=1
            ;;
    esac
done

if [ $USE_ORIGINAL_DATA -eq 1 ]
then
    bash ./scripts/extract_data.sh
else
    printf "\n$green\n" 'NOTE: Running all experiments takes ~1 WEEK'
fi

# Run
bash ./scripts/run_paper_experiments.sh
bash ./scripts/create_paper_figs.sh
bash ./scripts/extract_paper_figs.sh


