#!/bin/bash

set -e
green='\e[1;32m%s\e[0m\n'

printf "\n$green\n" 'RUN PAPER EXPERIMENTS'

# Non-block splitting experiment (SGD curve)
printf "\n$green\n" 'Progress: 1/9'
python exp/exp01_chen2018_fig2_cifar10.py

# Block-splitting experiment
printf "\n$green\n" 'Progress: 2/9'
python exp/exp02_chen2018_splitting_cifar10.py

# CNN experiment
printf "\n$green\n" 'Progress: 3/9'
python exp/exp08_c4d3_optimization_sgd.py

printf "\n$green\n" 'Progress: 4/9'
python exp/exp08_c4d3_optimization_adam.py

printf "\n$green\n" 'Progress: 5/9'
python exp/exp08_c4d3_optimization_cvp.py

printf "\n$green\n" 'Progress: 6/9'
python exp/exp08_c4d3_optimization_kfac.py

# DeepOBS experiment
printf "\n$green\n" 'Progress: 7/9'
python exp/exp09_cifar10_deepobs_3c3d_sgd.py

printf "\n$green\n" 'Progress: 8/9'
python exp/exp09_cifar10_deepobs_3c3d_adam.py

printf "\n$green\n" 'Progress: 9/9'
python exp/exp09_cifar10_deepobs_3c3d_cvp.py

printf "\n$green\n" 'RUN EXPERIMENTS SUCCESSFUL'
