#!/bin/bash

# Reproduce figures from the paper
set -e
green='\e[1;32m%s\e[0m\n'

printf "\n$green\n" 'CREATE PAPER FIGURES'

python exp/fig_exp02_chen2018_splitting_cifar10.py

python exp/fig_exp08_c4d3_optimization.py

python exp/fig_exp09_c3d3_optimization.py

printf "\n$green\n" 'CREATE PAPER FIGURES SUCCESSFUL'

printf "$green\n" 'Figures can be found in the fig/ directory'
