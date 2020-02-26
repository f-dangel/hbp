#!/bin/bash

green='\e[1;32m%s\e[0m\n'

printf "\n$green\n" 'Extracting original data'

unzip ./dat.zip -d ./dat
