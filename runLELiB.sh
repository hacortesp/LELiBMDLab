#!/bin/bash 


python lelib_mdlab.py \
  -anion_name PF6 \
  -workdir test_EC-EMC \
  -solvents EC EMC \
  -solvent-fracs 0.5 0.5 \
  -temperature 298.0 \
  > output_LELiB_MDLab.txt
