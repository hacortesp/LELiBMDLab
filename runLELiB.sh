#!/bin/bash 


python lelib_mdlab.py \
  -box 50.0 \
  -anion_name PF6 \
  -salt-conc 1.0 \
  -workdir sigma_segments \
  -solvents EC EMC \
  -solvent-fracs 0.5 0.5 \
  -charge-scale 0.85 \
  -temperature 300 \
  > output_LELiB_MDLab.txt
