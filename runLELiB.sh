#!/bin/bash 


python lelib_mdlab.py \
  -box 50.0 \
  -anion_name ClO4 \
  -salt-conc 1.0 \
  -workdir 1MLiClO4_DMC_T298K \
  -solvents DMC \
  -solvent-fracs 1.0 \
  -charge-scale 0.80 \
  -temperature 298 \
  > output_LELiB_MDLab.txt

