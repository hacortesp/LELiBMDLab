#!/bin/bash 


python lelib_mdlab.py \
  -box 50.0 \
  -anion_name TFSI \
  -salt-conc 0.5 \
  -workdir 0p5MLiTFSI_DME_298K \
  -solvents DME \
  -solvent-fracs 1.0 \
  -charge-scale 0.80 \
  -temperature 298 \
  > output_LELiB_MDLab.txt

