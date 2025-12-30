#!/bin/bash
    set -e

    gmx grompp -f em.mdp -c conf.pdb -p topol.top -o em.tpr
    gmx mdrun -v -deffnm em

    gmx grompp -f npt_eq.mdp -c em.gro -p topol.top -o npt_eq.tpr -maxwarn 1
    gmx mdrun -deffnm npt_eq

    gmx grompp -f npt_data.mdp -c npt_eq.gro -p topol.top -o npt_data.tpr -maxwarn 1
    gmx mdrun -deffnm npt_data
    
