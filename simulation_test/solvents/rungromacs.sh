#!/bin/bash 
#SBATCH --account=bcam-exclusive
#SBATCH --partition=bcam-exclusive
#SBATCH --job-name=solvent  
#SBATCH --ntasks-per-node=48
#SBATCH --mem=20gb
#SBATCH --cpus-per-task=1
#SBATCH --time=10-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
##SBATCH --partition=test #regular

module load GROMACS/2019.4-foss-2019b

mpirun -np 1 gmx_mpi grompp -f em.mdp -c conf.pdb -p topol.top -o em.tpr
mpirun -np 48 gmx_mpi mdrun -v -deffnm em

            
mpirun -np 1 gmx_mpi grompp -f npt_eq.mdp -c em.gro -p topol.top -o npt_eq.tpr -maxwarn 1 
mpirun -np 48 gmx_mpi mdrun -deffnm npt_eq


mpirun -np 1 gmx_mpi grompp -f nvt_prod.mdp -c npt_eq.gro -p topol.top -o nvt_prod_wrap.tpr -maxwarn 1
mpirun -np 48 gmx_mpi mdrun -deffnm nvt_prod_wrap

echo 0 |gmx trjconv -s nvt_prod.tpr -f nvt_prod_wrap.xtc -o nvt_prod_unwrap.xt
 


