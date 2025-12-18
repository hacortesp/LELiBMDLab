## Li-EMDLab

Li-ion Electrolyte Molecular Dynamics Lab (Li-EMDLab)
`code_traPPE.py` is a Python-based command-line workflow designed to construct molecular models of electrolytes composed of LiPF₆ salt and one or more organic carbonate solvents parameterized with the TraPPE force field.

Given a target box size, salt concentration, and solvent composition, the code computes the number of ion pairs and solvent molecules, generates an initial configuration using Packmol, optionally rescales ionic charges, and prepares a fully self-contained GROMACS simulation directory. The generated output includes topology files, force-field definitions, molecular dynamics parameter files, and execution scripts for both local and SLURM-based cluster environments.
