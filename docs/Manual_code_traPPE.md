# TraPPE Electrolyte Builder (Packmol → GROMACS)
**User Manual for `code_traPPE.py`**

This script builds liquid electrolyte systems composed of **LiPF6 and one or more organic solvents**
parameterized with **TraPPE** (https://github.com/SB8/trappe-electrolyte/tree/main), using **Packmol** for structure generation and **GROMACS** for molecular dynamics.

The workflow is designed to be **reproducible, modular, and repository-ready**, suitable for both local
execution and HPC clusters.

---

## 1. Purpose of the code

`code_traPPE.py` automates the following tasks:

1. Compute the number of salt pairs (LiPF6) and solvent molecules from:
   - Simulation box size
   - Salt concentration (mol/L)
   - Solvent volume fractions

2. Generate a Packmol input and build an initial configuration.

3. Create a complete GROMACS simulation folder containing:
   - `conf.pdb` with correct CRYST1 box definition
   - Force field and topology files
   - MDP input files
   - Execution scripts for local and SLURM-based runs

---

## 2. Expected repository structure

The script must be placed in the root directory together with the following folders (it is necessary to run the main .py inside the "TraPPE_codepy" folder):

```
TraPPE_codepy/
├── code_traPPE.py       
├── forcefield/
│   ├── ffbonded_TraPPE.itp
│   ├── ffnonbonded_TraPPE.itp
│   └── forcefield.itp
├── itp/
│   ├── EC_TraPPE.itp
│   ├── EMC_TraPPE.itp
│   ├── PC_TraPPE.itp
│   ├── DMC_TraPPE.itp
│   ├── DEC_TraPPE.itp
│   ├── DME_TraPPE.itp
│   ├── Li.itp
│   └── PF6.itp
├── mdp/
│   ├── minim.mdp
│   ├── npt_new_eq.mdp
│   └── npt_new_data.mdp
└── pdb/
    ├── EC_UA.pdb
    ├── EMC_UA.pdb
    ├── PC_UA.pdb
    ├── DMC_UA.pdb
    ├── DEC_UA.pdb
    ├── DME_UA.pdb
    ├── Li.pdb
    └── PF6.pdb
```

---

## 3. Dependencies

The following software must be available:

- Python ≥ 3.8
- Packmol (tested with `/usr/bin/packmol`)
- GROMACS (local or MPI version)

Required Python modules:
- argparse
- pathlib
- shutil
- subprocess

---

## 4. Command-line interface

### Basic usage

```bash
python code_traPPE.py \
  --box 40 40 40 \
  --salt-conc 1.0 \
  --solvents EC EMC \
  --solvent-fracs 0.5 0.5
```

### Full argument list

| Flag              |    Req? | Type        | Default        | Description                                                                                                                                                                                                    |
| ----------------- | ------: | ----------- | -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--box Lx Ly Lz`  | **yes** | 3×float     | —              | Simulation box dimensions in **Å**. Defines the total system volume used for molecule count estimation and Packmol placement. Example: `--box 40 40 40`.                                                       |
| `--salt`          |      no | str         | `LiPF6`        | Salt type. Currently only `LiPF6` is implemented, treated as ion pairs (Li⁺ + PF₆⁻).                                                                                                                           |
| `--salt-conc`     | **yes** | float       | —              | Salt concentration in **mol/L**. Used together with the box volume to compute the number of LiPF₆ ion pairs. Example: `--salt-conc 1.0`.                                                                       |
| `--solvents`      | **yes** | list(str)   | —              | List of solvent identifiers. **Available options:** `EC`, `PC`, `DMC`, `DEC`, `DME`, `EMC`. Each solvent must have a corresponding PDB file in `pdb/` and an ITP file in `itp/`. Example: `--solvents EC EMC`. |
| `--solvent-fracs` | **yes** | list(float) | —              | Relative **volume fractions** of each solvent specified in `--solvents`. Values are automatically **renormalized** if they do not sum to 1. Example: `--solvent-fracs 0.5 0.5`.                                |
| `--charge-scale`  |      no | float       | `1.0`          | Global charge scaling factor applied to **all atomic charges** in `Li.itp` and `PF6.itp`. For example, `0.85` corresponds to 85% charge scaling, commonly used in effective polarization approaches.           |
| `--margin`        |      no | float       | `0.5`          | Margin (in Å) applied in Packmol `inside box` instructions to avoid placing atoms directly on the box boundaries.                                                                                              |
| `--sim-name`      |      no | str         | auto-generated | Custom name for the simulation folder. If not provided, a descriptive name is generated automatically based on salt type, concentration, solvent list, box dimensions, and charge scaling factor.              |

Solvent fractions are automatically renormalized.

---

## 5. Molecule count calculation

The number of salt pairs is computed as:

```
N_salt = c · V_box · N_A
```

The remaining volume is distributed among solvents according to their
relative fractions and molecular volumes.

---

## 6. Charge scaling

The option:

```bash
--charge-scale 0.85
```

rescales all atomic charges in `Li.itp` and `PF6.itp` before topology generation.

---

## 7. Output structure

The script creates a folder such as:

```
sim_LiPF6_1M_EC_EMC_40x40x40A_q0p85/
├── conf.pdb
├── topol.top
├── minim.mdp
├── npt_new_eq.mdp
├── npt_new_data.mdp
├── run_local.sh
├── run_cluster.sh
├── forcefield/
├── itp/
└── packmol/
```

---

## 8. Running the simulations

### Local execution

```bash
cd sim_*
./run_local.sh
```

### SLURM cluster

```bash
sbatch run_cluster.sh
```

---

## 9. Example

```bash
python code_traPPE.py \
  --box 50 50 50 \
  --salt-conc 1.2 \
  --solvents EC EMC \
  --solvent-fracs 0.3 0.7 \
  --charge-scale 0.85
```

---

## 10. Notes

- Only LiPF6 is implemented by default.
- All solvents must be defined consistently (PDB + ITP).
- Packmol failures usually indicate excessive density.

---

## 11) Authors & acknowledgment
- **Oscar David Orozco González** – Basque Center for Applied Mathematics (BCAM), Spain  
- **Dr. Henry Andrés Cortés** – Basque Center for Applied Mathematics (BCAM), Spain  
- **Dr. Mauricio Rincón Bonilla** – Basque Center for Applied Mathematics (BCAM), Spain  
- **Prof. Elena Akhmatskaya** – Basque Center for Applied Mathematics (BCAM), Spain

