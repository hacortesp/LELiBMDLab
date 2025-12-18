#!/usr/bin/env python3
"""
System builder for LiPF6 + TraPPE-based solvents:
- Runs Packmol to generate the initial configuration.
- Prepares a full GROMACS simulation folder.

Overall workflow:

1. From:
   - Box dimensions (--box Lx Ly Lz, in Å)
   - Salt concentration (--salt-conc, in mol/L)
   - List of solvents (--solvents)
   - Relative volume fractions (--solvent-fracs)
   - Charge scaling factor (--charge-scale)
   it computes:
   - Number of LiPF6 ion pairs
   - Number of molecules of each solvent

2. It creates a simulation folder:
   sim_<salt>_<conc>M_<solvs>_<Lx>x<Ly>x<Lz>A_qXYY/

   Inside:
     - packmol/ (inputs + output conf.pdb with CRYST1)
     - forcefield/
     - itp/  (with Li.itp and PF6.itp rescaled if requested)
     - minim.mdp, npt_new_eq.mdp, npt_new_data.mdp
     - topol.top
     - run_local.sh
     - run_cluster.sh
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict

# -------------------- Physical constants --------------------

NA = 6.022141e23  # mol^-1

# M in g/mol, rho in g/cm^3, Vmol in Å^3 per molecule
SOLVENT_DATA: Dict[str, Dict[str, float]] = {
    "EC":   {"M": 88.0632,  "rho": 1.323, "Vmol": 110.530902479474},
    "PC":   {"M": 102.0902, "rho": 1.2,   "Vmol": 141.270637896326},
    "DMC":  {"M": 90.0792,  "rho": 1.063, "Vmol": 140.714986584917},
    "DEC":  {"M": 118.1332, "rho": 0.98,  "Vmol": 200.168156867614},
    "DME":  {"M": 90.1228,  "rho": 0.86,  "Vmol": 174.014453771041},
    "EMC":  {"M": 104.1062, "rho": 1.006, "Vmol": 171.841364050710},
}

# Map solvent -> PDB file (in pdb/ folder)
SOLVENT_PDB_MAP: Dict[str, str] = {
    "EC":  "EC_UA.pdb",
    "PC":  "PC_UA.pdb",
    "DMC": "DMC_UA.pdb",   # if you prefer DMC_cis-trans_UA.pdb, change it here
    "DEC": "DEC_UA.pdb",
    "DME": "DME_UA.pdb",
    "EMC": "EMC_UA.pdb",
}

# Map solvent -> TraPPE .itp file
SOLVENT_ITP_MAP: Dict[str, str] = {
    "EC":  "EC_TraPPE.itp",
    "PC":  "PC_TraPPE.itp",
    "DMC": "DMC_TraPPE.itp",
    "DEC": "DEC_TraPPE.itp",
    "DME": "DME_TraPPE.itp",
    "EMC": "EMC_TraPPE.itp",
}

# Salt treated as an "ion pair" (LiPF6)
SALT_DATA: Dict[str, Dict[str, float | str]] = {
    "LiPF6": {
        "M": 151.90518,            # g/mol
        "rho": 1.5,                # g/cm^3
        "Vmol": 168.162990597383,  # Å^3 per LiPF6 pair
        "cation_pdb": "Li.pdb",
        "anion_pdb": "PF6.pdb",
    }
}

# Map salt -> .itp and moleculetype names
SALT_ITP_MAP: Dict[str, Dict[str, str]] = {
    "LiPF6": {
        "cation_itp": "Li.itp",    # for other charges you may use Li_85pc_q.itp, etc.
        "anion_itp":  "PF6.itp",   # likewise PF6_85pc_q.itp if needed
        "cation_mt":  "Li",
        "anion_mt":   "PF6",
    }
}

# Path to Packmol executable
PACKMOL_EXE = "/usr/bin/packmol"


# ============================================================
# 1. SALT + SOLVENT COUNTS
# ============================================================

def compute_salt_and_solvent_counts(
    box: List[float],
    salt_name: str,
    salt_conc_M: float,
    solvent_names: List[str],
    solvent_fracs: List[float],
) -> dict:
    """
    Compute:
      - number of LiPF6 ion pairs
      - number of molecules of each solvent
    for a given box size and salt molarity.
    """
    if salt_name not in SALT_DATA:
        raise ValueError(f"Salt '{salt_name}' not defined in SALT_DATA.")
    if len(solvent_names) != len(solvent_fracs):
        raise ValueError("solvent_names and solvent_fracs must have the same length.")

    frac_sum = sum(solvent_fracs)
    if frac_sum <= 0.0:
        raise ValueError("The sum of solvent fractions must be positive.")

    # Normalize volume fractions
    norm_fracs = [f / frac_sum for f in solvent_fracs]

    # Box volume
    Lx, Ly, Lz = box
    V_box_A3 = Lx * Ly * Lz
    V_box_L = V_box_A3 * 1e-27  # 1 Å^3 = 1e-27 L

    # Moles and number of salt pairs
    salt = SALT_DATA[salt_name]
    n_salt_mol = salt_conc_M * V_box_L
    N_salt_pairs_float = n_salt_mol * NA
    N_salt_pairs = max(1, int(round(N_salt_pairs_float)))

    # Volume occupied by salt
    Vmol_salt = float(salt["Vmol"])
    V_salt_A3 = N_salt_pairs_float * Vmol_salt

    # Volume available for solvents
    V_solvent_total_A3 = V_box_A3 - V_salt_A3
    if V_solvent_total_A3 <= 0:
        raise ValueError(
            "Salt volume exceeds or equals box volume. "
            "Use a lower concentration or a larger box."
        )

    solvent_counts: Dict[str, Dict[str, float]] = {}
    for name, frac in zip(solvent_names, norm_fracs):
        if name not in SOLVENT_DATA:
            raise ValueError(f"Solvent '{name}' not defined in SOLVENT_DATA.")

        data = SOLVENT_DATA[name]
        V_i_A3 = frac * V_solvent_total_A3
        Vmol_i = data["Vmol"]
        N_i_float = V_i_A3 / Vmol_i
        N_i = max(1, int(round(N_i_float)))

        solvent_counts[name] = {
            "N_float": N_i_float,
            "N": N_i,
            "V_i_A3": V_i_A3,
        }

    return {
        "box": {
            "Lx": Lx,
            "Ly": Ly,
            "Lz": Lz,
            "V_box_A3": V_box_A3,
            "V_box_L": V_box_L,
        },
        "salt": {
            "name": salt_name,
            "N_pairs_float": N_salt_pairs_float,
            "N_pairs": N_salt_pairs,
            "V_salt_A3": V_salt_A3,
        },
        "solvents": solvent_counts,
    }


# ============================================================
# 2. PACKMOL: INPUT AND EXECUTION
# ============================================================

def generate_packmol_input(
    box: List[float],
    salt_name: str,
    counts: dict,
    margin: float = 0.5,
    output_name: str = "conf.pdb",
) -> str:
    """Generate the Packmol input file (conf_gen.inp)."""
    Lx, Ly, Lz = box
    x_min, y_min, z_min = margin, margin, margin
    x_max, y_max, z_max = Lx - margin, Ly - margin, Lz - margin

    lines: List[str] = []
    lines.append("# Auto-generated Packmol input")
    lines.append("tolerance 2.0")
    lines.append("filetype pdb")
    lines.append(f"output {output_name}")
    lines.append("")
    lines.append("seed 42")
    lines.append("")

    # Salt (LiPF6) = Li + PF6
    salt_info = SALT_DATA[salt_name]
    N_pairs = counts["salt"]["N_pairs"]
    cation_pdb = salt_info["cation_pdb"]
    anion_pdb = salt_info["anion_pdb"]

    lines.append("# >>>>> Electrolyte cation")
    lines.append(f"structure {cation_pdb}")
    lines.append(f"   number {N_pairs}")
    lines.append("   resnumbers 3")
    lines.append(
        f"   inside box {x_min:.3f} {y_min:.3f} {z_min:.3f} "
        f"{x_max:.3f} {y_max:.3f} {z_max:.3f}"
    )
    lines.append("   nloop 500")
    lines.append("end structure")
    lines.append("")

    lines.append("# >>>>> Electrolyte anion")
    lines.append(f"structure {anion_pdb}")
    lines.append(f"   number {N_pairs}")
    lines.append("   resnumbers 3")
    lines.append(
        f"   inside box {x_min:.3f} {y_min:.3f} {z_min:.3f} "
        f"{x_max:.3f} {y_max:.3f} {z_max:.3f}"
    )
    lines.append("   nloop 500")
    lines.append("end structure")
    lines.append("")

    # Solvents
    for name, info in counts["solvents"].items():
        N = info["N"]
        pdb_name = SOLVENT_PDB_MAP[name]
        lines.append(f"# >>>>> Solvent: {name}")
        lines.append(f"structure {pdb_name}")
        lines.append(f"   number {N}")
        lines.append("   resnumbers 3")
        lines.append(
            f"   inside box {x_min:.3f} {y_min:.3f} {z_min:.3f} "
            f"{x_max:.3f} {y_max:.3f} {z_max:.3f}"
        )
        lines.append("   nloop 500")
        lines.append("end structure")
        lines.append("")

    return "\n".join(lines) + "\n"


def run_packmol(packmol_dir: Path, inp_filename: str = "conf_gen.inp") -> None:
    """Run Packmol inside packmol_dir using the given input file."""
    inp_path = packmol_dir / inp_filename
    if not inp_path.exists():
        raise FileNotFoundError(f"{inp_path} not found to run Packmol.")

    log_path = packmol_dir / "packmol.log"

    with inp_path.open("r") as f_in, log_path.open("w") as f_log:
        result = subprocess.run(
            [PACKMOL_EXE],
            stdin=f_in,
            stdout=f_log,
            stderr=subprocess.STDOUT,
            cwd=str(packmol_dir),
        )

    if result.returncode != 0:
        raise RuntimeError(
            f"Packmol finished with non-zero return code {result.returncode}. "
            f"Check the log file at {log_path}"
        )


def add_cryst1_to_pdb(packmol_dir: Path, box: List[float], pdb_name: str = "conf.pdb") -> None:
    """
    Rewrite the Packmol-generated PDB to:
      - remove the original header
      - add a CRYST1 line with the box dimensions
    """
    pdb_path = packmol_dir / pdb_name
    if not pdb_path.exists():
        raise FileNotFoundError(f"{pdb_path} not found to add CRYST1.")

    lines = pdb_path.read_text().splitlines()

    # Find the first ATOM/HETATM line
    start_idx = None
    for i, line in enumerate(lines):
        if line.startswith(("ATOM", "HETATM")):
            start_idx = i
            break

    if start_idx is None:
        raise ValueError("Packmol PDB does not contain ATOM/HETATM lines.")

    atom_block = lines[start_idx:]

    Lx, Ly, Lz = box
    cryst1_line = f"CRYST1{Lx:9.3f}{Ly:9.3f}{Lz:9.3f}  90.00  90.00  90.00 P 1           1"

    new_lines = [cryst1_line] + atom_block
    pdb_path.write_text("\n".join(new_lines) + "\n")


# ============================================================
# 3. CHARGE RESCALING IN Li.itp AND PF6.itp
# ============================================================

def scale_charges_in_itp(itp_path: Path, factor: float, has_mass: bool) -> None:
    """
    Rescale charges in the [ atoms ] section of a .itp file.

    Parameters
    ----------
    itp_path : Path
        Path to the .itp file.
    factor : float
        Scaling factor to apply to the charges.
    has_mass : bool
        - True  -> Li.itp-like format where the charge is the
                   penultimate numerical column (before mass).
        - False -> PF6.itp-like format where the charge is the
                   last numerical column.
    """
    if not itp_path.exists():
        raise FileNotFoundError(f"{itp_path} not found for charge rescaling.")

    if abs(factor - 1.0) < 1e-8:
        # Nothing to do
        return

    lines = itp_path.read_text().splitlines()
    new_lines: List[str] = []
    in_atoms = False

    for line in lines:
        stripped = line.strip()

        # Section change
        if stripped.startswith("["):
            in_atoms = stripped.lower().startswith("[ atoms")
            new_lines.append(line)
            continue

        # Comments or empty lines
        if stripped.startswith(";") or not stripped:
            new_lines.append(line)
            continue

        if in_atoms:
            tokens = stripped.split()
            try:
                if has_mass:
                    # Li.itp format: ... charge mass
                    if len(tokens) >= 7:
                        charge = float(tokens[-2])
                        new_charge = charge * factor
                        tokens[-2] = f"{new_charge:.4f}"
                else:
                    # PF6.itp format: ... charge
                    if len(tokens) >= 7:
                        charge = float(tokens[-1])
                        new_charge = charge * factor
                        tokens[-1] = f"{new_charge:.4f}"

                # Rebuild with reasonable spacing
                new_line = "  " + "  ".join(tokens)
                new_lines.append(new_line)
            except ValueError:
                # If parsing fails for any reason, keep the original line
                new_lines.append(line)
        else:
            new_lines.append(line)

    itp_path.write_text("\n".join(new_lines) + "\n")


def apply_charge_scaling(itp_dir: Path, factor: float) -> None:
    """Apply charge scaling factor to Li.itp and PF6.itp inside itp_dir."""
    if abs(factor - 1.0) < 1e-8:
        return

    li_itp = itp_dir / "Li.itp"
    pf6_itp = itp_dir / "PF6.itp"

    scale_charges_in_itp(li_itp, factor, has_mass=True)
    scale_charges_in_itp(pf6_itp, factor, has_mass=False)


# ============================================================
# 4. TOPOLOGY AND GROMACS SCRIPTS
# ============================================================

def auto_sim_name(
    salt_name: str,
    salt_conc_M: float,
    solvent_names: List[str],
    box: List[float],
    charge_scale: float,
) -> str:
    """Generate a default simulation folder name."""
    Lx, Ly, Lz = box
    conc_str = f"{salt_conc_M:g}M"
    solv_str = "_".join(solvent_names)
    box_str = f"{Lx:.0f}x{Ly:.0f}x{Lz:.0f}A"
    q_str = f"q{charge_scale:.2f}".replace(".", "p")
    return f"sim_{salt_name}_{conc_str}_{solv_str}_{box_str}_{q_str}"


def generate_topology(
    sim_dir: Path,
    counts: dict,
    solvent_names: List[str],
    salt_name: str,
) -> None:
    """
    Generate topol.top in sim_dir:
      - includes for forcefield and .itp files
      - [ system ] section
      - [ molecules ] section with Li, PF6 and solvents
    """
    top_path = sim_dir / "topol.top"

    lines: List[str] = []
    lines.append("; Automatically generated topology for TraPPE + LiPF6")
    lines.append('#include "forcefield/forcefield.itp"')
    lines.append("")

    salt_itp = SALT_ITP_MAP[salt_name]
    lines.append(f'#include "itp/{salt_itp["cation_itp"]}"')
    lines.append(f'#include "itp/{salt_itp["anion_itp"]}"')

    for name in solvent_names:
        itp_name = SOLVENT_ITP_MAP[name]
        lines.append(f'#include "itp/{itp_name}"')

    lines.append("")
    lines.append("[ system ]")
    Lx = counts["box"]["Lx"]
    Ly = counts["box"]["Ly"]
    Lz = counts["box"]["Lz"]
    system_name = (
        f'LiPF6 in {" + ".join(solvent_names)} ; '
        f'{salt_name}, box = {Lx:.1f} x {Ly:.1f} x {Lz:.1f} Å'
    )
    lines.append(system_name)
    lines.append("")

    lines.append("[ molecules ]")

    N_pairs = counts["salt"]["N_pairs"]
    cation_mt = salt_itp["cation_mt"]
    anion_mt = salt_itp["anion_mt"]
    lines.append(f"{cation_mt:8s} {N_pairs:d}")
    lines.append(f"{anion_mt:8s} {N_pairs:d}")

    for name in solvent_names:
        N = counts["solvents"][name]["N"]
        lines.append(f"{name:8s} {N:d}")

    lines.append("")

    top_path.write_text("\n".join(lines))


def write_run_scripts(sim_dir: Path) -> None:
    """Create run_local.sh and run_cluster.sh in sim_dir."""
    run_local = sim_dir / "run_local.sh"
    run_cluster = sim_dir / "run_cluster.sh"

    local_text = """#!/bin/bash
set -e

# Energy minimization
gmx grompp -f minim.mdp -c conf.pdb   -p topol.top -o em.tpr
gmx mdrun  -v -deffnm em

# NPT equilibration
gmx grompp -f npt_new_eq.mdp   -c em.gro     -p topol.top -o npt_eq.tpr -maxwarn 1
gmx mdrun  -deffnm npt_eq

# NPT production
gmx grompp -f npt_new_data.mdp -c npt_eq.gro -p topol.top -o npt_data.tpr -maxwarn 1
gmx mdrun  -deffnm npt_data
"""
    run_local.write_text(local_text)
    os.chmod(run_local, 0o755)

    cluster_text = """#!/bin/bash
#SBATCH --account=bcam-exclusive
#SBATCH --partition=bcam-exclusive
########SBATCH --partition=regular 
#SBATCH --job-name=npt_data_traPPE
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --mem=50gb
#SBATCH --cpus-per-task=1
#SBATCH --time=3-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

module load GROMACS/2019.4-foss-2021a 

echo "-------">> /home/oorozco/reporte_jobs/job_log.txt

# Energy minimization
mpirun -np 1               gmx_mpi grompp -f minim.mdp        -c conf.pdb   -p topol.top -o em.tpr
mpirun -np $SLURM_NTASKS   gmx_mpi mdrun  -deffnm em

# NPT equilibration
mpirun -np 1               gmx_mpi grompp -f npt_new_eq.mdp   -c em.gro     -p topol.top -o npt_eq.tpr -maxwarn 1
mpirun -np $SLURM_NTASKS   gmx_mpi mdrun  -deffnm npt_eq

# NPT production
mpirun -np 1               gmx_mpi grompp -f npt_new_data.mdp -c npt_eq.gro -p topol.top -o npt_data.tpr -maxwarn 1
mpirun -np $SLURM_NTASKS   gmx_mpi mdrun  -deffnm npt_data

echo "end the job">> /home/oorozco/reporte_jobs/job_log.txt
echo  $PWD >> /home/oorozco/reporte_jobs/job_log.txt
"""
    run_cluster.write_text(cluster_text)
    os.chmod(run_cluster, 0o755)


# ============================================================
# 5. SIMULATION FOLDER CREATION
# ============================================================

def create_simulation_folder(
    base_dir: Path,
    counts: dict,
    solvent_names: List[str],
    salt_name: str,
    salt_conc_M: float,
    sim_name: str | None,
    charge_scale: float,
) -> tuple[Path, Path]:
    """
    Create the main simulation folder and, inside it, the packmol/ subfolder.

    Returns
    -------
    (sim_dir, packmol_dir)
    """
    box_list = [
        counts["box"]["Lx"],
        counts["box"]["Ly"],
        counts["box"]["Lz"],
    ]

    if sim_name is None:
        sim_name = auto_sim_name(salt_name, salt_conc_M, solvent_names, box_list, charge_scale)

    sim_dir = base_dir / sim_name
    sim_dir.mkdir(exist_ok=True)

    packmol_dir = sim_dir / "packmol"
    packmol_dir.mkdir(exist_ok=True)

    # Copy required PDBs from base_dir/pdb
    pdb_dir = base_dir / "pdb"
    salt_info = SALT_DATA[salt_name]

    # Salt PDBs
    for pdb_name in [salt_info["cation_pdb"], salt_info["anion_pdb"]]:
        src = pdb_dir / pdb_name
        dst = packmol_dir / pdb_name
        if not src.exists():
            raise FileNotFoundError(f"{src} not found (salt PDB).")
        shutil.copy2(src, dst)

    # Solvent PDBs
    for name in solvent_names:
        pdb_name = SOLVENT_PDB_MAP[name]
        src = pdb_dir / pdb_name
        dst = packmol_dir / pdb_name
        if not src.exists():
            raise FileNotFoundError(f"{src} not found (solvent PDB for {name}).")
        shutil.copy2(src, dst)

    return sim_dir, packmol_dir


# ============================================================
# 6. COMMAND-LINE INTERFACE
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Packmol + GROMACS folder generator (TraPPE, LiPF6 + solvents)."
    )

    parser.add_argument(
        "--box",
        nargs=3,
        type=float,
        metavar=("Lx", "Ly", "Lz"),
        required=True,
        help="Box dimensions in Å.",
    )

    parser.add_argument(
        "--salt",
        type=str,
        choices=list(SALT_DATA.keys()),
        default="LiPF6",
        help="Salt name (currently only LiPF6).",
    )

    parser.add_argument(
        "--salt-conc",
        type=float,
        required=True,
        help="Salt concentration in mol/L.",
    )

    parser.add_argument(
        "--solvents",
        nargs="+",
        type=str,
        choices=list(SOLVENT_DATA.keys()),
        required=True,
        help="List of solvents (e.g. EC EMC).",
    )

    parser.add_argument(
        "--solvent-fracs",
        nargs="+",
        type=float,
        required=True,
        help="Relative volume fractions of each solvent (will be normalized).",
    )

    parser.add_argument(
        "--margin",
        type=float,
        default=0.5,
        help="Margin in Å for the Packmol 'inside box' instruction.",
    )

    parser.add_argument(
        "--sim-name",
        type=str,
        default=None,
        help="Name of the simulation folder. If not provided, it is generated automatically.",
    )

    parser.add_argument(
        "--charge-scale",
        type=float,
        default=1.0,
        help="Scaling factor for Li and PF6 charges (e.g. 0.85).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base_dir = Path(__file__).resolve().parent

    # 1) Compute counts
    counts = compute_salt_and_solvent_counts(
        box=args.box,
        salt_name=args.salt,
        salt_conc_M=args.salt_conc,
        solvent_names=args.solvents,
        solvent_fracs=args.solvent_fracs,
    )

    print("=== Box ===")
    print(f"Lx, Ly, Lz [Å] = {args.box}")
    print(f"V_box [Å^3]    = {counts['box']['V_box_A3']:.3f}")
    print(f"V_box [L]      = {counts['box']['V_box_L']:.3e}")
    print()
    print("=== Salt ===")
    print(f"Salt: {counts['salt']['name']}")
    print(f"N_pairs (float) = {counts['salt']['N_pairs_float']:.3f}")
    print(f"N_pairs (int)   = {counts['salt']['N_pairs']}")
    print(f"V_salt [Å^3]    = {counts['salt']['V_salt_A3']:.3f}")
    print()
    print("=== Solvents ===")
    for name, info in counts["solvents"].items():
        print(
            f"{name}: N_float = {info['N_float']:.3f}, "
            f"N = {info['N']}, V_i [Å^3] = {info['V_i_A3']:.3f}"
        )
    print()
    print(f"Charge scaling factor: {args.charge_scale:.4f}")
    print()

    # 2) Create simulation folder and packmol/ subfolder
    sim_dir, packmol_dir = create_simulation_folder(
        base_dir=base_dir,
        counts=counts,
        solvent_names=args.solvents,
        salt_name=args.salt,
        salt_conc_M=args.salt_conc,
        sim_name=args.sim_name,
        charge_scale=args.charge_scale,
    )

    # 3) Generate Packmol input inside packmol/
    inp_text = generate_packmol_input(
        box=args.box,
        salt_name=args.salt,
        counts=counts,
        margin=args.margin,
        output_name="conf.pdb",
    )
    inp_path = packmol_dir / "conf_gen.inp"
    inp_path.write_text(inp_text)
    print(f"Packmol input written to: {inp_path}")

    # 4) Run Packmol
    print(f"Running Packmol in {packmol_dir} ...")
    run_packmol(packmol_dir)
    print("Packmol finished. conf.pdb and packmol.log generated.")

    # 5) Add CRYST1
    add_cryst1_to_pdb(packmol_dir, args.box)
    print("CRYST1 line added to conf.pdb with the specified box size.")

    # 6) Copy forcefield/, itp/ and mdp/ into simulation directory
    ff_src = base_dir / "forcefield"
    itp_src = base_dir / "itp"
    mdp_src_dir = base_dir / "mdp"

    ff_dst = sim_dir / "forcefield"
    itp_dst = sim_dir / "itp"

    shutil.copytree(ff_src, ff_dst, dirs_exist_ok=True)
    shutil.copytree(itp_src, itp_dst, dirs_exist_ok=True)

    if not mdp_src_dir.exists():
        raise FileNotFoundError(f"{mdp_src_dir} not found with .mdp files.")

    for mdp_name in ["minim.mdp", "npt_new_eq.mdp", "npt_new_data.mdp"]:
        src = mdp_src_dir / mdp_name
        if not src.exists():
            raise FileNotFoundError(f"{src} is missing. Create or copy that .mdp into mdp/.")
        dst = sim_dir / mdp_name
        shutil.copy2(src, dst)

    # 7) Rescale charges in Li.itp and PF6.itp (inside simulation directory)
    apply_charge_scaling(itp_dst, args.charge_scale)

    # 8) Copy the "clean" conf.pdb (with CRYST1) from packmol/ to sim_dir/
    conf_src = packmol_dir / "conf.pdb"
    conf_dst = sim_dir / "conf.pdb"
    shutil.copy2(conf_src, conf_dst)

    # 9) Generate topology
    generate_topology(sim_dir, counts, args.solvents, args.salt)

    # 10) Generate run scripts
    write_run_scripts(sim_dir)

    print(f"Simulation folder created at: {sim_dir}")
    print("Key files:")
    print("  - conf.pdb")
    print("  - topol.top")
    print("  - minim.mdp, npt_new_eq.mdp, npt_new_data.mdp")
    print("  - run_local.sh")
    print("  - run_cluster.sh")
    print("  - packmol/ (inputs and log)")


if __name__ == "__main__":
    main()
