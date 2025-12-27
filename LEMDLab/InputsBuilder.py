from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict
import shutil as _shutil

# ============================================================
# CONSTANTS
# ============================================================

NA = 6.022141e23

SOLVENT_DATA: Dict[str, Dict[str, float]] = {
    "EC":   {"M": 88.0632,  "rho": 1.323, "Vmol": 110.530902479474},
    "PC":   {"M": 102.0902, "rho": 1.2,   "Vmol": 141.270637896326},
    "DMC":  {"M": 90.0792,  "rho": 1.063, "Vmol": 140.714986584917},
    "DEC":  {"M": 118.1332, "rho": 0.98,  "Vmol": 200.168156867614},
    "DME":  {"M": 90.1228,  "rho": 0.86,  "Vmol": 174.014453771041},
    "EMC":  {"M": 104.1062, "rho": 1.006, "Vmol": 171.841364050710},
}

SOLVENT_PDB_MAP = {
    "EC": "EC_UA.pdb",
    "PC": "PC_UA.pdb",
    "DMC": "DMC_UA.pdb",
    "DEC": "DEC_UA.pdb",
    "DME": "DME_UA.pdb",
    "EMC": "EMC_UA.pdb",
}

SOLVENT_ITP_MAP = {
    "EC": "EC_TraPPE.itp",
    "PC": "PC_TraPPE.itp",
    "DMC": "DMC_TraPPE.itp",
    "DEC": "DEC_TraPPE.itp",
    "DME": "DME_TraPPE.itp",
    "EMC": "EMC_TraPPE.itp",
}

SALT_DATA = {
    "LiPF6": {
        "Vmol": 168.162990597383,
        "cation_pdb": "Li.pdb",
        "anion_pdb": "PF6.pdb",
    }
}

SALT_ITP_MAP = {
    "LiPF6": {
        "cation_itp": "Li.itp",
        "anion_itp": "PF6.itp",
        "cation_mt": "Li",
        "anion_mt": "PF6",
    }
}

PACKMOL_EXE = _shutil.which("packmol")
if PACKMOL_EXE is None:
    raise RuntimeError("Packmol executable not found in PATH")

# ============================================================
# INPUTS BUILDER CLASS
# ============================================================

class InputsBuilder:
    """
    Faithful class adaptation of the original Packmol + GROMACS builder.
    Outputs are created in ./input_files relative to execution directory.
    """

    def __init__(
        self,
        box: List[float],
        salt: str,
        salt_conc: float,
        solvents: List[str],
        solvent_fracs: List[float],
        margin: float,
        charge_scale: float,
        run_mode: str,
    ) -> None:

        self.box = box
        self.salt = salt
        self.salt_conc = salt_conc
        self.solvents = solvents
        self.solvent_fracs = solvent_fracs
        self.margin = margin
        self.charge_scale = charge_scale
        self.run_mode = run_mode

        # Assets live with the code
        self.asset_dir = Path(__file__).resolve().parent
        # Outputs live where builder.py is executed
        self.work_dir = Path.cwd()
        self.sim_dir = self.work_dir / "input_files"
        self.packmol_dir = self.sim_dir / "packmol"

        self.counts: dict | None = None

        self.run()

    # ========================================================
    # MAIN PIPELINE
    # ========================================================

    def run(self) -> None:
        print("=== Box ===")
        print(f"Lx, Ly, Lz [Å] = {self.box}")

        self.compute_counts()
        c = self.counts

        print(f"V_box [Å^3]    = {c['box']['V_box_A3']:.3f}")
        print()

        print("=== Salt ===")
        print(f"Salt: {c['salt']['name']}")
        print(f"N_pairs (int)   = {c['salt']['N_pairs']}")
        print()

        print("=== Solvents ===")
        for name, info in c["solvents"].items():
            print(
                f"{name}: N_float = {info['N_float']:.3f}, "
                f"N = {info['N']}, V_i [Å^3] = {info['V_i_A3']:.3f}"
            )
        print()

        print("Creating simulation folders...")
        self.create_simulation_folders()

        inp = self.write_packmol_input()
        inp_path = self.packmol_dir / "conf_gen.inp"
        inp_path.write_text(inp)

        print(f"Packmol input written to: {self.packmol_dir}")
        print(f"Running Packmol")
        self.run_packmol()
        print("Packmol finished. conf.pdb and packmol.log generated.")
        print()
        self.add_cryst1()
        self.copy_gromacs_files()
        self.apply_charge_scaling()
        self.generate_topology()
        self.write_run_scripts()

        print()
        print(f"Simulation folder created at: {self.sim_dir}")
        print("Key files:")
        print("  - conf.pdb")
        print("  - topol.top")
        print("  - minim.mdp, npt_new_eq.mdp, npt_new_data.mdp")
        print("  - run_local.sh")
        print("  - run_cluster.sh")
      


    # ========================================================
    # 1. COUNTS
    # ========================================================

    def compute_counts(self) -> None:
        frac_sum = sum(self.solvent_fracs)
        norm_fracs = [f / frac_sum for f in self.solvent_fracs]

        Lx, Ly, Lz = self.box
        V_box_A3 = Lx * Ly * Lz
        V_box_L = V_box_A3 * 1e-27

        n_salt_mol = self.salt_conc * V_box_L
        N_pairs_float = n_salt_mol * NA
        N_pairs = max(1, int(round(N_pairs_float)))

        V_salt_A3 = N_pairs_float * SALT_DATA[self.salt]["Vmol"]
        V_solvent_A3 = V_box_A3 - V_salt_A3

        solvents = {}
        for name, frac in zip(self.solvents, norm_fracs):
            V_i_A3 = frac * V_solvent_A3
            Vmol = SOLVENT_DATA[name]["Vmol"]
            N_float = V_i_A3 / Vmol
            solvents[name] = {
                "N_float": N_float,
                "N": max(1, int(round(N_float))),
                "V_i_A3": V_i_A3,
            }

        self.counts = {
            "box": {
                "Lx": Lx,
                "Ly": Ly,
                "Lz": Lz,
                "V_box_A3": V_box_A3,
                "V_box_L": V_box_L,
            },
            "salt": {
                "name": self.salt,
                "N_pairs_float": N_pairs_float,
                "N_pairs": N_pairs,
                "V_salt_A3": V_salt_A3,
            },
            "solvents": solvents,
        }

    # ========================================================
    # 2. FILE SYSTEM
    # ========================================================

    def create_simulation_folders(self) -> None:
        self.sim_dir.mkdir(exist_ok=True)
        self.packmol_dir.mkdir(exist_ok=True)

        pdb_src = self.asset_dir / "pdb"
        salt = SALT_DATA[self.salt]

        for pdb in (salt["cation_pdb"], salt["anion_pdb"]):
            shutil.copy2(pdb_src / pdb, self.packmol_dir / pdb)

        for name in self.solvents:
            shutil.copy2(
                pdb_src / SOLVENT_PDB_MAP[name],
                self.packmol_dir / SOLVENT_PDB_MAP[name],
            )

    # ========================================================
    # 3. PACKMOL
    # ========================================================

    def write_packmol_input(self) -> None:
        Lx, Ly, Lz = self.box
        m = self.margin

        x0, y0, z0 = m, m, m
        x1, y1, z1 = Lx - m, Ly - m, Lz - m

        lines = [
            "# Auto-generated Packmol input",
            "tolerance 2.0",
            "filetype pdb",
            "output conf.pdb",
            "",
            "seed 42",
            "",
        ]

        salt = SALT_DATA[self.salt]
        N = self.counts["salt"]["N_pairs"]

        # Li+
        lines.extend([
            "# >>>>> Electrolyte cation",
            f"structure {salt['cation_pdb']}",
            f"   number {N}",
            "   resnumbers 3",
            f"   inside box {x0:.3f} {y0:.3f} {z0:.3f} "
            f"{x1:.3f} {y1:.3f} {z1:.3f}",
            "   nloop 500",
            "end structure",
            "",
        ])

        # PF6-
        lines.extend([
            "# >>>>> Electrolyte anion",
            f"structure {salt['anion_pdb']}",
            f"   number {N}",
            "   resnumbers 3",
            f"   inside box {x0:.3f} {y0:.3f} {z0:.3f} "
            f"{x1:.3f} {y1:.3f} {z1:.3f}",
            "   nloop 500",
            "end structure",
            "",
        ])

        # Solvents
        for name, info in self.counts["solvents"].items():
            lines.extend([
                f"# >>>>> Solvent: {name}",
                f"structure {SOLVENT_PDB_MAP[name]}",
                f"   number {info['N']}",
                "   resnumbers 3",
                f"   inside box {x0:.3f} {y0:.3f} {z0:.3f} "
                f"{x1:.3f} {y1:.3f} {z1:.3f}",
                "   nloop 500",
                "end structure",
                "",
            ])

        return "\n".join(lines) + "\n"

    def run_packmol(self) -> None:
        inp_path = self.packmol_dir / "conf_gen.inp"
        log_path = self.packmol_dir / "packmol.log"

        with inp_path.open("r") as fin, log_path.open("w") as flog:
            result = subprocess.run(
                [PACKMOL_EXE],
                stdin=fin,
                stdout=flog,
                stderr=subprocess.STDOUT,
                cwd=self.packmol_dir,
            )

        if result.returncode != 0:
            raise RuntimeError(
                f"Packmol failed (code {result.returncode}). "
                f"Check {log_path}"
            )

    def add_cryst1(self) -> None:
        pdb = self.packmol_dir / "conf.pdb"
        lines = pdb.read_text().splitlines()
        atoms = [l for l in lines if l.startswith(("ATOM", "HETATM"))]
        Lx, Ly, Lz = self.box
        cryst1 = f"CRYST1{Lx:9.3f}{Ly:9.3f}{Lz:9.3f}  90.00  90.00  90.00 P 1           1"
        pdb.write_text("\n".join([cryst1] + atoms) + "\n")

        src = self.packmol_dir / "conf.pdb"
        dst = self.sim_dir / "conf.pdb"
        shutil.copy2(src, dst)

    # ========================================================
    # 4. GROMACS FILES
    # ========================================================

    def copy_gromacs_files(self) -> None:
        mdp_src = self.asset_dir / "mdp"

        if not mdp_src.exists():
            raise FileNotFoundError(f"Missing mdp directory: {mdp_src}")

        for mdp in ("minim.mdp", "npt_new_eq.mdp", "npt_new_data.mdp"):
            shutil.copy2(mdp_src / mdp, self.sim_dir / mdp)
            
    def apply_charge_scaling(self) -> None:
        itp_dir = self.asset_dir / "itp"

        for f in itp_dir.glob("*0p*.itp"):
            f.unlink()
            
        if abs(self.charge_scale - 1.0) < 1e-8:
            print("No charge scaling applied.")
            print()
            self.scaled_itp = None
            return

        print(f"Charge scaling factor: {self.charge_scale:.4f}")
        print()

        scale_tag = f"{self.charge_scale:.4f}".replace(".", "p")

        self.scaled_itp = {
            "Li": f"Li_{scale_tag}.itp",
            "PF6": f"PF6_{scale_tag}.itp",
        }

        for base, has_mass in [("Li", True), ("PF6", False)]:
            src = itp_dir / f"{base}.itp"
            dst = itp_dir / self.scaled_itp[base]

            lines = src.read_text().splitlines()
            new = []
            in_atoms = False

            for line in lines:
                s = line.strip()
                if s.startswith("["):
                    in_atoms = s.lower().startswith("[ atoms")
                    new.append(line)
                    continue

                if in_atoms and s and not s.startswith(";"):
                    t = s.split()
                    idx = -2 if has_mass else -1
                    t[idx] = f"{float(t[idx]) * self.charge_scale:.4f}"
                    new.append("  " + "  ".join(t))
                else:
                    new.append(line)

            dst.write_text("\n".join(new) + "\n")

            print(f"Charge-scaled ITP written: {dst}")

    def generate_topology(self) -> None:
        top = self.sim_dir / "topol.top"

        ff = self.asset_dir / "forcefield"
        itp = self.asset_dir / "itp"

        # Select ITPs depending on charge scaling
        if getattr(self, "scaled_itp", None):
            li_itp = self.scaled_itp["Li"]
            pf6_itp = self.scaled_itp["PF6"]
        else:
            li_itp = "Li.itp"
            pf6_itp = "PF6.itp"

        lines = [
            "; Auto-generated topology",
            f'#include "{ff / "forcefield.itp"}"\n',
            f'#include "{itp / li_itp}"',
            f'#include "{itp / pf6_itp}"',
        ]
        for s in self.solvents:
            lines.append(f'#include "{itp / SOLVENT_ITP_MAP[s]}"')

        lines += ["", "[ system ]", "Electrolyte system", "", "[ molecules ]"]
        N = self.counts["salt"]["N_pairs"]
        lines += [f"Li {N}", f"PF6 {N}"]
        for s in self.solvents:
            lines.append(f"{s} {self.counts['solvents'][s]['N']}")

        top.write_text("\n".join(lines))

    def write_run_scripts(self) -> None:
        if self.run_mode == "local":
            run_local = self.sim_dir / "run_local.sh"
            run_local.write_text("""#!/bin/bash
    set -e

    gmx grompp -f minim.mdp -c conf.pdb -p topol.top -o em.tpr
    gmx mdrun -v -deffnm em

    gmx grompp -f npt_new_eq.mdp -c em.gro -p topol.top -o npt_eq.tpr -maxwarn 1
    gmx mdrun -deffnm npt_eq

    gmx grompp -f npt_new_data.mdp -c npt_eq.gro -p topol.top -o npt_data.tpr -maxwarn 1
    gmx mdrun -deffnm npt_data
    """)
            os.chmod(run_local, 0o755)
            print("Local run script created: run_local.sh")

        elif self.run_mode == "server":
            run_cluster = self.sim_dir / "run_cluster.sh"
            run_cluster.write_text("""#!/bin/bash
    #SBATCH --job-name=npt_data
    #SBATCH --nodes=1
    #SBATCH --ntasks-per-node=48
    #SBATCH --time=3-00:00:00

    module load GROMACS/2019.4

    mpirun -np 1 gmx_mpi grompp -f minim.mdp -c conf.pdb -p topol.top -o em.tpr
    mpirun -np $SLURM_NTASKS gmx_mpi mdrun -deffnm em

    mpirun -np 1 gmx_mpi grompp -f npt_new_eq.mdp -c em.gro -p topol.top -o npt_eq.tpr -maxwarn 1
    mpirun -np $SLURM_NTASKS gmx_mpi mdrun -deffnm npt_eq

    mpirun -np 1 gmx_mpi grompp -f npt_new_data.mdp -c npt_eq.gro -p topol.top -o npt_data.tpr -maxwarn 1
    mpirun -np $SLURM_NTASKS gmx_mpi mdrun -deffnm npt_data
    """)
            os.chmod(run_cluster, 0o755)
            print("Cluster run script created: run_cluster.sh")

    