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

SALT_DATA = {
    "LiPF6": {
        "Vmol": 168.162990597383,
        "cation_pdb": "Li.pdb",
        "anion_pdb": "PF6.pdb",
    }
}


PACKMOL_EXE = _shutil.which("packmol")
if PACKMOL_EXE is None:
    raise RuntimeError("Packmol executable not found in PATH")

# ============================================================
# INPUTS BUILDER CLASS
# ============================================================

class Builder:
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
        workdir: str,
    ) -> None:

        self.box = box
        self.salt = salt
        self.salt_conc = salt_conc
        self.solvents = solvents
        self.solvent_fracs = solvent_fracs
        self.workdir = workdir

        # Assets live with the code
        self.asset_dir = Path(__file__).resolve().parent
        # Outputs live where builder.py is executed
        self.base_dir = Path.cwd()
        self.sim_dir = self.base_dir / self.workdir
        self.packmol_dir = self.sim_dir / "simulation_cell"

        self.counts: dict | None = None

        self.run()

    # ========================================================
    # MAIN PIPELINE
    # ========================================================

    def run(self) -> None:
        print("=== Box ===")
        print(f"{self.box[0]} [Å] x {self.box[1]} [Å] x {self.box[2]} [Å]")

        self.compute_counts()
        c = self.counts

        print(f"V_box [Å^3]: {c['box']['V_box_A3']:.3f}")
        print()

        print("=== Salt ===")
        print(f"{c['salt']['name']}: {c['salt']['N_pairs']}")
        print()

        print("=== Solvents ===")
        for name, info in c["solvents"].items():
            print(
                f"{name}: {info['N']} "
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
        m = 0.5

        x0, y0, z0 = m, m, m
        x1, y1, z1 = Lx - m, Ly - m, Lz - m

        lines = [
            "# Auto-generated Packmol input",
            "tolerance 2.0",
            "add_box_sides 0.5",
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
            f"{Lx:.3f} {Ly:.3f} {Lz:.3f}",
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


        src = self.packmol_dir / "conf.pdb"
        dst = self.sim_dir / "conf.pdb"
        shutil.copy2(src, dst)

