from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict
from LEMDLab.GROMACSrun import GROMACSrun


# ============================================================
# CONSTANTS
# ============================================================

CATION_NAME = "Li"

SOLVENT_ITP_MAP = {
    "EC": "EC_TraPPE.itp",
    "PC": "PC_TraPPE.itp",
    "DMC": "DMC_TraPPE.itp",
    "DEC": "DEC_TraPPE.itp",
    "DME": "DME_TraPPE.itp",
    "EMC": "EMC_TraPPE.itp",
}

SALT_ANION_ITP_MAP = {
    "PF6": "PF6.itp",
    # "TFSI": "TFSI.itp",
    # "FSI": "FSI.itp",
}


class MDsim:

    def __init__(
        self,
        charge_scale: float,
        workdir: str,
        temperature: float = 298.0,
        *,
        parallel: bool = False,
        gmx_exec: str = "gmx",
        gmx_mpi_exec: str = "gmx_mpi",

    ) -> None:
        self.charge_scale = charge_scale
        self.workdir = workdir
        self.temp = temperature

        self.parallel = parallel
        self.gmx_exec = gmx_exec
        self.gmx_mpi_exec = gmx_mpi_exec

        
        self.asset_dir = Path(__file__).resolve().parent
        self.base_dir = Path.cwd()
        self.sim_dir = self.base_dir / self.workdir
        self.packmol_dir = self.sim_dir / "simulation_cell"

        # Packmol → system definition
        self.molecule_counts = self._parse_packmol_input()
        self._classify_molecules()

        # Build simulation (no execution yet)
        self.apply_charge_scaling()
        self.generate_topology()

        # Write and run GROMACS LAST
        self._run_gromacs()

    # ========================================================
    # GROMACS orchestration
    # ========================================================

    def _run_gromacs(self) -> None:        
        gmx = GROMACSrun(
            workdir=self.sim_dir,
            temperature=self.temp,
            parallel=self.parallel,
            gmx_exec=self.gmx_exec,
            gmx_mpi_exec=self.gmx_mpi_exec,
        )

        gmx.write_all()
        gmx.run()

        self.average_volume, self.volume_frame = gmx.analyze_npt_volume(
            start=2000,
            dt_collection=2,
        )


    # ========================================================
    # Packmol parsing
    # ========================================================

    def _parse_packmol_input(self) -> Dict[str, int]:
        inp = self.packmol_dir / "conf_gen.inp"
        if not inp.exists():
            raise FileNotFoundError(f"Missing Packmol input: {inp}")

        counts: Dict[str, int] = {}
        current_mol: str | None = None

        for line in inp.read_text().splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue

            if s.lower().startswith("structure"):
                fname = s.split()[1]
                base = Path(fname).stem
                current_mol = base.replace("_UA", "")
                continue

            if current_mol and s.lower().startswith("number"):
                counts[current_mol] = int(s.split()[1])
                current_mol = None

        if not counts:
            raise ValueError("No molecules parsed from Packmol input.")

        return counts

    # ========================================================
    # Molecule classification
    # ========================================================

    def _classify_molecules(self) -> None:
        self.solvents: Dict[str, int] = {}
        self.anions: Dict[str, int] = {}

        if CATION_NAME not in self.molecule_counts:
            raise ValueError(f"Missing cation '{CATION_NAME}' in Packmol input.")

        for mol, n in self.molecule_counts.items():
            if mol == CATION_NAME:
                continue
            if mol in SOLVENT_ITP_MAP:
                self.solvents[mol] = n
            elif mol in SALT_ANION_ITP_MAP:
                self.anions[mol] = n
            else:
                raise KeyError(f"Unknown molecule '{mol}' not in solvent or salt maps.")

        if not self.anions:
            raise ValueError("No salt anions detected.")

    # ========================================================
    # Charge scaling
    # ========================================================

    def apply_charge_scaling(self) -> None:
        itp_dir = self.asset_dir / "itp"

        for f in itp_dir.glob("*0p*.itp"):
            f.unlink()

        if abs(self.charge_scale - 1.0) < 1e-8:
            self.scaled_itp = None
            return

        scale_tag = f"{self.charge_scale:.4f}".replace(".", "p")
        self.scaled_itp: Dict[str, str] = {}

        species = [(CATION_NAME, True)] + [(a, False) for a in self.anions]

        for mol, has_mass in species:
            src = itp_dir / f"{mol}.itp"
            dst = itp_dir / f"{mol}_{scale_tag}.itp"
            self.scaled_itp[mol] = dst.name

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

    # ========================================================
    # Topology generation
    # ========================================================

    def generate_topology(self) -> None:
        top = self.sim_dir / "topol.top"

        ff = self.asset_dir / "forcefield"
        itp = self.asset_dir / "itp"

        lines = [
            "; Auto-generated topology",
            f'#include "{ff / "forcefield.itp"}"\n',
        ]

        # cation
        li_itp = (
            self.scaled_itp[CATION_NAME]
            if self.scaled_itp
            else f"{CATION_NAME}.itp"
        )
        lines.append(f'#include "{itp / li_itp}"')

        # anions
        for a in self.anions:
            a_itp = (
                self.scaled_itp[a]
                if self.scaled_itp
                else SALT_ANION_ITP_MAP[a]
            )
            lines.append(f'#include "{itp / a_itp}"')

        # solvents
        for s in self.solvents:
            lines.append(f'#include "{itp / SOLVENT_ITP_MAP[s]}"')

        lines += [
            "",
            "[ system ]",
            "Electrolyte system",
            "",
            "[ molecules ]",
        ]

        lines.append(f"{CATION_NAME} {self.molecule_counts[CATION_NAME]}")

        for a, n in self.anions.items():
            lines.append(f"{a} {n}")

        for s, n in self.solvents.items():
            lines.append(f"{s} {n}")

        top.write_text("\n".join(lines) + "\n")
