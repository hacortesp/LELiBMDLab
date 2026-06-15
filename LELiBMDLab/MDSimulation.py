from __future__ import annotations

from pathlib import Path
from typing import Dict

from LELiBMDLab.GROMACSrun import GROMACSrun


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
    "TFSI": "TFSI.itp",
    "FSI": "FSI.itp",
    "BF4": "BF4.itp",
    "ClO4": "ClO4.itp", 
}


class MDsim:

    def __init__(
        self,
        charge_scale: float,
        workdir: str,
        temperature: float = 298.0,
        *,
        parallel: bool = False,
        equilibration: bool = True,
        gmx_exec: str = "gmx",
        gmx_mpi_exec: str = "gmx_mpi",
        nvt_time_ps: float = 30000,
        time_step_ps: float = 0.002


    ) -> None:
        self.charge_scale = charge_scale
        self.workdir = workdir
        self.temp = temperature

        self.parallel = parallel
        self.equilibration = equilibration
        self.gmx_exec = gmx_exec
        self.gmx_mpi_exec = gmx_mpi_exec

        self.time_step_ps = time_step_ps
        self.steps = int(nvt_time_ps / self.time_step_ps)

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
            equilibration=self.equilibration,
            gmx_exec=self.gmx_exec,
            gmx_mpi_exec=self.gmx_mpi_exec,
            steps=self.steps,
            time_step_ps =self.time_step_ps
        )

        gmx.write_all()
        gmx.run()

        if self.equilibration:
            self.average_volume, self.volume_frame = gmx.analyze_npt_volume(
                start=2000,
                dt_collection=2,
            )
        else:
            self.average_volume = None
            self.volume_frame = None



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
        self.cations: Dict[str, int] = {}

        for mol, n in self.molecule_counts.items():
            if mol == CATION_NAME:
                self.cations[mol] = n
            elif mol in SOLVENT_ITP_MAP:
                self.solvents[mol] = n
            elif mol in SALT_ANION_ITP_MAP:
                self.anions[mol] = n
            else:
                raise KeyError(f"Unknown molecule '{mol}' not in solvent or salt maps.")

        # consistency checks
        if self.anions and not self.cations:
            raise ValueError("Anions present but no cations found.")

        if self.cations and not self.anions:
            raise ValueError("Cations present but no anions found.")

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

        species = []

        if self.cations:
            species.append((CATION_NAME, True))

        for a in self.anions:
            species.append((a, False))

        if not species:
            self.scaled_itp = None
            return

        for mol, has_mass in species:
            src = itp_dir / f"{mol}.itp"
            dst = itp_dir / f"{mol}_{scale_tag}.itp"
            self.scaled_itp[mol] = dst.name

            lines = src.read_text().splitlines()

            charge_idx = -2 if has_mass else -1

            atom_rows = []
            in_atoms = False

            for line_idx, line in enumerate(lines):
                s = line.strip()

                if s.startswith("["):
                    in_atoms = s.lower().startswith("[ atoms")
                    continue

                if not in_atoms or not s or s.startswith(";"):
                    continue

                body, sep, comment = line.partition(";")
                tokens = body.split()

                if len(tokens) < abs(charge_idx):
                    raise ValueError(f"Could not parse atom line in {src}: {line}")

                original_charge = float(tokens[charge_idx])
                atom_rows.append(
                    {
                        "line_idx": line_idx,
                        "tokens": tokens,
                        "comment": (sep + comment) if sep else "",
                        "original_charge": original_charge,
                    }
                )

            if not atom_rows:
                raise ValueError(f"No atoms found in [ atoms ] section of {src}")

            original_total_charge = sum(row["original_charge"] for row in atom_rows)
            target_total_charge = round(
                original_total_charge * self.charge_scale,
                4,
            )

            scaled_charges = [
                round(row["original_charge"] * self.charge_scale, 4)
                for row in atom_rows
            ]

            rounded_total_charge = round(sum(scaled_charges), 4)
            residual_charge = round(target_total_charge - rounded_total_charge, 4)

            if abs(residual_charge) > 0.0:
                # Add the residual to the atom with the largest absolute charge.
                # This minimizes the relative perturbation of the charge distribution.
                correction_idx = max(
                    range(len(scaled_charges)),
                    key=lambda i: abs(scaled_charges[i]),
                )

                scaled_charges[correction_idx] = round(
                    scaled_charges[correction_idx] + residual_charge,
                    4,
                )

            final_total_charge = round(sum(scaled_charges), 4)

            if abs(final_total_charge - target_total_charge) > 1e-4:
                raise ValueError(
                    f"Charge correction failed for {mol}: "
                    f"target={target_total_charge:.4f}, "
                    f"final={final_total_charge:.4f}"
                )

            for row, new_charge in zip(atom_rows, scaled_charges):
                tokens = row["tokens"]
                tokens[charge_idx] = f"{new_charge:.4f}"

                new_line = "  " + "  ".join(tokens)

                if row["comment"]:
                    new_line += "  " + row["comment"].strip()

                lines[row["line_idx"]] = new_line

            dst.write_text("\n".join(lines) + "\n")

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
        if self.cations:
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

        if self.cations:
            lines.append(f"{CATION_NAME} {self.cations[CATION_NAME]}")

        for a, n in self.anions.items():
            lines.append(f"{a} {n}")

        for s, n in self.solvents.items():
            lines.append(f"{s} {n}")

        top.write_text("\n".join(lines) + "\n")
