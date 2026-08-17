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
    "PF6": {
        "Vmol": 168.162990597383,
        "pair_pdb": "LiPF6.pdb",
    },
    "TFSI": {
        "Vmol": 358.439218642812,
        "pair_pdb": "LiTFSI.pdb",
    },
    "FSI": {
        "Vmol": 358.439218642812,
        "pair_pdb": "LiFSI.pdb",
    },
    "BF4": {
        "Vmol": 182.81521655036,
        "pair_pdb": "LiBF4.pdb",
    },
    "ClO4": {
        "Vmol": 77.1461796317142,
        "pair_pdb": "LiClO4.pdb",
    },
}


PACKMOL_EXE = _shutil.which("packmol")
if PACKMOL_EXE is None:
    raise RuntimeError("Packmol executable not found in PATH")


# ============================================================
# PDB SORTING
# ============================================================

def pdb_resnames(pdb_path: Path) -> List[str]:
    """Return unique residue names from a PDB in their original order."""
    names = []
    with pdb_path.open() as handle:
        for line in handle:
            if line[:6].strip() in {"ATOM", "HETATM"}:
                resname = line[17:20].strip()
                if resname not in names:
                    names.append(resname)
    if not names:
        raise ValueError(f"No ATOM or HETATM records found in {pdb_path}")
    return names


def sort_pdb_species(
    input_pdb: Path,
    output_pdb: Path,
    species_order: List[str],
) -> None:
    """
    Sort complete residues into GROMACS topology order.

    Chain IDs A, B, C, ... are assigned according to species_order.
    Coordinates and atom names are unchanged. Atom and residue numbers are
    regenerated. CONECT and MASTER records are removed because their atom
    indices are invalid after sorting.
    """
    if len(species_order) > 26:
        raise ValueError("At most 26 species can be assigned chain IDs A-Z")
    if len(species_order) != len(set(species_order)):
        raise ValueError(f"Repeated residue names in species order: {species_order}")

    headers = []
    residues = []
    current_key = None
    current_lines = []

    with input_pdb.open() as handle:
        for line in handle:
            record = line[:6].strip()

            if record in {"ATOM", "HETATM"}:
                key = (
                    line[17:20].strip(),  # residue name
                    line[21:22],          # chain ID
                    line[22:26],          # residue number
                    line[26:27],          # insertion code
                )

                if current_key is not None and key != current_key:
                    residues.append((current_key[0], current_lines))
                    current_lines = []

                current_key = key
                current_lines.append(line)

            elif record not in {"CONECT", "MASTER", "END", "TER"}:
                headers.append(line)

    if current_key is not None:
        residues.append((current_key[0], current_lines))

    grouped = {resname: [] for resname in species_order}
    for resname, atom_lines in residues:
        if resname not in grouped:
            raise ValueError(
                f"Unexpected residue name {resname!r} in {input_pdb}; "
                f"expected {species_order}"
            )
        grouped[resname].append(atom_lines)

    atom_number = 1
    residue_number = 1

    with output_pdb.open("w") as handle:
        handle.writelines(headers)

        for chain_index, resname in enumerate(species_order):
            chain = chr(ord("A") + chain_index)

            for atom_lines in grouped[resname]:
                for line in atom_lines:
                    line = line.rstrip("\n").ljust(80)
                    handle.write(
                        f"{line[:6]}{atom_number:5d}{line[11:21]}{chain}"
                        f"{residue_number:4d}{line[26:]}\n"
                    )
                    atom_number += 1

                residue_number += 1

        handle.write("END\n")


# ============================================================
# INPUTS BUILDER CLASS
# ============================================================

class Builder:
    """
    Faithful class adaptation of the original Packmol + GROMACS builder.
    Outputs are created in the requested work directory.
    """

    def __init__(
        self,
        box: float,
        salt: str,
        salt_conc: float,
        solvents: List[str],
        solvent_fracs: List[float],
        workdir: str,
    ) -> None:

        self.box = [float(box)] * 3
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
        print("===== Simulation cell =====")
        print(f"{self.box[0]} [Å] x {self.box[1]} [Å] x {self.box[2]} [Å]")

        self.compute_counts()
        c = self.counts

        print(f"V_cell [nm^3]: {c['box']['V_box_A3'] / 1000:.3f}")
        print()

        print("\n\n===== Salt =====")
        print(f"{c['salt']['name']}: {c['salt']['N_pairs']}")
        print()

        print("\n\n===== Solvents =====")
        for name, info in c["solvents"].items():
            print(f"{name}: {info['N']} ")
        print()

        print("\n\n===Creating simulation folders...===")
        self.create_simulation_folders()

        inp = self.write_packmol_input()
        inp_path = self.packmol_dir / "conf_gen.inp"
        inp_path.write_text(inp)

        print(f"====Packmol input written to: {self.packmol_dir}=====")
        print("Running Packmol")
        self.run_packmol()
        print("Packmol finished. Sorted conf.pdb and packmol.log generated.")
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

        if self.salt_conc <= 0.0:
            N_pairs_float = 0.0
            N_pairs = 0
            V_salt_A3 = 0.0
        else:
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

        if self.counts["salt"]["N_pairs"] > 0:
            pair_pdb = SALT_DATA[self.salt]["pair_pdb"]
            shutil.copy2(pdb_src / pair_pdb, self.packmol_dir / pair_pdb)

        for name in self.solvents:
            shutil.copy2(
                pdb_src / SOLVENT_PDB_MAP[name],
                self.packmol_dir / SOLVENT_PDB_MAP[name],
            )

    # ========================================================
    # 3. PACKMOL
    # ========================================================

    def write_packmol_input(self) -> str:
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

        N = self.counts["salt"]["N_pairs"]

        if N > 0:
            salt = SALT_DATA[self.salt]

            lines.extend([
                "# >>>>> Electrolyte contact ion pair",
                f"structure {salt['pair_pdb']}",
                f"   number {N}",
                "   resnumbers 2",
                f"   inside box {x0:.3f} {y0:.3f} {z0:.3f} "
                f"{x1:.3f} {y1:.3f} {z1:.3f}",
                "   nloop 500",
                "end structure",
                "",
            ])

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

    def pdb_species_order(self) -> List[str]:
        """Return residue-name order matching [molecules]: Li, anion, solvents."""
        species_order = []

        if self.counts["salt"]["N_pairs"] > 0:
            pair_path = self.packmol_dir / SALT_DATA[self.salt]["pair_pdb"]
            pair_resnames = pdb_resnames(pair_path)

            if "LI" not in pair_resnames or len(pair_resnames) != 2:
                raise ValueError(
                    f"{pair_path} must contain exactly LI and one anion residue; "
                    f"found {pair_resnames}"
                )

            anion_resname = next(name for name in pair_resnames if name != "LI")
            species_order.extend(["LI", anion_resname])

        for name in self.solvents:
            solvent_path = self.packmol_dir / SOLVENT_PDB_MAP[name]
            solvent_resnames = pdb_resnames(solvent_path)

            if len(solvent_resnames) != 1:
                raise ValueError(
                    f"{solvent_path} must contain one residue name; "
                    f"found {solvent_resnames}"
                )

            species_order.append(solvent_resnames[0])

        return species_order

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

        sort_pdb_species(
            input_pdb=src,
            output_pdb=dst,
            species_order=self.pdb_species_order(),
        )