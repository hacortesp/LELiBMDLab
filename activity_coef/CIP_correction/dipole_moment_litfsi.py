from pathlib import Path
import re
import numpy as np
import pandas as pd


# ============================================================
# User settings
# ============================================================
COORD_DIR = Path("coordinates_lipf6")
OUTPUT_CSV = "lipf6_dipoles.csv"

# Charges in units of elementary charge, e
CHARGES = {
    "Li": 1.000,
    "P":  1.34,
    "F": -0.39,
}

# Conversion factor
# 1 e Angstrom = 4.803204712570263 Debye
EANG_TO_DEBYE = 4.803204712570263


# ============================================================
# Utilities
# ============================================================
def natural_sort_key(path: Path):
    """
    Sort coordinate_2, coordinate_2p5, coordinate_3, ...
    in numeric order.
    """
    name = path.name
    match = re.search(r"coordinate_(.+)$", name)
    if match is None:
        return name

    value = match.group(1).replace("p", ".")
    try:
        return float(value)
    except ValueError:
        return name


def read_coordinate_file(path: Path):
    atoms = []
    coords = []

    with path.open("r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 4:
                continue

            atom = parts[0]
            xyz = [float(parts[1]), float(parts[2]), float(parts[3])]

            if atom not in CHARGES:
                raise ValueError(f"Unknown atom '{atom}' in file {path}")

            atoms.append(atom)
            coords.append(xyz)

    return atoms, np.asarray(coords, dtype=float)


def calculate_dipole(atoms, coords):
    """
    Computes M = sum_i q_i r_i.

    Returns:
        dipole_vector_eA : np.ndarray, shape (3,)
        dipole_magnitude_eA : float
    """
    charges = np.array([CHARGES[atom] for atom in atoms], dtype=float)

    total_charge = charges.sum()
    if abs(total_charge) > 1e-8:
        print(
            f"Warning: total charge is {total_charge:.6f} e. "
            "For non-neutral systems, the dipole depends on the coordinate origin."
        )

    dipole_vector_eA = np.sum(charges[:, None] * coords, axis=0)
    dipole_magnitude_eA = np.linalg.norm(dipole_vector_eA)

    return dipole_vector_eA, dipole_magnitude_eA, total_charge


# ============================================================
# Main calculation
# ============================================================
records = []

coordinate_files = sorted(
    COORD_DIR.glob("coordinate_*"),
    key=natural_sort_key,
)

if not coordinate_files:
    raise FileNotFoundError(f"No files matching coordinate_* found in {COORD_DIR}")

for coord_file in coordinate_files:
    atoms, coords = read_coordinate_file(coord_file)

    dipole_vec_eA, dipole_mag_eA, total_charge = calculate_dipole(atoms, coords)

    dipole_vec_D = dipole_vec_eA * EANG_TO_DEBYE
    dipole_mag_D = dipole_mag_eA * EANG_TO_DEBYE

    records.append(
        {
            "file": coord_file.name,
            "n_atoms": len(atoms),
            "total_charge_e": total_charge,
            "mu_x_eA": dipole_vec_eA[0],
            "mu_y_eA": dipole_vec_eA[1],
            "mu_z_eA": dipole_vec_eA[2],
            "mu_total_eA": dipole_mag_eA,
            "mu_x_D": dipole_vec_D[0],
            "mu_y_D": dipole_vec_D[1],
            "mu_z_D": dipole_vec_D[2],
            "mu_total_D": dipole_mag_D,
        }
    )

df = pd.DataFrame(records)
df.to_csv(OUTPUT_CSV, index=False)

print(df.to_string(index=False))
print(f"\nSaved results to: {OUTPUT_CSV}")