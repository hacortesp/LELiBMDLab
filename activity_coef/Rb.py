import numpy as np

# ============================================================
# Physical constants (SI)
# ============================================================
e = 1.602176634e-19        # C
eps0 = 8.8541878128e-12   # F m^-1
NA = 6.02214076e23        # mol^-1

# ============================================================
# PCM dielectric used in DFT (critical!)
# ============================================================
EPS_PCM = 78.3  # Gaussian PCM dielectric at 298 K

# ============================================================
# DFT solvation energies (SI Table 3) [kJ/mol]
# ============================================================
DELTA_G_DFT = {
    "Li+": (-509, +1),
    "Na+": (-418, +1),
    "Cl-": (-305, -1),
    "Br-": (-281, -1),
}

# ============================================================
# Born radius from SI Eq. (1)
# ============================================================
def born_radius_from_dft(delta_g_kj_mol, z, eps_r):
    """
    Invert Supplementary Eq. (1) to obtain R_B
    """
    delta_g = delta_g_kj_mol * 1e3 / NA  # J per particle
    print(delta_g, delta_g_kj_mol)


    R = (z**2 * e**2) / (8.0 * np.pi * eps0 * delta_g) * ((1.0 / eps_r) - 1.0)
    return R * 1e10  # meters → Å


# ============================================================
# Compute and print
# ============================================================
print(f"{'Ion':<6} {'RB (Å)':>10}")
print("-" * 18)

for ion, (dg, z) in DELTA_G_DFT.items():
    rb = born_radius_from_dft(dg, z, EPS_PCM)
    print(f"{ion:<6} {rb:10.3f}")
