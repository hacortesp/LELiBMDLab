import numpy as np

# ============================================================
# Physical constants (SI)
# ============================================================
e = 1.602176634e-19 # C
eps0_vac = 8.8541878128e-12 # F m^-1
kB = 1.380649e-23 # J K^-1
NA = 6.02214076e23 # mol^-1
AMU_TO_KG = 1.66053906660e-27

def calc_activity(
    c: float,
    rho_solv: float,
    eps_solv: float,
    eps_sol: float,
    temp: float,
    *,
    a: float,
    R_plus: float,
    R_minus: float
):
    """
    Calculate mean activity coefficients:
    γ_DH, γ_Born, γ_DH+B

    Parameters
    ----------
    c : float
        Molality (mol kg^-1)
    rho_solv : float
        Solvent density (kg m^-3)
    eps_solv : float
        Dielectric constant of pure solvent
    eps_sol : float
        Dielectric constant of solution at c
    a : float
        Ion-size parameter (m)
    R_plus : float
        Born radius of cation (m)
    R_minus : float
        Born radius of anion (m)

    Returns
    -------
    gamma_DH : float
    gamma_B : float
    gamma_DH_B : float
    """

    # ---------------- DH part ----------------
    zeta = e**2 / (8.0 * np.pi * eps0_vac * eps_sol * kB * temp)

    kappa = np.sqrt(
        (NA * e**2 * rho_solv * 2.0 * c) /
        (eps0_vac * eps_sol * kB * temp)
    )

    ln_gamma_DH = -(zeta * kappa) / (1.0 + kappa * a)

    # ---------------- Born part ----------------
    pref_p = e**2 / (8.0 * np.pi * eps0_vac * kB * temp * R_plus)
    pref_m = e**2 / (8.0 * np.pi * eps0_vac * kB * temp * R_minus)

    ln_gamma_p = pref_p * ((1.0 / eps_sol) - (1.0 / eps_solv))
    ln_gamma_m = pref_m * ((1.0 / eps_sol) - (1.0 / eps_solv))

    ln_gamma_B = 0.5 * (ln_gamma_p + ln_gamma_m)

    # ---------------- Total ----------------
    gamma_DH = np.exp(ln_gamma_DH)
    gamma_B = np.exp(ln_gamma_B)
    gamma_DH_B = np.exp(ln_gamma_DH + ln_gamma_B)

    return gamma_DH, gamma_B, gamma_DH_B


def born_radius(z_plus, z_minus, eps_solv):
    delta_g_kj_mol_plus = -509 if z_plus > 0 else -305
    delta_g_pos = delta_g_kj_mol_plus * 1e3 / NA  # J per particle

    delta_g_kj_mol_minus = -509 if z_minus > 0 else -305
    delta_g_neg = delta_g_kj_mol_minus * 1e3 / NA  # J per particle

    R_pos = (z_plus**2 * e**2) / (8.0 * np.pi * eps0_vac * delta_g_pos) * ((1.0 / eps_solv) - 1.0)
    R_neg = (z_minus**2 * e**2) / (8.0 * np.pi * eps0_vac * delta_g_neg) * ((1.0 / eps_solv) - 1.0)
    return R_pos * 1e10, R_neg * 1e10  # meters → Å




"""
def ion_charge(universe, resname: str, tol: float = 1e-3) -> float:
    ag = universe.select_atoms(resname)

    if ag.n_atoms == 0:
        raise ValueError(f"No atoms found for selection: {resname}")

    # group by residue → one ion per residue
    charges = []
    for res in ag.residues:
        q = res.atoms.charges.sum()
        charges.append(q)

    charges = np.array(charges)
    q_mean = charges.mean()

    if np.std(charges) > tol:
        raise ValueError(
            f"Non-uniform charges detected for {resname}: {charges}"
        )

    if abs(q_mean - round(q_mean)) > tol:
        raise ValueError(
            f"Ionic charge for {resname} not integer: {q_mean}"
        )

    return float(round(q_mean))
"""