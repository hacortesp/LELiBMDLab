import numpy as np

# ============================================================
# Physical constants (SI)
# ============================================================
e = 1.602176634e-19 # C
eps0_vac = 8.8541878128e-12 # F m^-1
kB = 1.380649e-23 # J K^-1
NA = 6.02214076e23 # mol^-1
AMU_TO_KG = 1.66053906660e-27

ION_FITS = {
    "resname LIP and name LI": (-515.407, 515.409, 0.000),
    "resname _BF and name B1": (-248.408, 248.456, 0.012),
    "resname ClO and name Cl": (-238.867, 239.628, 0.014),
    "resname FSI and name N2": (-199.503, 201.290, 0.033),
    "resname _PF and name P1": (-227.982, 232.768, 0.047),
    "resname TFS and name N1": (-190.692, 206.142, 0.143),
}

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


def born_radius(
    cation_name: str,
    anion_name: str,
    eps_solv: float,
):
    delta_g_pos_kjmol = delta_g_from_fit(cation_name, eps_solv)
    delta_g_neg_kjmol = delta_g_from_fit(anion_name, eps_solv)

    print(f"delta_g_pos = {delta_g_pos_kjmol:.4f} kJ/mol")
    print(f"delta_g_neg = {delta_g_neg_kjmol:.4f} kJ/mol")

    # kJ/mol -> J per ion
    delta_g_pos = delta_g_pos_kjmol * 1000.0 / NA
    delta_g_neg = delta_g_neg_kjmol * 1000.0 / NA

    prefactor = 8.0 * np.pi * eps0_vac

    z_plus = 1; z_minus = -1
    R_pos = (
        (z_plus**2 * e**2)
        / (prefactor * delta_g_pos)
        * ((1.0 / eps_solv) - 1.0)
    )

    R_neg = (
        (z_minus**2 * e**2)
        / (prefactor * delta_g_neg)
        * ((1.0 / eps_solv) - 1.0)
    )

    return R_pos, R_neg

def delta_g_from_fit(ion_name: str, eps_solv: float) -> float:
    if ion_name not in ION_FITS:
        raise ValueError(f"No fit parameters for ion: {ion_name}")

    A, B, C = ION_FITS[ion_name]

    delta_g_kj_mol = A + B / (eps_solv + C)
    return delta_g_kj_mol 

