import numpy as np
from pathlib import Path
import LELiBMDLab.tools.constants as const


ION_FITS = {
    "resname LIP and name LI": (-515.407, 515.409, 0.000),
    "resname _BF and name B1": (-248.408, 248.456, 0.012),
    "resname ClO and name Cl": (-238.867, 239.628, 0.014),
    "resname FSI and name N2": (-199.503, 201.290, 0.033),
    "resname _PF and name P1": (-227.982, 232.768, 0.047),
    "resname TFS and name N1": (-191.297, 205.422, 0.124),
}


def calc_activity(
    c: float,
    eps_solv: float,
    eps_sol: float,
    temp: float,
    *,
    a: float,
    R_plus: float,
    R_minus: float,
):


    if c < 0.0:
        raise ValueError("Concentration must be non-negative.")

    if eps_solv <= 0.0 or eps_sol <= 0.0:
        raise ValueError("Dielectric constants must be positive.")

    if temp <= 0.0:
        raise ValueError("Temperature must be positive.")

    if a <= 0.0 or R_plus <= 0.0 or R_minus <= 0.0:
        raise ValueError("Ion-size and Born radii must be positive.")

    # mol L^-1 -> mol m^-3
    c_mol_m3 = c * 1000.0

    # ========================================================
    # Debye-Hückel part
    # ========================================================

    # Bjerrum-like prefactor for a monovalent ion
    zeta = (
        const.e2c**2
        / ( 8.0 * np.pi* const.eps0 * eps_sol * const.kB * temp )
    )

    # For a fully dissociated 1:1 salt:
    #
    # sum_i(c_i * z_i^2)
    #     = c_cation*(+1)^2 + c_anion*(-1)^2
    #     = 2*c
    #
    # Here c must be expressed in mol m^-3.
    kappa = np.sqrt(
        ( const.NA * const.e2c**2 * 2.0 * c_mol_m3 )
        / ( const.eps0  * eps_sol * const.kB * temp )
    )

    ln_gamma_DH = -(zeta * kappa) / (1.0 + kappa * a)

    # ========================================================
    # Born part
    # ========================================================

    pref_p = (
        const.e2c**2
        / ( 8.0  * np.pi * const.eps0 * const.kB * temp * R_plus )
    )

    pref_m = (
        const.e2c**2
        / ( 8.0 * np.pi * const.eps0 * const.kB * temp * R_minus )
    )

    dielectric_change = ((1.0 / eps_sol) - (1.0 / eps_solv))

    ln_gamma_p = pref_p * dielectric_change
    ln_gamma_m = pref_m * dielectric_change

    # Mean activity coefficient for a 1:1 electrolyte
    ln_gamma_B = 0.5 * (ln_gamma_p + ln_gamma_m)

    # ========================================================
    # Total activity coefficients
    # ========================================================

    gamma_DH = np.exp(ln_gamma_DH)
    gamma_B = np.exp(ln_gamma_B)
    gamma_DH_B = np.exp(ln_gamma_DH + ln_gamma_B)

    return gamma_DH, gamma_B, gamma_DH_B


def born_radius(
    cation_name: str,
    anion_name: str,
    eps_solv: float,
):
    """
    Calculate Born radii from fitted Gibbs solvation energies.
    """

    if eps_solv <= 1.0:
        raise ValueError(
            "eps_solv must be greater than 1 when calculating Born radii."
        )

    delta_g_pos_kjmol = delta_g_from_fit(cation_name, eps_solv)
    delta_g_neg_kjmol = delta_g_from_fit(anion_name, eps_solv)

    #print(f"delta_g_pos = {delta_g_pos_kjmol:.4f} kJ/mol")
    #print(f"delta_g_neg = {delta_g_neg_kjmol:.4f} kJ/mol")

    # kJ mol^-1 -> J per ion
    delta_g_pos = delta_g_pos_kjmol * 1000.0 / const.NA
    delta_g_neg = delta_g_neg_kjmol * 1000.0 / const.NA

    prefactor = 8.0 * np.pi * const.eps0

    z_plus = 1
    z_minus = -1

    R_pos = (
        (z_plus**2 * const.e2c**2)
        / (prefactor * delta_g_pos)
        * ((1.0 / eps_solv) - 1.0)
    )

    R_neg = (
        (z_minus**2 * const.e2c**2)
        / (prefactor * delta_g_neg)
        * ((1.0 / eps_solv) - 1.0)
    )

    if R_pos <= 0.0 or R_neg <= 0.0:
        raise ValueError(
            "Calculated Born radii are not positive. "
            "Check the Gibbs solvation energies and dielectric constant."
        )

    return R_pos, R_neg


def delta_g_from_fit(
    ion_name: str,
    eps_solv: float,
) -> float:
    """
    Calculate Gibbs solvation energy in kJ mol^-1.
    """

    try:
        A, B, C = ION_FITS[ion_name]
    except KeyError as exc:
        raise ValueError(
            f"No fit parameters for ion: {ion_name}"
        ) from exc

    return A + B / (eps_solv + C)


def calc_thermodynamic_factor(
    c_values,
    activity_coefficients_file,
    print_details=False,
):
    """
    Calculate the thermodynamic factor for requested concentrations:

        xi = 1 + d ln(gamma_DH_B) / d ln(c)

    For each requested concentration, the closest concentration in the
    activity-coefficient table is identified. The derivative is then
    evaluated at that tabulated concentration using the rows:

        closest_index - 1
        closest_index
        closest_index + 1

    Parameters
    ----------
    c_values : array-like
        Concentrations at which the thermodynamic factor is requested,
        in mol L^-1. Values must be positive and finite.

    activity_coefficients_file : str or pathlib.Path
        CSV file containing at least these columns:

        - molarity_mol_L
        - ln_molarity_mol_L
        - ln_gamma_DH_B

    Returns
    -------
    therm_factor : numpy.ndarray
        Thermodynamic factors in the same order as c_values.

    Notes
    -----
    The derivative is calculated at the closest tabulated concentration,
    not necessarily at the exact requested concentration.

    If the closest value is the first or last CSV row, the calculation
    cannot use both neighboring rows and a ValueError is raised.
    """

    # ---------------------------------------------------------
    # Validate requested concentrations
    # ---------------------------------------------------------
    c_values = np.asarray(c_values, dtype=float)

    if c_values.ndim == 0:
        c_values = c_values.reshape(1)

    if c_values.ndim != 1:
        raise ValueError("c_values must be one-dimensional.")

    if c_values.size == 0:
        raise ValueError("At least one concentration is required.")

    if not np.all(np.isfinite(c_values)):
        raise ValueError("All requested concentrations must be finite.")

    if np.any(c_values <= 0.0):
        raise ValueError("All requested concentrations must be positive.")

    # ---------------------------------------------------------
    # Read the CSV file
    # ---------------------------------------------------------
    activity_coefficients_file = Path(activity_coefficients_file)

    if not activity_coefficients_file.is_file():
        raise FileNotFoundError(
            f"Activity-coefficient file not found: "
            f"{activity_coefficients_file}"
        )

    table = np.genfromtxt(
        activity_coefficients_file,
        delimiter=",",
        names=True,
        dtype=float,
        encoding="utf-8",
    )

    table = np.atleast_1d(table)

    if table.dtype.names is None:
        raise ValueError(
            "The activity-coefficient file must have a comma-separated "
            "header row."
        )

    required_columns = {
        "molarity_mol_L",
        "ln_molarity_mol_L",
        "gamma_DH_B",
        "ln_gamma_DH_B",
    }

    missing_columns = required_columns.difference(table.dtype.names)

    if missing_columns:
        raise ValueError(
            "The activity-coefficient file is missing the following "
            f"required columns: {sorted(missing_columns)}"
        )

    molarity = np.asarray(
        table["molarity_mol_L"],
        dtype=float,
    )

    ln_molarity = np.asarray(
        table["ln_molarity_mol_L"],
        dtype=float,
    )

    gamma_DH_B = np.asarray(
        table["gamma_DH_B"],
        dtype=float,
    )

    ln_gamma = np.asarray(
        table["ln_gamma_DH_B"],
        dtype=float,
    )

    # ---------------------------------------------------------
    # Validate tabulated data
    # ---------------------------------------------------------
    if molarity.size < 3:
        raise ValueError(
            "The activity-coefficient file must contain at least "
            "three data rows."
        )

    if not (
        np.all(np.isfinite(molarity))
        and np.all(np.isfinite(ln_molarity))
        and np.all(np.isfinite(gamma_DH_B))
        and np.all(np.isfinite(ln_gamma))
    ):
        raise ValueError(
            "The required CSV columns must contain only finite values."
        )

    if np.any(gamma_DH_B <= 0.0):
        raise ValueError(
            "All gamma_DH_B values must be positive."
        )

    if np.any(np.diff(molarity) <= 0.0):
        raise ValueError(
            "The molarity_mol_L column must be strictly increasing."
        )

    # Check that the stored logarithms are consistent with molarity.
    if not np.allclose(
        ln_molarity,
        np.log(molarity),
        rtol=1.0e-7,
        atol=1.0e-8,
    ):
        raise ValueError(
            "ln_molarity_mol_L is inconsistent with "
            "log(molarity_mol_L)."
        )

    # ---------------------------------------------------------
    # Find the closest tabulated molarity
    # ---------------------------------------------------------
    insertion_indices = np.searchsorted(molarity, c_values)

    right_indices = np.clip(
        insertion_indices,
        0,
        molarity.size - 1,
    )
    left_indices = np.clip(
        insertion_indices - 1,
        0,
        molarity.size - 1,
    )

    left_distances = np.abs(
        c_values - molarity[left_indices]
    )
    right_distances = np.abs(
        c_values - molarity[right_indices]
    )

    # If distances are equal, select the lower concentration.
    closest_indices = np.where(
        right_distances < left_distances,
        right_indices,
        left_indices,
    )

    # ---------------------------------------------------------
    # Ensure that every selected row has two neighbors
    # ---------------------------------------------------------
    boundary_mask = (
        (closest_indices == 0)
        | (closest_indices == molarity.size - 1)
    )

    if np.any(boundary_mask):
        invalid_requested = c_values[boundary_mask]
        invalid_matched = molarity[closest_indices[boundary_mask]]

        pairs = ", ".join(
            f"{requested:g} -> {matched:g}"
            for requested, matched in zip(
                invalid_requested,
                invalid_matched,
            )
        )

        raise ValueError(
            "The closest tabulated value is an endpoint and therefore "
            "does not have both neighboring rows. Problematic mappings: "
            f"{pairs}"
        )

    # ---------------------------------------------------------
    # Calculate one thermodynamic factor per requested value
    # ---------------------------------------------------------
    therm_factor = np.empty(c_values.size, dtype=float)

    for output_index, table_index in enumerate(closest_indices):
        row_slice = slice(table_index - 1, table_index + 2)

        local_ln_c = ln_molarity[row_slice]
        local_ln_gamma = ln_gamma[row_slice]

        dln_gamma_dln_c = np.gradient(
            local_ln_gamma,
            local_ln_c,
            edge_order=2,
        )[1]

        therm_factor[output_index] = 1.0 + dln_gamma_dln_c
    
    if print_details:
        print("\n===== Thermodynamic Factor Results =====")

        header = (
            f"{'Requested c':>14} "
            f"{'Closest c':>14} "
            f"{'ln(c)':>14} "
            f"{'gamma_DH_B':>14} "
            f"{'ln(gamma_DH_B)':>18} "
            f"{'TDF':>14}"
        )

        print(header)
        print("-" * len(header))

        for requested_c, table_index, xi in zip(
            c_values,
            closest_indices,
            therm_factor,
        ):
            print(
                f"{requested_c:14.8f} "
                f"{molarity[table_index]:14.8f} "
                f"{ln_molarity[table_index]:14.8f} "
                f"{gamma_DH_B[table_index]:14.8f} "
                f"{ln_gamma[table_index]:18.8f} "
                f"{xi:14.8f}"
            )

    return therm_factor
  