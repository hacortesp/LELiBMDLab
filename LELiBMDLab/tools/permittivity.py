import os
import numpy as np
import MDAnalysis as mda
import LELiBMDLab.tools.constants as const

alphas = {
    "EC":  0.9123,
    "DMC": 1.7903,
    "EMC": 2.0761,
    "PC":  1.0284,
    "DEC": 0.9874,
    "DME": 0.8866,
}

# ---------- DIPOLE CORRECTION FITS ----------
# xi(eps) = a - b / (eps + c)
dipole_correction_xi = {
    "resname ClO and name Cl": (0.9365122651, 1.0474708536, 3.4520340388),  # LiClO4
    "resname _BF and name B1": (0.9354347857, 0.4786633693, 1.6982050863),  # LiBF4
    "resname FSI and name N2": (0.9935864788, 0.6504155227, 0.6632177156),  # LiFSI
    "resname _PF and name P1": (0.9205251214, 0.4224063422, 0.2018108317),  # LiPF6
    "resname TFS and name N1": (0.9205251214, 0.4224063422, 0.2018108317),  # LiTFSI
}

# ---------- solvent selections ----------
solvent_selections = {
    "EC": "resname _EC",
    "DMC": "resname DMC",
    "EMC": "resname EMC",
    "PC": "resname _PC",
    "DEC": "resname DEC",
    "DME": "resname DME",
}

def correction_factor_from_fit(ion_name: str, eps_solv: float) -> float:
    if ion_name not in dipole_correction_xi:
        raise ValueError(f"No dipole correction parameters for ion: {ion_name}")

    a, b, c = dipole_correction_xi[ion_name]

    xi = a - b / (eps_solv + c)
    return xi


def corrected_cip_dipoles(
    start: int,
    end: int,
    cip_array,
    anion: str,
    eps_solv: float,
    q_scale: float,
):

    charge_tolerance = 1e-4
    cip_diagnostics = []
    xi = correction_factor_from_fit(ion_name=anion, eps_solv=eps_solv)

    # The trajectory contains q_sim = q_scale * q_FF.
    # Therefore:
    #
    # q_corr = xi * q_FF
    #        = (xi / q_scale) * q_sim
    xi_effective = xi / q_scale

    # Initialize every analysed frame with zero CIP dipole.
    # Frames without CIPs therefore automatically contain [0, 0, 0].
    cip_dipoles_by_frame = {
        frame: np.zeros(3, dtype=float)
        for frame in range(start, end)
    }

    n_cips = 0

    for cip in cip_array:

        frame = int(cip["frame"])
        positions = np.asarray(cip["positions"], dtype=float)
        charges = np.asarray(cip["charges"], dtype=float)
        
        total_charge = charges.sum()

        if abs(total_charge) > charge_tolerance:
            raise ValueError(
                f"CIP in frame {frame} is not neutral: "
                f"total charge = {total_charge:.8f} e"
            )
        
        # npj Comput Mater 9, 175 (2023)
        # q_corrected = xi * q_MD
        corrected_charges = xi_effective * charges

        # Corrected dipole vector in e·Å:
        # M = sum_i(q_i * r_i)
        cip_dipole = np.dot(
            corrected_charges,
            positions,
        )

        # There may be several CIPs in the same frame.
        cip_dipoles_by_frame[frame] += cip_dipole

        n_cips += 1

    n_frames_with_cip = sum(
        np.any(dipole != 0.0)
        for dipole in cip_dipoles_by_frame.values()
    )

   
    n_frames = end - start
    avg_cips_per_frame = n_cips / n_frames
    print(f"\nAverage number of CIPs per frame: {avg_cips_per_frame:.2f}")

    return cip_dipoles_by_frame


def permittivity_corr(
    start: int,
    end: int,
    run,
    temperature,
    volume_m3,
    cip_dipoles_by_frame=None,
    eps_solv_reference=None,
    print_decomposition=True,
):
    make_whole = True
    charge_tolerance = 1e-6

    atoms = run.atoms

    grouped_atomgroups = {}

    # ---------- select neutral solvent molecules ----------
    for name, selection in solvent_selections.items():

        ag = atoms.select_atoms(selection)

        if ag.n_atoms == 0:
            continue

        neutral_fragments = [
            frag
            for frag in ag.fragments
            if abs(frag.total_charge()) <= charge_tolerance
        ]

        if not neutral_fragments:
            continue

        grouped_ag = neutral_fragments[0]

        for frag in neutral_fragments[1:]:
            grouped_ag += frag

        grouped_atomgroups[name] = grouped_ag

    if not grouped_atomgroups:
        raise ValueError(
            "No neutral solvent atomgroups found."
        )

    # ==========================================================
    # Total dipole accumulators
    # ==========================================================
    M_total = np.zeros(3, dtype=float)
    M2_total = np.zeros(3, dtype=float)

    # ==========================================================
    # Solvent dipole accumulators
    # ==========================================================
    M_solv_total = np.zeros(3, dtype=float)
    M2_solv_total = np.zeros(3, dtype=float)

    # ==========================================================
    # CIP dipole accumulators
    # ==========================================================
    M_cip_total = np.zeros(3, dtype=float)
    M2_cip_total = np.zeros(3, dtype=float)

    # <M_solv * M_cip>, component by component
    M_solv_cip_total = np.zeros(3, dtype=float)

    n_frames = 0

    # ---------- trajectory loop ----------
    for ts in run.trajectory[start:end]:

        # ======================================================
        # Corrected solvent dipole
        # ======================================================
        M_solv = np.zeros(3, dtype=float)

        for name, ag in grouped_atomgroups.items():

            if make_whole:
                ag.unwrap(compound="fragments")

            alpha = alphas.get(name, 1.0)

            M_type = np.dot(
                ag.charges,
                ag.positions,
            )

            # npj Comput Mater 9, 175 (2023)
            # M_corrected = alpha * M_solv
            M_type *= alpha

            M_solv += M_type

        # ======================================================
        # Corrected CIP dipole
        # ======================================================
        if cip_dipoles_by_frame is None:

            # Pure-solvent calculation
            M_cip = np.zeros(3, dtype=float)

        else:

            # Solution calculation
            M_cip = np.asarray(
                cip_dipoles_by_frame.get(
                    ts.frame,
                    np.zeros(3, dtype=float),
                ),
                dtype=float,
            )

        # ======================================================
        # Total dipole for this frame
        #
        # M_total = M_solv + M_cip
        # ======================================================
        frame_M = M_solv + M_cip

        # ---------- total dipole statistics ----------
        M_total += frame_M
        M2_total += frame_M * frame_M

        # ---------- solvent statistics ----------
        M_solv_total += M_solv
        M2_solv_total += M_solv * M_solv

        # ---------- CIP statistics ----------
        M_cip_total += M_cip
        M2_cip_total += M_cip * M_cip

        # ---------- solvent-CIP cross statistics ----------
        M_solv_cip_total += M_solv * M_cip

        n_frames += 1

    if n_frames == 0:
        raise ValueError(
            "No trajectory frames were analysed."
        )

    # ==========================================================
    # Total fluctuation
    # ==========================================================
    M_mean = M_total / n_frames
    M2_mean = M2_total / n_frames

    fluct_total = (
        M2_mean
        - M_mean * M_mean
    )

    # ==========================================================
    # Solvent fluctuation
    # ==========================================================
    M_solv_mean = M_solv_total / n_frames
    M2_solv_mean = M2_solv_total / n_frames

    fluct_solv = (
        M2_solv_mean
        - M_solv_mean * M_solv_mean
    )

    # ==========================================================
    # CIP fluctuation
    # ==========================================================
    M_cip_mean = M_cip_total / n_frames
    M2_cip_mean = M2_cip_total / n_frames

    fluct_cip = (
        M2_cip_mean
        - M_cip_mean * M_cip_mean
    )

    # ==========================================================
    # Solvent-CIP covariance
    #
    # Cov(M_solv, M_cip)
    #     = <M_solv M_cip>
    #     - <M_solv><M_cip>
    # ==========================================================
    M_solv_cip_mean = (
        M_solv_cip_total / n_frames
    )

    cov_solv_cip = (
        M_solv_cip_mean
        - M_solv_mean * M_cip_mean
    )

    # The total fluctuation must satisfy:
    #
    # Var(M_solv + M_cip)
    #     = Var(M_solv)
    #     + Var(M_cip)
    #     + 2 Cov(M_solv, M_cip)
    fluct_reconstructed = (
        fluct_solv
        + fluct_cip
        + 2.0 * cov_solv_cip
    )

    # ==========================================================
    # Convert fluctuations to permittivity
    # ==========================================================
    factor = (
        const.dipole_conv
        / (
            const.eps0
            * const.kB
            * temperature
            * volume_m3
        )
    )

    # ---------- direct total permittivity ----------
    eps_components = (
        1.0
        + factor * fluct_total
    )

    eps_mean = eps_components.mean()

    # ---------- solvent contribution inside solution ----------
    eps_solv_components = (
        1.0
        + factor * fluct_solv
    )

    eps_solv_in_solution = (
        eps_solv_components.mean()
    )

    # ---------- CIP contribution ----------
    delta_eps_cip_components = (
        factor * fluct_cip
    )

    delta_eps_cip = (
        delta_eps_cip_components.mean()
    )

    # ---------- solvent-CIP cross contribution ----------
    delta_eps_cross_components = (
        2.0
        * factor
        * cov_solv_cip
    )

    delta_eps_cross = (
        delta_eps_cross_components.mean()
    )

    # ---------- reconstructed total ----------
    eps_reconstructed = (
        eps_solv_in_solution
        + delta_eps_cip
        + delta_eps_cross
    )

    # ==========================================================
    # Print decomposition
    # ==========================================================
    if print_decomposition:

        reconstruction_error = np.max(
            np.abs(
                fluct_total
                - fluct_reconstructed
            )
        )

        print("\nPermittivity decomposition:")

        if eps_solv_reference is not None:
            print(
                f"  Pure-solvent reference:       "
                f"{eps_solv_reference:.6f}"
            )

        print(
            f"  Solvent in solution:          "
            f"{eps_solv_in_solution:.6f}"
        )

        print(
            f"  CIP contribution:             "
            f"{delta_eps_cip:+.6f}"
        )

        print(
            f"  Solvent-CIP cross term:       "
            f"{delta_eps_cross:+.6f}"
        )

        print(
            f"  Reconstructed total:          "
            f"{eps_reconstructed:.6f}"
        )

        print(
            f"  Direct total:                 "
            f"{eps_mean:.6f}"
        )

        print("\nComponent-wise decomposition:")

        print(
            f"  Solvent epsilon:              "
            f"{eps_solv_components}"
        )

        print(
            f"  CIP delta epsilon:            "
            f"{delta_eps_cip_components}"
        )

        print(
            f"  Cross delta epsilon:          "
            f"{delta_eps_cross_components}"
        )

        print(
            f"  Total epsilon:                "
            f"{eps_components}"
        )

        print("\nFluctuation validation:")

        print(
            f"  Direct fluctuation:           "
            f"{fluct_total}"
        )

        print(
            f"  Reconstructed fluctuation:    "
            f"{fluct_reconstructed}"
        )

        print(
            f"  Maximum reconstruction error: "
            f"{reconstruction_error:.6e} (eÅ)^2"
        )

    return eps_mean
