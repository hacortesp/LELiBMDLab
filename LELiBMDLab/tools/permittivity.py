import os
import numpy as np
import MDAnalysis as mda
import LELiBMDLab.tools.constants as const

alphas = {
    "EC":  0.9291,
    "DMC": 1.9028,
    "EMC": 2.0548,
    "PC":  0.9888,
    "DEC": 2.3009,
    "DME": 0.8842,
}

# ---------- DIPOLE CORRECTION FITS ----------
# xi(eps) = a - b / (eps + c)

DIPOLE_CORRECTION_FITS = {
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
    if ion_name not in DIPOLE_CORRECTION_FITS:
        raise ValueError(f"No dipole correction parameters for ion: {ion_name}")

    a, b, c = DIPOLE_CORRECTION_FITS[ion_name]

    xi = a - b / (eps_solv + c)
    return xi


def permittivity_corr(
    start: int,
    end: int,
    run,
    cation,
    anion,
    temperature,
    volume_m3,
    
):
    make_whole = True
    charge_tolerance = 1e-6
   
    atoms = run.atoms

    grouped_atomgroups = {}

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

    M_total = np.zeros(3)
    M2_total = np.zeros(3)

    n_frames = 0

    for ts in run.trajectory[start:end]:

        frame_M = np.zeros(3)

        for name, ag in grouped_atomgroups.items():
            if make_whole:
                ag.unwrap(compound="fragments")

            alpha = alphas.get(name, 1.0)
            M_type = np.dot(
                ag.charges,
                ag.positions,
            )
            M_type *= alpha

            frame_M += M_type

        M_total += frame_M
        M2_total += frame_M * frame_M
    
        n_frames += 1

    # ---------- averages ----------
    M_mean = M_total / n_frames
    M2_mean = M2_total / n_frames

    # (e·Å)^2
    fluct = M2_mean - M_mean * M_mean

    # convert fluctuations
    fluct_si = fluct * const.dipole_conv

    eps = (
        1
        + fluct_si
        / (const.eps0 * const.kB * temperature * volume_m3)
    )

    eps_mean = eps.mean()


    print(f"M: {M_mean}")
    print(f"M2: {M2_mean}")
    print(f"fluct: {fluct}")

    print(f"eps: {eps}")
    print(f"eps_mean: {eps_mean}")


    return eps_mean