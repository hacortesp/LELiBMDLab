import os
import numpy as np
import MDAnalysis as mda


q =  1.602176634E-19 # C
eps0_vac = 8.8541878128e-12 # F m^-1
kb = 1.38064852e-23  # Boltzmann Constant, J/K

alphas = {
    "EC":  0.9291,
    "DMC": 1.9028,
    "EMC": 2.0548,
    "PC":  0.9888,
    "DEC": 2.3009,
    "DME": 0.8842,
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

def permittivity_corr(
    start: int,
    end: int,
    universe,
    temperature,
    volume_m3,
    
):
    make_whole = True
    charge_tolerance = 1e-6
   

    # ---------- build grouped atomgroups ----------
    atoms = universe.atoms
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

    # ---------- trajectory accumulation ----------
    M_total = np.zeros(3)
    M2_total = np.zeros(3)

    n_frames = 0

    for ts in universe.trajectory[start:end]:

        frame_M = np.zeros(3)

        for name, ag in grouped_atomgroups.items():
            if make_whole:
                ag.unwrap(compound="fragments")

            alpha = alphas.get(name, 1.0)
            M_type = np.dot(
                ag.charges,
                ag.positions,
            )

            # Equation 8
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

    # ---------- dielectric ----------
    # (C·m)^2 / (e·Å)^2
    dipole_conv = (q * 1e-10) ** 2

    # convert fluctuations
    fluct_si = fluct * dipole_conv

    eps = (
        1
        + fluct_si
        / (eps0_vac * kb * temperature * volume_m3)
    )

    eps_mean = eps.mean()


    print(f"M: {M_mean}")
    print(f"M2: {M2_mean}")
    print(f"fluct: {fluct}")

    print(f"eps: {eps}")
    print(f"eps_mean: {eps_mean}")

    ################################################

    return eps_mean