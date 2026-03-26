#!/usr/bin/env python3
"""
Generate synthetic MD electrolyte dataset for charge-scaling optimization testing.

Creates 1000 synthetic data points including:
- Salt type
- Solvent composition (EC, PC, DMC, EMC, DEC, DME)
- Concentration
- Temperature
- Charge scaling factor (alpha)
- Conductivity
- Diffusivity
- Permittivity
- Coordination number
- Activity coefficient

Output:
    synthetic_md_electrolyte_dataset.csv
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(seed=42)

N_SAMPLES = 1000
OUTPUT_FILE = "synthetic_md_electrolyte_dataset.csv"

SALTS = ["PF6", "BF4", "TFSI", "FSI"]
SOLVENTS = ["EC", "PC", "DMC", "EMC", "DEC", "DME"]

# Relative intrinsic salt factors (synthetic physics diversity)
SALT_FACTORS = {
    "PF6": 1.0,
    "BF4": 0.9,
    "TFSI": 1.2,
    "FSI": 1.15,
}

# Relative dielectric constants (approximate physical trend)
SOLVENT_DIELECTRIC = {
    "EC": 90.0,
    "PC": 65.0,
    "DMC": 3.0,
    "EMC": 3.0,
    "DEC": 2.8,
    "DME": 7.2,
}


def sample_solvent_composition() -> dict:
    """Sample random solvent mixture using Dirichlet distribution."""
    fractions = RNG.dirichlet(np.ones(len(SOLVENTS)))
    return dict(zip(SOLVENTS, fractions))


def compute_mixture_permittivity(composition: dict) -> float:
    """Weighted dielectric constant."""
    return sum(
        composition[solvent] * SOLVENT_DIELECTRIC[solvent]
        for solvent in SOLVENTS
    )


def compute_viscosity(concentration: float, temperature: float) -> float:
    """
    Synthetic viscosity model.
    Higher concentration -> higher viscosity.
    Higher temperature -> lower viscosity.
    """
    base = 1.0 + 0.8 * concentration
    thermal = np.exp(300.0 / temperature)
    return base * thermal


def compute_coordination_number(alpha: float, concentration: float) -> float:
    """
    Synthetic ion pairing model.
    Higher alpha and concentration -> more pairing.
    """
    return 3.0 + 2.5 * alpha * concentration


def compute_diffusivity(temperature: float, viscosity: float) -> float:
    """Stokes-Einstein inspired diffusivity."""
    return temperature / (viscosity * 1e3)


def compute_activity_coefficient(alpha: float, concentration: float) -> float:
    """Synthetic mean activity coefficient."""
    return np.exp(0.5 * alpha * concentration)


def compute_conductivity(
    alpha: float,
    temperature: float,
    concentration: float,
    permittivity: float,
    viscosity: float,
    salt_factor: float,
    coordination_number: float,
) -> float:
    """
    Synthetic conductivity model.

    Includes:
    - alpha^2 dependence
    - inverse viscosity
    - salt intrinsic factor
    - dielectric screening
    - exponential penalty for coordination
    """
    screening = np.sqrt(permittivity)
    pairing_penalty = np.exp(-0.15 * coordination_number)

    sigma = (
        alpha**2
        * salt_factor
        * concentration
        * temperature
        * screening
        * pairing_penalty
        / viscosity
    )

    noise = RNG.normal(0.0, 0.05 * sigma)
    return max(sigma + noise, 0.0)


def generate_dataset(n_samples: int) -> pd.DataFrame:
    """Generate synthetic dataset."""
    rows = []

    for _ in range(n_samples):
        salt = RNG.choice(SALTS)
        composition = sample_solvent_composition()

        concentration = RNG.uniform(0.2, 2.5)
        temperature = RNG.uniform(273.0, 330.0)
        alpha = RNG.uniform(0.75, 1.0)

        permittivity = compute_mixture_permittivity(composition)
        viscosity = compute_viscosity(concentration, temperature)
        coordination = compute_coordination_number(alpha, concentration)
        diffusivity = compute_diffusivity(temperature, viscosity)
        activity = compute_activity_coefficient(alpha, concentration)

        conductivity = compute_conductivity(
            alpha,
            temperature,
            concentration,
            permittivity,
            viscosity,
            SALT_FACTORS[salt],
            coordination,
        )

        row = {
            "salt": salt,
            "concentration_molkg": concentration,
            "temperature_K": temperature,
            "alpha": alpha,
            "permittivity": permittivity,
            "viscosity": viscosity,
            "coordination_number": coordination,
            "diffusivity": diffusivity,
            "activity_coefficient": activity,
            "conductivity_mScm": conductivity,
        }

        row.update({f"solvent_{k}": v for k, v in composition.items()})

        rows.append(row)

    return pd.DataFrame(rows)



df = generate_dataset(N_SAMPLES)
df.to_csv(OUTPUT_FILE, index=False)
print(f"Dataset written to {OUTPUT_FILE}")
print(df.head())


