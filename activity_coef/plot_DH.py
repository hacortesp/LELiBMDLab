"""
Extended Debye–Hückel (DH) activity coefficient
Fully corrected and SI-consistent

Sources:
- Main manuscript Eq. (2) and (3)
- Supplementary Table 2: ionic radii
- Supplementary Table 6: water density (ρ = 0.9970 g/cm^3 at 298 K)
- Eq. (11): εr(c)

System:
LiCl, NaCl, KCl in water at 298 K
"""

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Physical constants (SI)
# -----------------------------
e = 1.602176634e-19        # C
eps0 = 8.8541878128e-12   # F m^-1
kB = 1.380649e-23         # J K^-1
NA = 6.02214076e23        # mol^-1
T = 298.0                 # K

# -----------------------------
# Water properties at 298 K
# -----------------------------
EPS_WATER = 78.3                  # dielectric constant
RHO_WATER = 997.0                 # kg m^-3 (Supplementary Table 6)

# -----------------------------
# εr(c) parameters + ionic size
# Ionic radii from Supplementary Table 2
# a = r+ + r-
# -----------------------------
PARAMS = {
    "LiCl": {"beta": 1.53, "lambda": 0.13, "a": (0.60 + 1.81) * 1e-10},
    "NaCl": {"beta": 1.92, "lambda": 0.10, "a": (0.95 + 1.81) * 1e-10},
    "KCl":  {"beta": 0.97, "lambda": 0.17, "a": (1.33 + 1.81) * 1e-10},
}

# -----------------------------
# Concentration range (mol kg^-1)
# -----------------------------
c = np.linspace(0.01, 4.0, 200)

# -----------------------------
# Functions
# -----------------------------
def epsilon_r(c, beta, lam):
    """Static permittivity εr(c), Eq. (11)"""
    return EPS_WATER * np.exp(-beta * np.arctan(lam * c))

def zeta(eps_r):
    """Electrostatic prefactor ζ (Eq. 3)"""
    return e**2 / (8.0 * np.pi * eps0 * eps_r * kB * T)

def kappa(c, eps_r):
    """
    Debye screening parameter κ (Eq. 3)
    Correct conversion from molality to number density
    """
    # Σ z_i^2 c_i = 2c for 1:1 electrolyte
    sum_z2_c = 2.0 * c

    return np.sqrt(
        (NA * e**2 * RHO_WATER * sum_z2_c) /
        (eps0 * eps_r * kB * T)
    )

def ln_gamma_dh(c, eps_r, a):
    """Extended Debye–Hückel activity coefficient (Eq. 2)"""
    kap = kappa(c, eps_r)
    return -(zeta(eps_r) * kap) / (1.0 + kap * a)

# -----------------------------
# Plot: two subplots
# -----------------------------
plt.figure(figsize=(10, 4))

# Left: ln gamma_DH
plt.subplot(1, 2, 1)
for salt, p in PARAMS.items():
    eps = epsilon_r(c, p["beta"], p["lambda"])
    plt.plot(c, ln_gamma_dh(c, eps, p["a"]), label=salt)

plt.xlabel("Concentration (mol kg$^{-1}$)")
plt.ylabel("ln γ$_{±}^{DH}$")
plt.legend()

# Right: epsilon_r
plt.subplot(1, 2, 2)
for salt, p in PARAMS.items():
    eps = epsilon_r(c, p["beta"], p["lambda"])
    plt.plot(c, eps, label=salt)

plt.xlabel("Concentration (mol kg$^{-1}$)")
plt.ylabel("ε$_r$")
plt.legend()

plt.tight_layout()
plt.show()



