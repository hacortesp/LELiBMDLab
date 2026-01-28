import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Physical constants (SI)
# ============================================================
e = 1.602176634e-19        # C
eps0 = 8.8541878128e-12   # F m^-1
kB = 1.380649e-23         # J K^-1
T = 298.0                 # K

# ============================================================
# Water permittivity at infinite dilution
# ============================================================
EPS_SOLV = 78.3  # εr of pure water

# ============================================================
# Concentration range (mol kg^-1)
# ============================================================
c = np.linspace(0.01, 4.0, 300)

# ============================================================
# Dielectric decrement parameters (SI Table 9)
# ε(c) = ε_solv − A c + B c^(3/2)
# ============================================================
DIELECTRIC_PARAMS = {
    "LiCl": {"A": 12.41, "B": 3.02},
    "NaCl": {"A": 20.15, "B": 5.85},
    "KCl":  {"A": 20.13, "B": 5.20},
}

# ============================================================
# Born radii (EXPERIMENTAL, SI Table 3) [m]
# ============================================================
BORN_RADII = {
    "LiCl": {"R_plus": 1.30e-10, "R_minus": 2.26e-10},
    "NaCl": {"R_plus": 1.62e-10, "R_minus": 2.26e-10},
    "KCl":  {"R_plus": 1.95e-10, "R_minus": 2.26e-10},
}

# ============================================================
# Functions
# ============================================================
def epsilon_r_poly(c, A, B):
    """Polynomial dielectric decrement (SI Table 9)"""
    return EPS_SOLV - A * c + B * c**1.5


def ln_gamma_born_single(eps_r, eps_r_0, R_i):
    """
    Single-ion Born contribution ln γ_B,i
    Main text Eq. (4), reduced free energy
    """
    prefactor = e**2 / (8.0 * np.pi * eps0 * kB * T * R_i)
    return prefactor * ((1.0 / eps_r) - (1.0 / eps_r_0))


def ln_gamma_born_mean(c, A, B, R_plus, R_minus):
    """
    Mean Born activity coefficient ln γ_B±
    Supplementary Eq. (3), 1:1 electrolyte
    """
    eps = epsilon_r_poly(c, A, B)
    eps0 = epsilon_r_poly(0.0, A, B)

    ln_g_plus = ln_gamma_born_single(eps, eps0, R_plus)
    ln_g_minus = ln_gamma_born_single(eps, eps0, R_minus)

    return 0.5 * (ln_g_plus + ln_g_minus)

# ============================================================
# Plot: ln γ_B± and εr(c)
# ============================================================
plt.figure(figsize=(10, 4))

# --- Left: ln γ_B± ---
plt.subplot(1, 2, 1)
for salt in DIELECTRIC_PARAMS:
    A = DIELECTRIC_PARAMS[salt]["A"]
    B = DIELECTRIC_PARAMS[salt]["B"]
    R_p = BORN_RADII[salt]["R_plus"]
    R_m = BORN_RADII[salt]["R_minus"]

    ln_gB = ln_gamma_born_mean(c, A, B, R_p, R_m)
    plt.plot(c, ln_gB, label=salt)

plt.xlabel("c (mol kg$^{-1}$)")
plt.ylabel("ln γ$_{±}^{B}$")
plt.legend()

# --- Right: εr(c) ---
plt.subplot(1, 2, 2)
for salt in DIELECTRIC_PARAMS:
    A = DIELECTRIC_PARAMS[salt]["A"]
    B = DIELECTRIC_PARAMS[salt]["B"]
    plt.plot(c, epsilon_r_poly(c, A, B), label=salt)

plt.xlabel("c (mol kg$^{-1}$)")
plt.ylabel("ε$_r$")
plt.legend()

plt.tight_layout()
plt.show()

