import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Physical constants (SI)
# ============================================================
e = 1.602176634e-19
eps0 = 8.8541878128e-12
kB = 1.380649e-23
NA = 6.02214076e23
T = 298.0

# ============================================================
# Water properties
# ============================================================
EPS_WATER = 78.3
RHO_WATER = 997.0  # kg m^-3

# ============================================================
# Concentration range (mol kg^-1)
# ============================================================
c = np.linspace(0.01, 4.0, 300)

# ============================================================
# PARAMETERS
# ============================================================

# --- DH dielectric (Eq. 11) + ionic size (SI Table 2)
DH_PARAMS = {
    "LiCl": {"beta": 1.53, "lambda": 0.13, "a": (0.60 + 1.81) * 1e-10},
    "NaCl": {"beta": 1.92, "lambda": 0.10, "a": (0.95 + 1.81) * 1e-10},
    "KCl":  {"beta": 0.97, "lambda": 0.17, "a": (1.33 + 1.81) * 1e-10},
}

# --- Born dielectric (SI Table 9)
BORN_DIELECTRIC = {
    "LiCl": {"A": 12.41, "B": 3.02},
    "NaCl": {"A": 20.15, "B": 5.85},
    "KCl":  {"A": 20.13, "B": 5.20},
}

# --- Born radii (EXPERIMENTAL, SI Table 3)
BORN_RADII = {
    "LiCl": {"R_plus": 1.30e-10, "R_minus": 2.26e-10},
    "NaCl": {"R_plus": 1.62e-10, "R_minus": 2.26e-10},
    "KCl":  {"R_plus": 1.95e-10, "R_minus": 2.26e-10},
}

# ============================================================
# FUNCTIONS
# ============================================================

# ---------- DH part ----------
def epsilon_r_dh(c, beta, lam):
    return EPS_WATER * np.exp(-beta * np.arctan(lam * c))

def zeta(eps_r):
    return e**2 / (8.0 * np.pi * eps0 * eps_r * kB * T)

def kappa(c, eps_r):
    return np.sqrt(
        (NA * e**2 * RHO_WATER * 2.0 * c) /
        (eps0 * eps_r * kB * T)
    )

def ln_gamma_dh(c, eps_r, a):
    kap = kappa(c, eps_r)
    return -(zeta(eps_r) * kap) / (1.0 + kap * a)


# ---------- Born part ----------
def epsilon_r_poly(c, A, B):
    return EPS_WATER - A * c + B * c**1.5

def ln_gamma_born_single(eps_r, eps_r_0, R_i):
    pref = e**2 / (8.0 * np.pi * eps0 * kB * T * R_i)
    return pref * ((1.0 / eps_r) - (1.0 / eps_r_0))

def ln_gamma_born_mean(c, A, B, R_plus, R_minus):
    eps = epsilon_r_poly(c, A, B)
    eps0 = epsilon_r_poly(0.0, A, B)

    ln_g_p = ln_gamma_born_single(eps, eps0, R_plus)
    ln_g_m = ln_gamma_born_single(eps, eps0, R_minus)

    return 0.5 * (ln_g_p + ln_g_m)

# ============================================================
# COMPUTE & PLOT ln γ_{±}^{DH+B}
# ============================================================
plt.figure(figsize=(6, 4))

for salt in DH_PARAMS:
    # DH contribution
    eps_dh = epsilon_r_dh(c, DH_PARAMS[salt]["beta"], DH_PARAMS[salt]["lambda"])
    ln_dh = ln_gamma_dh(c, eps_dh, DH_PARAMS[salt]["a"])

    # Born contribution
    A = BORN_DIELECTRIC[salt]["A"]
    B = BORN_DIELECTRIC[salt]["B"]
    R_p = BORN_RADII[salt]["R_plus"]
    R_m = BORN_RADII[salt]["R_minus"]

    ln_b = ln_gamma_born_mean(c, A, B, R_p, R_m)

    # Total
    plt.plot(c, ln_dh + ln_b, label=salt)

plt.xlabel("c (mol kg$^{-1}$)")
plt.ylabel("ln γ$_{±}^{DH+B}$")
plt.legend()
plt.tight_layout()
plt.show()
