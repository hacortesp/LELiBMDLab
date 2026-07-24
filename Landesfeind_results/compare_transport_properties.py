# file: compare_transport_properties.py

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Landesfeind & Gasteiger (2019)
# LiPF6 in EC:EMC (3:7 w:w)
# Focus:
#   - Ionic conductivity kappa
#   - Transference number t_plus
# ============================================================

T = 20.0 + 273.15

# Continuous concentration range for model
c = np.linspace(0.0, 4.0, 500)

# ============================================================
# Fit parameters
# ============================================================

kappa_p = {
    "p1": 5.21e-01,
    "p2": 2.28e+02,
    "p3": -1.06e+00,
    "p4": 3.53e-01,
    "p5": -3.59e-03,
    "p6": 1.48e-03,
}

tplus_p = {
    "p1": -1.28e+01,
    "p2": -6.12e+00,
    "p3": 8.21e-02,
    "p4": 9.04e-01,
    "p5": 3.18e-02,
    "p6": -1.27e-04,
    "p7": 1.75e-02,
    "p8": -3.12e-03,
    "p9": -3.96e-05,
}

# ============================================================
# Model equations
# ============================================================

def kappa(c, T, p):
    return (
        p["p1"]
        * (1.0 + (T - p["p2"]))
        * c
        * (
            1.0
            + p["p3"] * np.sqrt(c)
            + p["p4"] * (1.0 + p["p5"] * np.exp(1000.0 / T)) * c
        )
        / (
            1.0
            + c**4 * (p["p6"] * np.exp(1000.0 / T))
        )
    )


def poly3(c, T, p):
    return (
        p["p1"]
        + p["p2"] * c
        + p["p3"] * T
        + p["p4"] * c**2
        + p["p5"] * c * T
        + p["p6"] * T**2
        + p["p7"] * c**3
        + p["p8"] * c**2 * T
        + p["p9"] * c * T**2
    )

# ============================================================
# Model predictions
# ============================================================

k_model = kappa(c, T, kappa_p)
tplus_model = poly3(c, T, tplus_p)

# ============================================================
# Load MD simulation data
# ============================================================

# Expected format:
# concentration   value   error

k_data = np.loadtxt("set1_k.dat")
tplus_data = np.loadtxt("set1_tplus.dat")

# Conductivity
k_conc = k_data[:, 0]
k_md = k_data[:, 1]
k_err = k_data[:, 2]

# Transference number
tplus_conc = tplus_data[:, 0]
tplus_md = tplus_data[:, 1]
tplus_err = tplus_data[:, 2]

# ============================================================
# Plot
# ============================================================

fig, axs = plt.subplots(
    1,
    2,
    figsize=(12, 5),
    constrained_layout=True
)

# ------------------------------------------------------------
# Conductivity
# ------------------------------------------------------------

axs[0].plot(
    c,
    k_model,
    linewidth=2,
    label="Landesfeind model"
)

axs[0].errorbar(
    k_conc,
    k_md,
    yerr=k_err,
    fmt="o",
    capsize=4,
    label="MD simulation"
)

axs[0].set_title(r"Ionic conductivity $\kappa$")
axs[0].set_xlabel("Concentration")
axs[0].set_ylabel("mS/cm")
axs[0].set_xlim(0, 4.2)
axs[0].grid(True, alpha=0.3)
axs[0].legend()

# ------------------------------------------------------------
# Transference number
# ------------------------------------------------------------

axs[1].plot(
    c,
    tplus_model,
    linewidth=2,
    label="Landesfeind model"
)

axs[1].errorbar(
    tplus_conc,
    tplus_md,
    yerr=tplus_err,
    fmt="o",
    capsize=4,
    label="MD simulation"
)

axs[1].set_title(r"Transference number $t_+$")
axs[1].set_xlabel("Concentration")
axs[1].set_ylabel("dimensionless")
axs[1].set_xlim(0, 4.2)
axs[1].grid(True, alpha=0.3)
axs[1].legend()

# ============================================================
# Figure title and save
# ============================================================

fig.suptitle(
    r"LiPF$_6$ in EC:EMC (3:7 w:w), $T = 20^\circ$C",
    fontsize=14
)

plt.savefig(
    "compare_kappa_tplus_md_vs_model.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# Reference values
# ============================================================

print(f"T = {T:.2f} K")

print(
    f"kappa(c=1 M) = "
    f"{kappa(np.array([1.0]), T, kappa_p)[0]:.4f} mS/cm"
)

print(
    f"t+(c=1 M) = "
    f"{poly3(np.array([1.0]), T, tplus_p)[0]:.4f}"
)