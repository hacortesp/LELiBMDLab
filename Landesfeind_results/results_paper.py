import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Landesfeind & Gasteiger (2019)
# EC:EMC (3:7 w:w), Table II coefficients
# T = 20 °C = 293.15 K
# Units:
#   c      : mol/L (M)
#   T      : K
#   kappa  : mS/cm
#   D      : cm^2/s
#   TDF    : dimensionless
#   t_plus : dimensionless
# ============================================================

T = 20.0 + 273.15
c = np.linspace(0.0, 3.0, 500)

# Eq. 15: ionic conductivity κ(c, T)
# κ(c, T) = p1 * (1 + (T - p2)) * c *
#           (1 + p3*sqrt(c) + p4*(1 + p5*exp(1000/T))*c) /
#           (1 + c^4 * (p6*exp(1000/T)))  [mS/cm]
kappa_p = {
    "p1": 5.21e-01,
    "p2": 2.28e+02,
    "p3": -1.06e+00,
    "p4": 3.53e-01,
    "p5": -3.59e-03,
    "p6": 1.48e-03,
}

# Eq. 18: binary diffusion coefficient D±(c, T)
# D±(c, T) = p1 * exp(p2*c) * exp(p3/T) * exp((p4/T)*c) * 1e-6 [cm^2/s]
D_p = {
    "p1": 1.01e+03,
    "p2": 1.01e+00,
    "p3": -1.56e+03,
    "p4": -4.87e+02,
}

# Eq. 19: thermodynamic factor TDF(c, T)
# TDF(c, T) = p1 + p2*c + p3*T + p4*c^2 + p5*c*T + p6*T^2
#             + p7*c^3 + p8*c^2*T + p9*c*T^2
TDF_p = {
    "p1": 2.57e+01,
    "p2": -4.51e+01,
    "p3": -1.77e-01,
    "p4": 1.94e+00,
    "p5": 2.95e-01,
    "p6": 3.08e-04,
    "p7": 2.59e-01,
    "p8": -9.46e-03,
    "p9": -4.54e-04,
}

# Eq. 20: transference number t+(c, T)
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

def kappa(c, T, p):
    return (
        p["p1"] * (1.0 + (T - p["p2"])) * c *
        (
            1.0
            + p["p3"] * np.sqrt(c)
            + p["p4"] * (1.0 + p["p5"] * np.exp(1000.0 / T)) * c
        )
        / (1.0 + c**4 * (p["p6"] * np.exp(1000.0 / T)))
    )

def diffusion(c, T, p):
    return (
        p["p1"]
        * np.exp(p["p2"] * c)
        * np.exp(p["p3"] / T)
        * np.exp((p["p4"] / T) * c)
        * 1e-6
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

k = kappa(c, T, kappa_p)
D = diffusion(c, T, D_p)
tdf = poly3(c, T, TDF_p)
tplus = poly3(c, T, tplus_p)

fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

axs[0, 0].plot(c, k)
axs[0, 0].set_title(r"Ionic conductivity $\kappa$")
axs[0, 0].set_xlabel("c / M")
axs[0, 0].set_ylabel("mS/cm")
axs[0, 0].set_xlim(0, 3)

axs[0, 1].plot(c, D)
axs[0, 1].set_title(r"Binary diffusion coefficient $D_{\pm}$")
axs[0, 1].set_xlabel("c / M")
axs[0, 1].set_ylabel(r"cm$^2$/s")
axs[0, 1].set_xlim(0, 3)

axs[1, 0].plot(c, tdf)
axs[1, 0].set_title("Thermodynamic factor TDF")
axs[1, 0].set_xlabel("c / M")
axs[1, 0].set_ylabel("dimensionless")
axs[1, 0].set_xlim(0, 3)

axs[1, 1].plot(c, tplus)
axs[1, 1].set_title(r"Transference number $t_+$")
axs[1, 1].set_xlabel("c / M")
axs[1, 1].set_ylabel("dimensionless")
axs[1, 1].set_xlim(0, 3)

for ax in axs.flat:
    ax.grid(True, alpha=0.3)
    ax.axvline(0, linewidth=0.8, alpha=0.4)

fig.suptitle(r"LiPF$_6$ in EC:EMC (3:7 w:w), $T = 20^\circ$C", fontsize=14)

plt.savefig("ec_emc_transport_fits_20C.png", dpi=300, bbox_inches="tight")
plt.show()

print(f"T = {T:.2f} K")
print(f"kappa(c=1 M) = {kappa(np.array([1.0]), T, kappa_p)[0]:.4f} mS/cm")
print(f"D(c=1 M)     = {diffusion(np.array([1.0]), T, D_p)[0]:.4e} cm^2/s")
print(f"TDF(c=1 M)   = {poly3(np.array([1.0]), T, TDF_p)[0]:.4f}")
print(f"t+(c=1 M)    = {poly3(np.array([1.0]), T, tplus_p)[0]:.4f}")