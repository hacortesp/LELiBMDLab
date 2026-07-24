import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# ============================================================
# Input
# ============================================================

INPUT_CSV = "dipole_correction_factor_lipf6.csv"


# ============================================================
# Read data
# ============================================================

df = pd.read_csv(INPUT_CSV)
df.columns = [col.strip() for col in df.columns]

df = df.sort_values("Dielectric_constant").reset_index(drop=True)

eps_data = df["Dielectric_constant"].to_numpy(dtype=float)

# Use the correction factor already stored in the CSV
xi_data = df["correction_factor"].to_numpy(dtype=float)

# Alternatively, recompute it directly:
# xi_data = (
#     df["Dipole_moment_Gaussian"].to_numpy(dtype=float)
#     / df["Dipole_moment_MD"].to_numpy(dtype=float)
# )


# ============================================================
# Rational model
# xi(eps) = xi_inf - A / (eps + B)
# ============================================================

def rational_correction_factor(
    eps: np.ndarray,
    xi_inf: float,
    A: float,
    B: float,
) -> np.ndarray:
    return xi_inf - A / (eps + B)


# ============================================================
# Initial guess and bounds
# ============================================================

eps_min = eps_data.min()

xi_inf_guess = xi_data.max() + 0.02
B_guess = 1.0
A_guess = (xi_inf_guess - xi_data.min()) * (eps_min + B_guess)

p0 = [xi_inf_guess, A_guess, B_guess]

# Keep denominator positive over the fitted range
lower_bounds = [0.0, 0.0, -eps_min + 1e-8]
upper_bounds = [2.0, np.inf, np.inf]


# ============================================================
# Fit
# ============================================================

params, covariance = curve_fit(
    rational_correction_factor,
    eps_data,
    xi_data,
    p0=p0,
    bounds=(lower_bounds, upper_bounds),
    maxfev=100000,
)

xi_inf, A, B = params

# Parameter standard errors
param_errors = np.sqrt(np.diag(covariance))
xi_inf_err, A_err, B_err = param_errors


# ============================================================
# Evaluate fit
# ============================================================

xi_fit_data = rational_correction_factor(eps_data, xi_inf, A, B)
residuals = xi_data - xi_fit_data

ss_res = np.sum(residuals**2)
ss_tot = np.sum((xi_data - np.mean(xi_data))**2)

r2 = 1.0 - ss_res / ss_tot
rmse = np.sqrt(np.mean(residuals**2))
mae = np.mean(np.abs(residuals))
max_abs_error = np.max(np.abs(residuals))


# ============================================================
# Print results
# ============================================================

print("Rational correction factor fit")
print("--------------------------------")
print("Model:")
print("xi(eps) = xi_inf - A / (eps + B)")
print()

print("Fitted parameters:")
print(f"xi_inf = {xi_inf:.12f} ± {xi_inf_err:.12f}")
print(f"A      = {A:.12f} ± {A_err:.12f}")
print(f"B      = {B:.12f} ± {B_err:.12f}")
print()

print("Fit errors:")
print(f"R²            = {r2:.8f}")
print(f"RMSE          = {rmse:.8f}")
print(f"MAE           = {mae:.8f}")
print(f"Max abs error = {max_abs_error:.8f}")
print()

print("Final equation:")
print(
    f"xi(eps) = {xi_inf:.10f} - "
    f"{A:.10f} / (eps + {B:.10f})"
)
print()


# ============================================================
# Reusable function with fitted parameters
# ============================================================

def correction_factor(eps: np.ndarray) -> np.ndarray:
    return rational_correction_factor(eps, xi_inf, A, B)


# ============================================================
# Example values
# ============================================================

print("Example values:")
for eps_value in [3.1075, 5.0, 10.0, 80.0]:
    xi_value = correction_factor(np.array([eps_value]))[0]
    print(f"correction_factor({eps_value}) = {xi_value:.12f}")


# ============================================================
# Save fit data
# ============================================================

fit_df = df.copy()
fit_df["correction_factor_fit"] = xi_fit_data
fit_df["residual"] = residuals
fit_df["abs_error"] = np.abs(residuals)

fit_df.to_csv("lipf6_rational_fit_residuals.csv", index=False)


# ============================================================
# Plot fit
# ============================================================

eps_plot = np.linspace(eps_data.min(), eps_data.max(), 1000)
xi_plot = correction_factor(eps_plot)

plt.figure(figsize=(7, 5))

plt.scatter(
    eps_data,
    xi_data,
    label="Data",
    s=35,
)

plt.plot(
    eps_plot,
    xi_plot,
    label=rf"Rational fit, $R^2={r2:.4f}$, RMSE={rmse:.4f}",
    linewidth=2,
)

plt.xlabel(r"Dielectric constant $\varepsilon_r$")
plt.ylabel(r"Correction factor $\xi$")
plt.title(r"LiClO$_4$ rational dipole correction factor")

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("lipf6_rational_correction_factor_fit.png", dpi=300)
plt.show()


# ============================================================
# Plot residuals
# ============================================================

plt.figure(figsize=(7, 4))

plt.axhline(0.0, linestyle="--", linewidth=1)
plt.scatter(
    eps_data,
    residuals,
    s=35,
)

plt.xlabel(r"Dielectric constant $\varepsilon_r$")
plt.ylabel("Residual: data - fit")
plt.title(r"LiClO$_4$ rational fit residuals")

plt.grid(True)

plt.tight_layout()
plt.savefig("lipf6_rational_fit_residuals.png", dpi=300)
plt.show()