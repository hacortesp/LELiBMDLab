import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# ============================================================
# Input file
# ============================================================

INPUT_CSV = "dipole_correction_factor_liclo4.csv"


# ============================================================
# Read data
# ============================================================

df = pd.read_csv(INPUT_CSV)
df.columns = [col.strip() for col in df.columns]

df = df.sort_values("Dielectric_constant")

eps_data = df["Dielectric_constant"].to_numpy(dtype=float)

# Use the existing correction_factor column
correction_data = df["correction_factor"].to_numpy(dtype=float)

# Alternatively, recompute it directly:
# correction_data = (
#     df["Dipole_moment_Gaussian"].to_numpy(dtype=float)
#     / df["Dipole_moment_MD"].to_numpy(dtype=float)
# )


# ============================================================
# Fit functions
# ============================================================

def rational_fit(eps: np.ndarray, xi_inf: float, A: float, B: float) -> np.ndarray:
    """
    Saturating rational function.

    xi(eps) = xi_inf - A / (eps + B)
    """
    return xi_inf - A / (eps + B)


def exp_fit(eps: np.ndarray, xi_inf: float, A: float, k: float) -> np.ndarray:
    """
    Exponential saturation.

    xi(eps) = xi_inf - A * exp(-k * eps)
    """
    return xi_inf - A * np.exp(-k * eps)


def log_fit(eps: np.ndarray, A: float, B: float) -> np.ndarray:
    """
    Logarithmic growth.

    xi(eps) = A + B * log(eps)
    """
    return A + B * np.log(eps)


# ============================================================
# Fit models
# ============================================================

# Rational fit initial guess
p0_rational = [
    correction_data.max() + 0.02,  # xi_inf
    1.0,                           # A
    1.0,                           # B
]

params_rational, _ = curve_fit(
    rational_fit,
    eps_data,
    correction_data,
    p0=p0_rational,
    maxfev=10000,
)

# Exponential fit initial guess
p0_exp = [
    correction_data.max() + 0.02,                 # xi_inf
    correction_data.max() - correction_data.min(), # A
    0.1,                                          # k
]

params_exp, _ = curve_fit(
    exp_fit,
    eps_data,
    correction_data,
    p0=p0_exp,
    maxfev=10000,
)

# Logarithmic fit
params_log, _ = curve_fit(
    log_fit,
    eps_data,
    correction_data,
    maxfev=10000,
)


# ============================================================
# Metrics
# ============================================================

def fit_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    r2 = 1.0 - ss_res / ss_tot
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    return r2, rmse


pred_rational = rational_fit(eps_data, *params_rational)
pred_exp = exp_fit(eps_data, *params_exp)
pred_log = log_fit(eps_data, *params_log)

r2_rational, rmse_rational = fit_metrics(correction_data, pred_rational)
r2_exp, rmse_exp = fit_metrics(correction_data, pred_exp)
r2_log, rmse_log = fit_metrics(correction_data, pred_log)


# ============================================================
# Print fitted equations
# ============================================================

xi_inf, A_rat, B_rat = params_rational
xi_inf_exp, A_exp, k_exp = params_exp
A_log, B_log = params_log

print("Rational fit:")
print(
    f"correction_factor(eps) = "
    f"{xi_inf:.10f} - {A_rat:.10f} / (eps + {B_rat:.10f})"
)
print(f"R²   = {r2_rational:.8f}")
print(f"RMSE = {rmse_rational:.8f}")

print("\nExponential fit:")
print(
    f"correction_factor(eps) = "
    f"{xi_inf_exp:.10f} - {A_exp:.10f} * exp(-{k_exp:.10f} * eps)"
)
print(f"R²   = {r2_exp:.8f}")
print(f"RMSE = {rmse_exp:.8f}")

print("\nLogarithmic fit:")
print(
    f"correction_factor(eps) = "
    f"{A_log:.10f} + {B_log:.10f} * log(eps)"
)
print(f"R²   = {r2_log:.8f}")
print(f"RMSE = {rmse_log:.8f}")


# ============================================================
# Choose best model by RMSE
# ============================================================

models = {
    "rational": {
        "rmse": rmse_rational,
        "r2": r2_rational,
        "params": params_rational,
    },
    "exponential": {
        "rmse": rmse_exp,
        "r2": r2_exp,
        "params": params_exp,
    },
    "logarithmic": {
        "rmse": rmse_log,
        "r2": r2_log,
        "params": params_log,
    },
}

best_model = min(models, key=lambda name: models[name]["rmse"])

print("\nBest model by RMSE:", best_model)
print("Best RMSE:", models[best_model]["rmse"])
print("Best R²:", models[best_model]["r2"])


# ============================================================
# Reusable correction factor function
# ============================================================

def correction_factor(eps: np.ndarray) -> np.ndarray:
    """
    Default selected model.
    I recommend starting with the rational fit.
    """
    return rational_fit(eps, *params_rational)


# ============================================================
# Example values
# ============================================================

print("\nExample values using rational fit:")
print("correction_factor(3.1075) =", correction_factor(np.array([3.1075]))[0])
print("correction_factor(5.0)    =", correction_factor(np.array([5.0]))[0])
print("correction_factor(10.0)   =", correction_factor(np.array([10.0]))[0])
print("correction_factor(80.0)   =", correction_factor(np.array([80.0]))[0])


# ============================================================
# Plot
# ============================================================

eps = np.linspace(eps_data.min(), eps_data.max(), 1000)

corr_rational = rational_fit(eps, *params_rational)
corr_exp = exp_fit(eps, *params_exp)
corr_log = log_fit(eps, *params_log)

plt.figure(figsize=(7, 5))

plt.scatter(
    eps_data,
    correction_data,
    label="Data",
    s=30,
)

plt.plot(
    eps,
    corr_rational,
    label=rf"Rational fit, $R^2={r2_rational:.4f}$",
    linewidth=2,
)

plt.plot(
    eps,
    corr_exp,
    "--",
    label=rf"Exponential fit, $R^2={r2_exp:.4f}$",
    linewidth=2,
)

plt.plot(
    eps,
    corr_log,
    ":",
    label=rf"Log fit, $R^2={r2_log:.4f}$",
    linewidth=2,
)

plt.xlabel(r"Dielectric constant $\varepsilon_r$")
plt.ylabel(r"Correction factor $\xi$")
plt.title(r"LiClO$_4$ dipole correction factor")

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("liclo4_correction_factor_fit_comparison.png", dpi=300)
plt.show()