# file: plot_deltaG_full.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.optimize import curve_fit


FILES = [
    "delta_G_BF4.csv",
    "delta_G_ClO4.csv",
    "delta_G_FSI.csv",
    "delta_G_Li.csv",
    "delta_G_PF6.csv",
    "delta_G_TFSI.csv",
]


# ---------- STYLE ----------
def configure_style():
    rc("text", usetex=False)
    rc("font", family="serif")

    font_list = {"label": 16, "tick": 12}

    color_list = [
        "#3C3846",
        "#DF543F",
        "#2286A9",
        "#FBBF7C",
        "#6A994E",
        "#8E44AD",
    ]

    return font_list, color_list


# ---------- DATA ----------
def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    if df.columns[0].startswith("Unnamed") or df.columns[0] == "":
        df = df.iloc[:, 1:]

    df.columns = [col.strip() for col in df.columns]
    return df


# ---------- LABEL ----------
def extract_label(filename: str) -> str:
    base = os.path.splitext(filename)[0].replace("delta_G_", "")

    latex_map = {
        "BF4": r"$\mathrm{BF_4^-}$",
        "ClO4": r"$\mathrm{ClO_4^-}$",
        "FSI": r"$\mathrm{FSI^-}$",
        "TFSI": r"$\mathrm{TFSI^-}$",
        "PF6": r"$\mathrm{PF_6^-}$",
        "Li": r"$\mathrm{Li^+}$",
    }

    return latex_map.get(base, rf"$\mathrm{{{base}}}$")


# ---------- MODELS ----------
def rational_model(x, a, b, c):
    return a + b / (x + c)


def exponential_model(x, a, b, c):
    return a + b * np.exp(-c * x)


# ---------- FIT ----------
def fit_best_model(x: np.ndarray, y: np.ndarray):
    models = {
        "rational": (rational_model, [min(y), -100.0, 1.0]),
        "exponential": (exponential_model, [min(y), -100.0, 0.1]),
    }

    best = {"name": None, "params": None, "func": None, "r2": -np.inf}

    for name, (model, p0) in models.items():
        try:
            popt, _ = curve_fit(
                model,
                x,
                y,
                p0=p0,
                bounds=([-np.inf, -np.inf, 0], [np.inf, np.inf, np.inf]),
                maxfev=20000,
            )

            y_pred = model(x, *popt)

            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - ss_res / ss_tot

            if r2 > best["r2"]:
                best.update({
                    "name": name,
                    "params": popt,
                    "func": model,
                    "r2": r2,
                })

        except Exception:
            continue

    return best["params"], best["func"], best["name"], best["r2"]

# ---------- FORMAT ----------
def format_equation(model_name: str, params: np.ndarray) -> str:
    a, b, c = params

    if model_name == "rational":
        return f"y = {a:.3f} + {b:.3f}/(x + {c:.3f})"

    elif model_name == "exponential":
        return f"y = {a:.3f} + {b:.3f} * exp(-{c:.3f}x)"

    return "Unknown model"


# ---------- MAIN ----------
def plot_all(files):
    font_list, color_list = configure_style()

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for i, file in enumerate(files):
        df = load_data(file)

        x = df["E_K"].values
        y = df["ΔG_opt (kJ/mol)"].values

        params, func, model_name, r2 = fit_best_model(x, y)

        label = extract_label(file)

        # PRINT RESULTS
        print(f"{label}:")
        print(f"  Model: {model_name}")
        print(f"  Fit: {format_equation(model_name, params)}")
        print(f"  R² = {r2:.6f}")
        print("-" * 40)

        # PLOT
        ax = axes[i]
        color = color_list[i]

        ax.plot(x, y, "o", color=color)

        x_fit = np.linspace(min(x), max(x), 300)
        y_fit = func(x_fit, *params)

        ax.plot(x_fit, y_fit, "-", color=color, linewidth=2)

        ax.set_title(label, fontsize=font_list["label"])
        ax.set_xlabel(r"$\epsilon$ (T)", fontsize=font_list["label"])
        ax.set_ylabel(r"$\Delta G_{\mathrm{opt}}$ (kJ/mol)", fontsize=font_list["label"])
        ax.tick_params(axis="both", labelsize=font_list["tick"])
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("deltaG_fitted.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    plot_all(FILES)

    
    """
    df = load_data("delta_G_ClO4.csv")

    x = df["E_K"].values
    y = df["ΔG_opt (kJ/mol)"].values
    for xi, yi in zip(x, y):
        print(f"{xi:.3f}    {yi:.3f}")
        """