import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc


FILES = [
    "dipole_correction_factor_libf4.csv",
    "dipole_correction_factor_liclo4.csv",
    "dipole_correction_factor_lifsi.csv",
    "dipole_correction_factor_lipf6.csv",
    "dipole_correction_factor_litfsi.csv",
]


# ---------- STYLE ----------
def configure_style():
    rc("text", usetex=False)
    rc("font", family="serif")

    font_list = {"label": 16, "tick": 12, "fit": 10}

    color_map = {
        "LiBF4": "#3C3846",
        "LiClO4": "#DF543F",
        "LiFSI": "#2286A9",
        "LiPF6": "#6A994E",
        "LiTFSI": "#8E44AD",
    }

    return font_list, color_map


# ---------- DATA ----------
def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    if df.columns[0].startswith("Unnamed") or df.columns[0] == "":
        df = df.iloc[:, 1:]

    df.columns = [col.strip() for col in df.columns]
    df = df.sort_values("Dielectric_constant").reset_index(drop=True)

    return df


# ---------- LABEL ----------
def extract_system_name(filename: str) -> str:
    base = os.path.splitext(filename)[0].replace("dipole_correction_factor_", "")

    name_map = {
        "libf4": "LiBF4",
        "liclo4": "LiClO4",
        "lifsi": "LiFSI",
        "lipf6": "LiPF6",
        "litfsi": "LiTFSI",
    }

    return name_map[base.lower()]


def extract_label(system_name: str) -> str:
    latex_map = {
        "LiBF4": r"LiBF$_{4}$",
        "LiClO4": r"LiCl$_{4}$",
        "LiFSI": r"LiFSI",
        "LiPF6": r"LiP$_{6}$",
        "LiTFSI": r"LiTFSI",
    }

    return latex_map[system_name]


# ---------- FITTING PARAMETERS ----------
# xi(eps) = a - b / (eps + c)
FITTING_PARAMS = {
    "LiClO4": {
        "a": 0.9365122651,
        "b": 1.0474708536,
        "c": 3.4520340388,
    },
    "LiBF4": {
        "a": 0.9354347857,
        "b": 0.4786633693,
        "c": 1.6982050863,
    },
    "LiFSI": {
        "a": 0.9935864788,
        "b": 0.6504155227,
        "c": 0.6632177156,
    },
    "LiPF6": {
        "a": 0.9205251214,
        "b": 0.4224063422,
        "c": 0.2018108317,
    },
    "LiTFSI": {
        "a": 0.9205251214,
        "b": 0.4224063422,
        "c": 0.2018108317,
    },
}


# ---------- MODEL ----------
def correction_factor_model(eps, a, b, c):
    return a - b / (eps + c)


# ---------- METRICS ----------
def calculate_metrics(y_true, y_pred):
    residuals = y_true - y_pred

    rmse = np.sqrt(np.mean(residuals**2))
    mae = np.mean(np.abs(residuals))

    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)

    r2 = 1.0 - ss_res / ss_tot

    return r2, rmse, mae


# ---------- FORMAT ----------
def format_equation_for_print(params: dict) -> str:
    a = params["a"]
    b = params["b"]
    c = params["c"]

    return f"xi(eps) = {a:.10f} - {b:.10f} / (eps + {c:.10f})"


def format_equation_for_plot(params: dict) -> str:
    a = params["a"]
    b = params["b"]
    c = params["c"]

    return (
        rf"$\xi(\epsilon) = {a:.4f}"
        "\n"
        rf"- \frac{{{b:.4f}}}{{\epsilon + {c:.4f}}}$"
    )


# ---------- MAIN ----------
def plot_all(files):
    font_list, color_map = configure_style()

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    subplot_position = {
        "LiBF4": 0,
        "LiClO4": 1,
        "LiFSI": 2,
        "LiPF6": 4,
        "LiTFSI": 5,
    }

    for file in files:
        df = load_data(file)

        system_name = extract_system_name(file)
        label = extract_label(system_name)

        params = FITTING_PARAMS[system_name]

        x = df["Dielectric_constant"].values
        y = df["correction_factor"].values

        a = params["a"]
        b = params["b"]
        c = params["c"]

        y_pred = correction_factor_model(x, a, b, c)
        r2, rmse, mae = calculate_metrics(y, y_pred)

        # PRINT RESULTS IN TERMINAL
        print(f"{label}:")
        print(f"  Fit: {format_equation_for_print(params)}")
        print(f"  R²   = {r2:.8f}")
        print(f"  RMSE = {rmse:.8f}")
        print(f"  MAE  = {mae:.8f}")
        print("-" * 50)

        # PLOT
        ax = axes[subplot_position[system_name]]
        color = color_map[system_name]

        ax.plot(
            x,
            y,
            "o",
            color=color,
            markersize=6,
            label="Data",
        )

        x_fit = np.linspace(np.min(x), np.max(x), 500)
        y_fit = correction_factor_model(x_fit, a, b, c)

        ax.plot(
            x_fit,
            y_fit,
            "-",
            color=color,
            linewidth=2,
            label="Fit",
        )

        # FITTING INFORMATION INSIDE THE PLOT
        fit_text = (
            rf"$R^2 = {r2:.4f}$"
            + "\n"
            + rf"$\mathrm{{RMSE}} = {rmse:.4f}$"
        )

        ax.text(
            0.96,
            0.07,
            fit_text,
            transform=ax.transAxes,
            fontsize=font_list["fit"],
            ha="right",
            va="bottom",
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                edgecolor="gray",
                alpha=0.85,
            ),
        )

        ax.set_title(label, fontsize=font_list["label"])
        ax.set_xlabel(r"$\epsilon$", fontsize=font_list["label"])
        ax.set_ylabel(r"$\xi(\epsilon)$", fontsize=font_list["label"])
        ax.tick_params(axis="both", labelsize=font_list["tick"])
        ax.grid(True, linestyle="--", alpha=0.5)

    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig("dipole_correction_factor_fitted.png", dpi=300)
    plt.savefig("dipole_correction_factor_fitted.pdf")
    plt.show()


if __name__ == "__main__":
    plot_all(FILES)