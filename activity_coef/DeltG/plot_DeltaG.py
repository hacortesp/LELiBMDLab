
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc


FILES = [
    "delta_G_BF4.csv",
    "delta_G_ClO4.csv",
    "delta_G_FSI.csv",
    "delta_G_Li.csv",
    "delta_G_PF6.csv",
    "delta_G_TFSI.csv",
]


def configure_style() -> dict:
    rc("text", usetex=False)
    rc("font", family="serif")

    font_list = {"label": 16, "tick": 12, "legend": 14}

    color_list = [
        "#3C3846",  # dark gray
        "#DF543F",  # red-orange
        "#2286A9",  # blue
        "#FBBF7C",  # light orange
        "#6A994E",  # green
        "#8E44AD",  # purple
    ]

    return font_list, color_list


def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    if df.columns[0].startswith("Unnamed") or df.columns[0] == "":
        df = df.iloc[:, 1:]

    df.columns = [col.strip() for col in df.columns]

    return df


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


def plot_all(files: list[str]) -> None:
    font_list, color_list = configure_style()

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for i, file in enumerate(files):
        df = load_data(file)

        x = df["E_K"]
        y = df["ΔG_opt (kJ/mol)"]

        ax = axes[i]

        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2,
            markersize=5,
            color=color_list[i % len(color_list)],
        )

        ax.set_title(extract_label(file), fontsize=font_list["label"])
        ax.set_xlabel(r"$\epsilon$ (T)", fontsize=font_list["label"])
        ax.set_ylabel(r"$\Delta G_{opt}$ (kJ/mol)", fontsize=font_list["label"])

        ax.tick_params(axis="both", labelsize=font_list["tick"])
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout(pad=2.0)
    plt.savefig("deltaG.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    plot_all(FILES)