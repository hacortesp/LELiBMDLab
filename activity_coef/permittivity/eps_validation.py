"""Comparison of literature and present-MD static permittivities."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


# Literature data. LiClO4 concentration is mol/kg; the others are mol/L.
literature = {
    r"LiPF$_6$ in DEC": {
        "c": np.array([0.0, 0.098, 0.29, 0.50, 1.00]),
        "eps": np.array([3.10, 4.64, 7.50, 9.2041, 11.3407]),
    },
    r"LiPF$_6$ in DMC": {
        "c": np.array([0.0, 0.098, 0.29, 0.50, 1.00]),
        "eps": np.array([3.10, 4.77, 8.308, 9.6923, 10.4615]),
    },
    "LiTFSI in DME": {
        "c": np.array(
            [0.0, 0.097, 0.2876, 0.47345, 0.6504, 0.9026]
        ),
        "eps": np.array(
            [9.8835, 11.8804, 12.7705, 12.9691, 13.3176, 13.5512]
        ),
    },
    "LiTFSI in DME corr": {
        "c": np.array(
            [0.0, 0.097, 0.2876, 0.47345, 0.6504, 0.9026]
        ),
        "eps": np.array(
            [7.20, 9.19, 10.08, 10.28, 10.63, 10.86]
        ),
    },
}


# Present-MD data.
present_md = {
    r"LiPF$_6$ in DEC": {
        "c": np.array([0.0, 0.26, 0.52, 0.78, 1.05]),
        "eps": np.array([3.10, 5.7363, 8.4943, 9.4257, 12.084]),
    },
    r"LiPF$_6$ in DMC": {
        "c": np.array([0.0, 0.26, 0.52, 0.78, 1.05]),
        "eps": np.array([3.10, 5.9645, 9.5344, 12.1692, 12.4138]),
    },
    "LiTFSI in DME": {
        "c": np.array([0.0, 0.26, 0.53, 0.81, 1.10]),
        "eps": np.array([7.20, 8.6072, 9.867, 10.5273, 10.76]),
    },
}


paper_palette = {
    "teal": "#008080",
    "orange": "#FF7F0F",
    "purple": "#9B44B3",
    "dark_gray": "#505050",
    "light_gray": "#AFABAB",
}

# Electrolyte colors.
colors = {
    r"LiPF$_6$ in DEC": paper_palette["orange"],
    r"LiPF$_6$ in DMC": paper_palette["teal"],
    "LiTFSI in DME": paper_palette["purple"],
    "LiTFSI in DME corr": paper_palette["purple"],
}

# Symbols distinguish the corrected LiTFSI literature data.
markers = {
    r"LiPF$_6$ in DEC": "o",
    r"LiPF$_6$ in DMC": "o",
    "LiTFSI in DME": "o",
    "LiTFSI in DME corr": "s",
}


def make_plot(
    output: str | Path = "static_permittivity_comparison.png",
) -> None:
    """Create and save the static-permittivity comparison."""

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 20,
            "axes.labelsize": 23,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 16,
            "legend.title_fontsize": 17,
            "axes.linewidth": 1.3,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
        }
    )

    # Same marker size for literature and present-MD datasets.
    marker_size_lit = 14.5
    marker_size_md = 12.5
    line_width = 1.5

    # Reserve horizontal space for the legends.
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    fig.subplots_adjust(
        left=0.13,
        right=0.66,
        bottom=0.19,
        top=0.95,
    )

    # Literature: filled symbols connected by continuous lines.
    for system, data in literature.items():
        ax.plot(
            data["c"],
            data["eps"],
            linestyle="-",
            linewidth=line_width,
            color=colors[system],
            marker=markers[system],
            markersize=marker_size_lit,
            markerfacecolor=colors[system],
            markeredgecolor="white",
            markeredgewidth=0.8,
            zorder=3,
        )

    # Present MD: open symbols connected by dotted lines.
    for system, data in present_md.items():
        ax.plot(
            data["c"],
            data["eps"],
            linestyle=":",
            linewidth=line_width,
            color=colors[system],
            marker=markers[system],
            markersize=marker_size_md,
            markerfacecolor="white",
            markeredgecolor=colors[system],
            markeredgewidth=2.0,
            zorder=4,
        )

    ax.set_xlim(-0.12, 1.15)
    ax.set_ylim(-1.0, 21.0)

    ax.set_xticks([0.00, 0.25, 0.50, 0.75, 1.00])
    ax.set_yticks([0, 5, 10, 15, 20])

    ax.set_xlabel(
        r"Concentration (mol/L)",
        labelpad=10,
    )
    ax.set_ylabel(
        r"$\varepsilon_r$",
        labelpad=12,
    )

    ax.grid(
        True,
        color="#dddddd",
        linestyle=":",
        linewidth=0.9,
        zorder=0,
    )

    ax.tick_params(
        length=7,
        width=1.2,
    )

    # Legend describing the data source and line style.
    source_handles = [
        Line2D(
            [],
            [],
            color="black",
            linestyle="-",
            linewidth=line_width,
            marker="o",
            markersize=marker_size_lit,
            markerfacecolor="black",
            markeredgecolor="white",
            markeredgewidth=0.8,
            label="Literature",
        ),
        Line2D(
            [],
            [],
            color="black",
            linestyle=":",
            linewidth=line_width,
            marker="o",
            markersize=marker_size_md,
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1.8,
            label="Present MD",
        ),
    ]

    source_legend = ax.legend(
        handles=source_handles,
        title="Data source",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.00),
        frameon=False,
        borderaxespad=0.0,
        handlelength=2.6,
        handletextpad=0.7,
    )

    ax.add_artist(source_legend)

    # Legend identifying each electrolyte system.
    system_handles = [
        Line2D(
            [],
            [],
            color=colors[r"LiPF$_6$ in DEC"],
            linestyle="-",
            linewidth=line_width,
            marker=markers[r"LiPF$_6$ in DEC"],
            markersize=marker_size_lit,
            markerfacecolor=colors[r"LiPF$_6$ in DEC"],
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=r"LiPF$_6$ in DEC",
        ),
        Line2D(
            [],
            [],
            color=colors[r"LiPF$_6$ in DMC"],
            linestyle="-",
            linewidth=line_width,
            marker=markers[r"LiPF$_6$ in DMC"],
            markersize=marker_size_lit,
            markerfacecolor=colors[r"LiPF$_6$ in DMC"],
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=r"LiPF$_6$ in DMC",
        ),
        Line2D(
            [],
            [],
            color=colors["LiTFSI in DME"],
            linestyle="-",
            linewidth=line_width,
            marker=markers["LiTFSI in DME"],
            markersize=marker_size_lit,
            markerfacecolor=colors["LiTFSI in DME"],
            markeredgecolor="white",
            markeredgewidth=0.8,
            label="LiTFSI in DME",
        ),
        Line2D(
            [],
            [],
            color=colors["LiTFSI in DME corr"],
            linestyle="-",
            linewidth=line_width,
            marker=markers["LiTFSI in DME corr"],
            markersize=marker_size_lit,
            markerfacecolor=colors["LiTFSI in DME corr"],
            markeredgecolor="white",
            markeredgewidth=0.8,
            label="LiTFSI in DME corr.",
        ),
    ]

    ax.legend(
        handles=system_handles,
        title="Electrolyte",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.65),
        frameon=False,
        borderaxespad=0.0,
        handlelength=2.6,
        handletextpad=0.7,
    )

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        output.with_suffix(".pdf"),
        bbox_inches="tight",
    )

    #plt.show()
    plt.close(fig)


if __name__ == "__main__":
    make_plot()