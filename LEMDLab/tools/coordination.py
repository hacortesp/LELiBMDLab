import os

import logging
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import MDAnalysis as mda
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from collections import deque
from rdkit.Geometry import Point3D
from MDAnalysis.analysis import rdf
from sklearn.cluster import DBSCAN
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from matplotlib import rc

from matplotlib.colors import LinearSegmentedColormap
from concurrent.futures import ProcessPoolExecutor, as_completed


warnings.filterwarnings("ignore", category=UserWarning, module='MDAnalysis.coordinates.PDB')
logging.getLogger('MDAnalysis').setLevel(logging.WARNING)


def calc_rdf_coord(group1, group2, v, nbins=200, range_rdf=(0.0, 10.0)):
    # Initialize RDF analysis
    rdf_analysis = rdf.InterRDF(group1, group2, nbins=nbins, range=range_rdf)
    rdf_analysis.run()

    # Calculate coordination numbers
    rho = group2.n_atoms / v  # Density of the second group
    bins = rdf_analysis.results.bins
    rdf_values = rdf_analysis.results.rdf
    coord_numbers = np.cumsum(4 * np.pi * bins**2 * rdf_values * np.diff(np.append(0, bins)) * rho)

    return bins, rdf_values, coord_numbers

def obtain_rdf_coord(bins, rdf, coord_numbers):
    deriv_sign_changes = np.diff(np.sign(np.diff(rdf)))
    peak_index = np.where(deriv_sign_changes < 0)[0] + 1

    if len(peak_index) == 0:
        raise ValueError("No peak found in RDF data.")

    # Reject noise peaks by RDF height
    peak_ptr = 0
    while peak_ptr < len(peak_index) and rdf[peak_index[peak_ptr]] < 0.1:
        peak_ptr += 1

    if peak_ptr >= len(peak_index):
        raise ValueError("No physical RDF peak found (all peaks below threshold).")

    first_peak_index = peak_index[peak_ptr]

    min_after_peak_index = np.where(deriv_sign_changes[first_peak_index:] > 0)[0] + first_peak_index + 1

    if len(min_after_peak_index) == 0:
        raise ValueError("No minimum found after the first peak in RDF data.")

    first_min_index = min_after_peak_index[0]

    x_val = round(float(bins[first_min_index]), 3)
    y_coord = round(float(np.interp(x_val, bins, coord_numbers)), 3)

    return x_val, y_coord


def plot_rdf_coordination(
    bins,
    rdf,
    coord_numbers,
    output_dir,
    filename,
):
    rc("text", usetex=False)
    rc("font", family="serif")
    font_list = {"label": 16, "ticket": 12, "legend": 14}
    color_list = ["#3C3846", "#DF543F", "#2286A9", "#FBBF7C"]

    coor_dir = os.path.join(output_dir, "coordination_files")
    os.makedirs(coor_dir, exist_ok=True)  # ensure output exists

    output_path = os.path.join(coor_dir, filename)

    fig, ax1 = plt.subplots()
    fig.set_size_inches(5.5, 4)

    ax1.plot(
        bins,
        rdf,
        "-",
        linewidth=1.5,
        color=color_list[0],
        label=r"g(r)",
    )
    ax1.set_xlabel(r"Distance (Å)", fontsize=font_list["label"])
    ax1.set_ylabel(r"g(r)", fontsize=font_list["label"])
    ax1.tick_params(
        axis="both",
        which="both",
        direction="in",
        labelsize=font_list["ticket"],
    )

    ax2 = ax1.twinx()
    ax2.plot(
        bins,
        coord_numbers,
        "--",
        linewidth=2,
        color="grey",
        label=r"Coord. Number",
    )
    ax2.set_ylabel(r"Coordination Number", fontsize=font_list["label"])
    ax2.tick_params(
        axis="y",
        which="both",
        direction="in",
        labelsize=font_list["ticket"],
    )

    ax1.set_xlim(0, 10)
    ax1.grid(True, linestyle="--")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)  # critical in batch runs

def select_shell(
        select,
        distance,
        center_atom,
        kw
):
    if isinstance(select, dict):
        species_selection = select[kw]
        if species_selection is None:
            raise ValueError("Species specified does not match entries in the select dict.")
    else:
        species_selection = select
    if isinstance(distance, dict):
        distance_value = distance[kw]
        if distance_value is None:
            raise ValueError("Species specified does not match entries in the distance dict.")
        distance_str = str(distance_value)
    else:
        distance_str = distance
    return "(" + species_selection + ") and (around " + distance_str + " index " + str(center_atom.index) + ")"


def analyze_coordination_structure(
    run: mda.Universe,
    run_start: int,
    run_end: int,
    select_dict: dict[str, str],
    distance: float,
    center_atom: str = "cation",
    counter_atom: str = "anion",
    plot_path: str = None,
) -> pd.DataFrame:

    def select_shell(select, distance, center_atom, kw):
        if isinstance(select, dict):
            species_selection = select[kw]
            if species_selection is None:
                raise ValueError("Species specified does not match entries in the select dict.")
        else:
            species_selection = select
        if isinstance(distance, dict):
            distance_value = distance[kw]
            if distance_value is None:
                raise ValueError("Species specified does not match entries in the distance dict.")
            distance_str = str(distance_value)
        else:
            distance_str = distance
        return "(" + species_selection + ") and (around " + distance_str + " index " + str(center_atom.index) + ")"

    def num_of_neighbor_simple(nvt_run, center_atom, distance_dict, select_dict, run_start, run_end):
        trj_analysis = nvt_run.trajectory[run_start:run_end:]
        species = next(iter(distance_dict.keys()))
        cn_values = np.zeros(int(len(trj_analysis)))
        for time_count, _ts in enumerate(trj_analysis):
            selection = select_shell(select_dict, distance_dict, center_atom, species)
            shell = nvt_run.select_atoms(selection, periodic=True)
            shell_molecules = shell.residues
            shell_len = len(shell_molecules)
            if shell_len == 0:
                cn_values[time_count] = 1
            elif shell_len == 1:
                coordination_atoms = shell_molecules.atoms.select_atoms("same type as index " + str(shell.atoms[0].index))
                unique_lithium_indices = set()
                for atom in coordination_atoms:
                    selection_species = select_shell("same type as index " + str(center_atom.index), distance_dict, atom, species)
                    shell_species = nvt_run.select_atoms(selection_species, periodic=True)
                    for li_atom in shell_species:
                        unique_lithium_indices.add(li_atom.index)
                shell_species_len = len(unique_lithium_indices) - 1
                if shell_species_len == 0:
                    cn_values[time_count] = 2
                else:
                    cn_values[time_count] = 3
            else:
                cn_values[time_count] = 3
        return {"total": cn_values}

    def concat_coord_array(nvt_run, func, center_atoms, distance_dict, select_dict, run_start, run_end):
        num_array = func(nvt_run, center_atoms[0], distance_dict, select_dict, run_start, run_end)
        for atom in tqdm(center_atoms[1::]):
            this_atom = func(nvt_run, atom, distance_dict, select_dict, run_start, run_end)
            for kw in num_array:
                num_array[kw] = np.concatenate((num_array.get(kw), this_atom.get(kw)), axis=0)
        return num_array

    distance_dict = {counter_atom: distance}
    center_atoms = run.select_atoms(select_dict.get(center_atom))
    num_array = concat_coord_array(
        run,
        num_of_neighbor_simple,
        center_atoms,
        distance_dict,
        select_dict,
        run_start,
        run_end,
    )["total"]

    shell_component, shell_count = np.unique(num_array.flatten(), return_counts=True)
    combined = np.vstack((shell_component, shell_count)).T
    item_dict = {"1": "ssip", "2": "cip", "3": "agg"}
    item_list = []
    percent_list = []
    for i in range(len(combined)):
        item = str(int(combined[i, 0]))
        item_list.append(item_dict.get(item))
        percent_list.append(f"{(combined[i, 1] / combined[:, 1].sum() * 100):.4f}%")
    df_dict = {"Solvation structure": item_list, "Percentage": percent_list}

    
    order = ["ssip", "cip", "agg"]
    perc_map = {k: float(v.strip("%")) for k, v in zip(item_list, percent_list)}
    plot_vals = [perc_map.get(k, 0.0) for k in order]

    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    bars = ax.bar(order, plot_vals,  width=0.35, edgecolor="black", linewidth=0.50)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Percentage (%)")
    ax.grid(axis="y", linestyle=":", alpha=0.35)

    # Annotate the top of each bar
    for rect, v in zip(bars, plot_vals):
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            rect.get_height() + 0.6,
            f"{v:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return pd.DataFrame(df_dict)


def compute_periodic_distance_matrix(
    positions: np.ndarray,
    box_length: float,
) -> np.ndarray:
    """
    Compute full pairwise distance matrix under periodic boundary conditions
    without relying on a helper function.
    """
    n = positions.shape[0]
    dist_matrix = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        delta = positions[i] - positions
        delta -= box_length * np.round(delta / box_length)
        dist_matrix[i] = np.linalg.norm(delta, axis=1)

    return dist_matrix


def calc_population_frame(ts, cations, anions, index):
    all_atoms = cations + anions
    print("cations:", cations, "\nanions:", anions.types[0])
    positions = all_atoms.positions
    box_size = ts.dimensions[0]

    dist_matrix = compute_periodic_distance_matrix(
        positions=positions,
        box_length=box_size,
    )

    model = DBSCAN(
        eps=2.0,
        min_samples=1,        # preserves original clustering behavior
        metric="precomputed",
        n_jobs=-1,
    )

    labels = model.fit_predict(dist_matrix)

    all_clusters = []
    for lbl in np.unique(labels):
        cluster = np.where(labels == lbl)[0].tolist()
        all_clusters.append(cluster)

    type_id = cations.types[0]
    type_id3 = anions.types[0]
    pop_matrix = np.zeros((50, 50, 1))

    for cluster in all_clusters:
        cations_count = 0
        anions_count = 0

        for atom_id in cluster:
            if all_atoms[atom_id].type == type_id:
                cations_count += 1
            if all_atoms[atom_id].type == type_id3:
                anions_count += 1

        if cations_count < 50 and anions_count < 50:
            pop_matrix[cations_count][anions_count] += 1

    return pop_matrix, index


def calc_population_parallel(run, run_start, run_end, select_cations, select_anions, core, plot):
    stacked_population = np.array([])

    with ProcessPoolExecutor(max_workers=core) as executor:
        futures = []
        for idx, ts in enumerate(run.trajectory[run_start:run_end]):
            cations = run.select_atoms(select_cations)
            anions = run.select_atoms(select_anions)
            futures.append(executor.submit(calc_population_frame, ts, cations, anions, idx))

        results = [future.result() for future in
                   tqdm(as_completed(futures), total=len(futures), desc='Processing trajectory')]

    # Sort results by index and combine
    sorted_results = sorted(results, key=lambda x: x[1])
    for current_population, _ in sorted_results:
        if stacked_population.size == 0:
            stacked_population = current_population
        else:
            stacked_population = np.dstack((stacked_population, current_population))

    avg_population = np.mean(stacked_population, axis=2)
    np.savetxt('avg_population.txt', avg_population, fmt='%.6f')

    if plot:
        plot_population_heatmap(avg_population)

    return 'avg_population.txt'

def plot_population_heatmap(avg_population):
    rc("text", usetex=False)
    rc("font", family="serif")

    n = min(21, avg_population.shape[0], avg_population.shape[1])
    matrix = avg_population[:n, :n].T
    matrix = np.flipud(matrix)

    mat_plot = matrix.copy()
    mat_plot[mat_plot <= 0] = 1e-12

    # --- CHANGE 1: colormap ---
    colors = ["#f8f9fb", "#406179", "#004474"]  # dark blue → blue → yellow
    cmap = LinearSegmentedColormap.from_list("custom_blue_yellow", colors)
    norm = LogNorm(vmin=1e-5, vmax=1e2)

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        mat_plot,
        annot=False,
        cmap=cmap,
        square=True,
        linewidths=0.5,
        linecolor="white",
        norm=norm,
        cbar=True
    )

    # --- CHANGE 2: colorbar formatting ---
    cbar = ax.collections[0].colorbar
    cbar.ax.minorticks_off()
    cbar.set_ticks([10**i for i in range(-3, 3)])
    cbar.set_ticklabels([
        r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$",
        r"$10^{0}$", r"$10^{1}$", r"$10^{2}$"
    ])
    cbar.ax.tick_params(labelsize=20)  # ↓ smaller colorbar numbers

    # --- CHANGE 3: axis ticks & labels ---
    x_labels = list(range(0, n))
    y_labels = list(range(n - 1, -1, -1))

    ax.set_xticks(np.arange(0.5, n, 4))
    ax.set_yticks(np.arange(0.5, n, 4))
    ax.set_xticklabels(x_labels[0::4], fontsize=20)
    ax.set_yticklabels(y_labels[0::4], fontsize=20)
    ax.tick_params(length=0)

    # --- frame ---
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_edgecolor("black")

    # --- grid rectangles ---
    for i in range(n):
        for j in range(n):
            ax.add_patch(
                Rectangle(
                    (j + 0.05, i + 0.05),
                    0.9, 0.9,
                    fill=False,
                    edgecolor="#000000",
                    lw=0.5
                )
            )

    ax.plot([0, n], [n, 0], lw=1, color="#002845")

    # --- CHANGE 4: axis label size ---
    ax.set_xlabel(r"$n+$", fontsize=26, labelpad=20)
    ax.set_ylabel(r"$n-$", fontsize=26, labelpad=20)

    plt.tight_layout()
    plt.savefig("clusters.png", dpi=300, bbox_inches="tight")
    plt.close()


def minimum_image_displacement(
    x0: np.ndarray,
    x1: np.ndarray,
    box_length: np.ndarray | float,
) -> np.ndarray:
    """Return the displacement vector under periodic boundary conditions.

    Parameters
    ----------
    x0, x1
        Arrays containing the reference coordinates. Broadcasting between the
        two operands is supported, matching the behaviour of ``numpy``
        arithmetic.
    box_length
        Simulation box lengths. Either a scalar (cubic box) or an array-like
        object with three components describing the orthogonal box lengths.

    Returns
    -------
    numpy.ndarray
        The displacement vectors taking the minimum image convention into
        account.
    """

    delta = np.asarray(x1) - np.asarray(x0)
    box = np.asarray(box_length)

    # ``ts.dimensions`` may provide 6 values (length + angles).  Only the
    # translational components are relevant for the minimum image convention.
    if box.ndim > 0 and box.shape[-1] == 6:
        box = box[..., :3]

    half_box = 0.5 * box
    delta = np.where(delta > half_box, delta - box, delta)
    delta = np.where(delta < -half_box, delta + box, delta)
    return delta


