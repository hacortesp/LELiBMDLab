import os

import logging
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import MDAnalysis as mda
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from MDAnalysis.analysis import rdf
from sklearn.cluster import DBSCAN
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from matplotlib import rc

from scipy.signal import find_peaks
from matplotlib.colors import LinearSegmentedColormap
from concurrent.futures import ProcessPoolExecutor, as_completed


warnings.filterwarnings("ignore", category=UserWarning, module='MDAnalysis.coordinates.PDB')
logging.getLogger('MDAnalysis').setLevel(logging.WARNING)


def calc_rdf_coord(group1, group2, v, nbins=200, range_rdf=(0.0, 10.0)):
    universe = group1.universe
    n_frames = len(universe.trajectory)
    start_frame = max(0, n_frames - 5000)
    
    # Initialize RDF analysis
    rdf_analysis = rdf.InterRDF(group1, group2, nbins=nbins, range=range_rdf)
    rdf_analysis.run(
        start=start_frame,
        stop=n_frames,
    )

    # Calculate coordination numbers
    rho = group2.n_atoms / v  # Density of the second group
    bins = rdf_analysis.results.bins
    rdf_values = rdf_analysis.results.rdf
    coord_numbers = np.cumsum(4 * np.pi * bins**2 * rdf_values * np.diff(np.append(0, bins)) * rho)

    return bins, rdf_values, coord_numbers

def obtain_rdf_coord(
    bins,
    rdf,
    coord_numbers,
    prominence_threshold=0.5,
):

    # Find physically significant RDF peaks
    peak_indices, properties = find_peaks(
        rdf,
        prominence=prominence_threshold,
    )

    if len(peak_indices) == 0:
        raise ValueError(
            "No physical RDF peak found "
            f"(prominence < {prominence_threshold})."
        )

    for i, peak_idx in enumerate(peak_indices):
        prominence = properties["prominences"][i]
        right_base_idx = properties["right_bases"][i]
        """
        print(
            f"Peak at {bins[peak_idx]:.3f} Å, "
            f"g(r)={rdf[peak_idx]:.3f}, "
            f"prominence={prominence:.3f}, "
            f"right minimum at "
            f"{bins[right_base_idx]:.3f} Å, "
            f"g(r)={rdf[right_base_idx]:.3f}"
        )
        """

    # First physically meaningful peak
    first_peak_index = peak_indices[0]

    # Position of the first RDF peak
    peak_position = round(
        float(bins[first_peak_index]),
        3,
    )

    # Minimum on the right-hand side that defines
    # the prominence of this peak
    first_min_index = properties["right_bases"][0]
    """
    print(
        f"Selected first peak: "
        f"r={bins[first_peak_index]:.3f} Å, "
        f"g(r)={rdf[first_peak_index]:.3f}"
    )

    print(
        f"Selected first minimum: "
        f"r={bins[first_min_index]:.3f} Å, "
        f"g(r)={rdf[first_min_index]:.3f}"
    )
    """

    # Coordination-shell cutoff
    x_val = round(
        float(bins[first_min_index]),
        3,
    )

    # Coordination number evaluated at the minimum,
    # NOT at the peak
    y_coord = round(
        float(
            np.interp(
                x_val,
                bins,
                coord_numbers,
            )
        ),
        3,
    )

    return x_val, y_coord, peak_position


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
        for atom in center_atoms[1::]:
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
    labels = ["SSIP", "CIP", "AGG"]
    perc_map = {k: float(v.strip("%")) for k, v in zip(item_list, percent_list)}
    plot_vals = [perc_map.get(k, 0.0) for k in order]
    # Match styling used in plot_population_heatmap()
    rc("text", usetex=False)
    rc("font", family="serif")

    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    blue = "#004474"
    bars = ax.bar(labels, plot_vals,  width=0.35, color=blue, edgecolor="black", linewidth=0.50)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    # Larger x- and y-axis numbers/text
    ax.tick_params(
        axis="x",
        labelsize=12,
        length=0
    )
    ax.tick_params(
        axis="y",
        labelsize=12
    )

    ax.grid(axis="y", linestyle=":", alpha=0.35)

    # Annotate the top of each bar
    for rect, v in zip(bars, plot_vals):
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            rect.get_height() + 0.6,
            f"{v:.1f}%",
            ha="center",
            va="bottom",
            fontsize=12,
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

def calc_population_frame_data(positions, box, types, index, r_cut):
    dist_matrix = compute_periodic_distance_matrix(
        positions=positions,
        box_length=box,
    )

    model = DBSCAN(
        eps=r_cut,
        min_samples=1,
        metric="precomputed",
    )

    labels = model.fit_predict(dist_matrix)

    pop_matrix = np.zeros((50, 50, 1))
    cation_type = types[0]
    anion_type = np.unique(types[types != cation_type])[0]

    for lbl in np.unique(labels):
        cluster = np.where(labels == lbl)[0]

        c_count = np.sum(types[cluster] == cation_type)
        a_count = np.sum(types[cluster] == anion_type)

        if c_count < 50 and a_count < 50:
            pop_matrix[c_count, a_count] += 1

    return pop_matrix, index

def calc_population_parallel(
    run,
    run_start,
    run_end,
    select_cations,
    select_anions,
    r_cut,
    core,
    csv_path,
    png_path,
):
    stacked_population = np.array([])

    with ProcessPoolExecutor(max_workers=core) as executor:
        futures = []

        for idx in range(run_start, run_end):
            run.trajectory[idx]

            cations = run.select_atoms(select_cations)
            anions = run.select_atoms(select_anions)

            # COPY DATA — CRITICAL
            positions = (cations + anions).positions.copy()
            box = run.trajectory.ts.dimensions[:3].copy()
            types = (cations + anions).types.copy()

            futures.append(
                executor.submit(
                    calc_population_frame_data,
                    positions,
                    box,
                    types,
                    idx,
                    r_cut,
                )
            )

        results = [
            f.result()
            for f in as_completed(futures)
        ]

    sorted_results = sorted(results, key=lambda x: x[1])

    for current_population, _ in sorted_results:
        if stacked_population.size == 0:
            stacked_population = current_population
        else:
            stacked_population = np.dstack((stacked_population, current_population))

    avg_population = np.mean(stacked_population, axis=2)
    np.savetxt(csv_path, avg_population, fmt="%.6f")
    plot_population_heatmap(avg_population, png_path)

def plot_population_heatmap(avg_population, png_path):
    rc("text", usetex=False)
    rc("font", family="serif")

    # Plot only a 10 × 10 matrix
    n = min(11, avg_population.shape[0], avg_population.shape[1])

    matrix = avg_population[:n, :n].T
    matrix = np.flipud(matrix)

    mat_plot = matrix.copy()
    mat_plot[mat_plot <= 0] = 1e-12

    # --- colormap ---
    colors = ["#f8f9fb", "#5B656D", "#004474"]
    cmap = LinearSegmentedColormap.from_list(
        "custom_blue_yellow", colors
    )
    norm = LogNorm(vmin=1e-4, vmax=1e2)

    plt.figure(figsize=(8, 7))

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

    # --- colorbar formatting ---
    cbar = ax.collections[0].colorbar
    cbar.ax.minorticks_off()
    cbar.set_ticks([10**i for i in range(-3, 3)])
    cbar.set_ticklabels([
        r"$10^{-3}$",
        r"$10^{-2}$",
        r"$10^{-1}$",
        r"$10^{0}$",
        r"$10^{1}$",
        r"$10^{2}$"
    ])
    cbar.ax.tick_params(labelsize=20)

    # --- axis ticks & labels ---
    x_labels = list(range(n))
    y_labels = list(range(n - 1, -1, -1))

    # One tick for every box
    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_yticks(np.arange(n) + 0.5)

    ax.set_xticklabels(x_labels, fontsize=20)
    ax.set_yticklabels(y_labels, fontsize=20)

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
                    0.9,
                    0.9,
                    fill=False,
                    edgecolor="#000000",
                    lw=0.5
                )
            )

    # --- diagonal ---
    ax.plot(
        [0, n],
        [n, 0],
        lw=1,
        color="#002845"
    )

    # --- axis labels ---
    ax.set_xlabel(r"$n+$", fontsize=26, labelpad=20)
    ax.set_ylabel(r"$n-$", fontsize=26, labelpad=20)

    plt.tight_layout()
    plt.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

def cip_finder(
    start,
    end,
    run,
    cation,
    anion,
    r_cut
):
    cluster_data = []
    print(f"{anion.split()[-1] }–{cation.split()[-1]} Distance cutoff (Å): {r_cut}")

    for idx in range(start, end):

        run.trajectory[idx]

        cations = run.select_atoms(cation)
        anions = run.select_atoms(anion)

        ions = cations + anions
        positions = ions.positions.copy()
        charges = ions.charges.copy()
        types = ions.types.copy()
        indices = ions.indices.copy()
        box = run.trajectory.ts.dimensions[:3].copy()

        dist_matrix = compute_periodic_distance_matrix(
            positions=positions,
            box_length=box,
        )

        model = DBSCAN(
            eps=r_cut,
            min_samples=1,
            metric="precomputed",
        )

        labels = model.fit_predict(dist_matrix)
        unique_clusters = np.unique(labels)
        #print(f"\nFrame {idx}")
        
        for cluster_id in unique_clusters:
            mask = labels == cluster_id

            if mask.sum() != 2:
                continue

            cluster_atoms = ions[mask]

            full_cluster = cluster_atoms.residues.atoms

            cluster_charges = (full_cluster.charges.copy())
            total_charge = cluster_charges.sum()
            if abs(total_charge) > 1e-4:
                continue


            #"""
            cluster_positions = unwrap_cluster_pbc(
                cluster_atoms=full_cluster,
                reference_name=anion.split()[-1],
                box=box,
            )
            #"""
            #cluster_positions = (full_cluster.positions.copy())
            
            cluster_types = (full_cluster.types.copy())
            cluster_indices = (full_cluster.indices.copy())


            #print( f"Cluster {cluster_id}: " f"n_atoms={len(full_cluster)}, " f"charge={total_charge:.3f}" ) 
            #for i in range(len(cluster_positions)):
            #     x, y, z = cluster_positions[i] 
            #     print( f" {cluster_types[i]:>4s} " f"q={cluster_charges[i]:>7.3f} " f"xyz=(" f"{x:8.3f}, " f"{y:8.3f}, " f"{z:8.3f})" )       
            
            cluster_data.append(
                {
                    "frame": idx,
                    "cluster": int(cluster_id),
                    "n_atoms": len(full_cluster),
                    "total_charge": float(total_charge),
                    "positions": cluster_positions,
                    "charges": cluster_charges,
                    "types": cluster_types,
                    "indices": cluster_indices,
                }
            )

    return cluster_data

def unwrap_cluster_pbc(
    cluster_atoms,
    reference_name,
    box,
):

    positions = cluster_atoms.positions.copy()

    ref_atoms = cluster_atoms.select_atoms(
        f"name {reference_name}"
    )

    if len(ref_atoms) == 0:
        raise ValueError(
            f"No reference atom '{reference_name}' "
            f"found in cluster."
        )

    # Use first matching atom
    ref_pos = ref_atoms.positions[0]

    unwrapped_positions = positions.copy()

    for i in range(len(unwrapped_positions)):
        delta = unwrapped_positions[i] - ref_pos
        delta -= box * np.round(delta / box)
        unwrapped_positions[i] = ref_pos + delta

    return unwrapped_positions


