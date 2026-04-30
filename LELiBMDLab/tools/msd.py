import os
from tqdm.auto import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
from scipy.stats import linregress



def autocorrFFT(x):
    N = len(x)
    F = np.fft.fft(x, n=2*N)
    PSD = F * F.conjugate()
    res = np.fft.ifft(PSD).real[:N]
    n = N*np.ones(N) - np.arange(0, N)
    return res / n

def msd_fft(r):
    N = len(r)
    D = np.square(r).sum(axis=1)
    D = np.append(D, 0)
    S2 = sum([autocorrFFT(r[:, i]) for i in range(r.shape[1])])
    Q = 2*D.sum()
    S1 = np.zeros(N)
    for m in range(N):
        Q = Q - D[m-1] - D[N-m]
        S1[m] = Q / (N-m)
    return S1 - 2*S2

def cross_corr(x, y):
    N = len(x)
    F1 = np.fft.fft(x, n=2**(N*2 - 1).bit_length())
    F2 = np.fft.fft(y, n=2**(N*2 - 1).bit_length())
    PSD = F1 * F2.conjugate()
    res = np.fft.ifft(PSD).real[:N]
    n = N*np.ones(N) - np.arange(0, N)
    return res / n

def msd_fft_cross(r, k):
    N = len(r)
    D = np.multiply(r, k).sum(axis=1)
    D = np.append(D, 0)
    S2 = sum([cross_corr(r[:, i], k[:, i]) for i in range(r.shape[1])])
    S3 = sum([cross_corr(k[:, i], r[:, i]) for i in range(k.shape[1])])
    Q = 2*D.sum()
    S1 = np.zeros(N)
    for m in range(N):
        Q = Q - D[m-1] - D[N-m]
        S1[m] = Q / (N-m)
    return S1 - S2 - S3

def msd_variance_cross(r1, r2, msd):

    # compute A1, recursive relation with D = r1^2 r2^2
    N = len(r1)
    D = np.square(r1)*np.square(r2)
    D = np.append(D, 0)
    Q = 2 * D.sum()
    A1 = np.zeros(N)
    for m in range(N):
        Q = Q - D[m - 1] - D[N - m]
        A1[m] = Q / (N - m)

    # compute A2, cross correlation of r1^2r2 and r2
    A2 = cross_corr(np.square(r1)*r2, r2)

    # compute A3, cross correlation of r1^2 and r2^2
    A3 = cross_corr(np.square(r1), np.square(r2))

    # compute A4, cross correlation of (r1r2^2) and r1
    A4 = cross_corr(r1*np.square(r2), r1)

    # compute A5, cross correlation of r1r2 and r1r2
    A5 = cross_corr(r1*r2, r1*r2)

    # compute A6, cross correlation of r1 and r1r2^2
    A6 = cross_corr(r1, np.square(r2)*r1)

    # compute A7, cross correlation of r2^2 and r1^2
    A7 = cross_corr(np.square(r2), np.square(r1))

    # compute A8, cross correlation of r2 and r1^2r2
    A8 = cross_corr(r2, np.square(r1)*r2)    

    var_x = A1 - 2*A2 + A3 - 2*A4 +4*A5 - 2*A6 + A7 - 2*A8 - msd**2
    n_minus_m = N * np.ones(N) - np.arange(0, N)   # divide by (N-m)^2 (Var[E[X]] = Var[X]/n)

    return var_x/n_minus_m

def msd_variance(r, msd):

    # compute A1, recursive relation with D = r^4
    N = len(r)
    D = r**4
    D = np.append(D, 0)
    Q = 2 * D.sum()
    A1 = np.zeros(N)
    for m in range(N):
        Q = Q - D[m - 1] - D[N - m]
        A1[m] = Q / (N - m)

    # compute A2, autocorrelation of r^2
    A2 = cross_corr(r**2, r**2)

    # compute A3 and A4, cross correlations of r and r^3
    A3 = cross_corr(r, r**3)
    A4 = cross_corr(r**3, r)

    var_x = A1 + 6*A2 - 4*A3 - 4*A4 - msd**2
    n_minus_m = N * np.ones(N) - np.arange(0, N)   # divide by (N-m)^2 (Var[E[X]] = Var[X]/n)

    return var_x/n_minus_m

# ========================= L_ij building blocks (conductivity branch) =========================
def calc_Lii_self(atom_positions, times):
    Lii_self = np.zeros(len(times))
    for atom_num in tqdm(range(np.shape(atom_positions)[1]), desc="Calculating MSD"):
        r = atom_positions[:, atom_num, :]
        Lii_self += msd_fft(np.array(r))
    return np.array(Lii_self)

def calc_Lii(atom_positions):
    return msd_fft(np.sum(atom_positions, axis=1))

def calc_Lij(cation_positions, anion_positions):
    r_cat = np.sum(cation_positions, axis = 1)
    r_an = np.sum(anion_positions, axis = 1)
    msd = msd_fft_cross(np.array(r_cat),np.array(r_an))
    return np.array(msd)

def calc_Ltot(run, cations, anions, start=0, stop=None):
    """
    Compute total MSD for conductivity using a trajectory slice.
    """

    traj = run.trajectory

    if stop is None:
        stop = traj.n_frames

    # Pre-split once (important for performance)
    cations_list = cations.atoms.split("residue")
    anions_list = anions.atoms.split("residue")

    qr = []

    for _ts in tqdm(traj[start:stop], desc="Calculating conductivity"):
        qr_temp = np.zeros(3)

        for cation in cations_list:
            qr_temp += cation.center_of_mass() * 1

        for anion in anions_list:
            qr_temp += anion.center_of_mass() * -1

        qr.append(qr_temp)

    return msd_fft(np.array(qr))


def compute_all_Lij(cation_positions, anion_positions, times):
    msd_self_cation = calc_Lii_self(cation_positions, times) 
    msd_cation = calc_Lii(cation_positions)
    msd_self_anion = calc_Lii_self(anion_positions, times)
    msd_anion = calc_Lii(anion_positions)
    msd_distinct_CatAn = calc_Lij(cation_positions, anion_positions)    
    return [msd_cation, msd_self_cation, msd_anion, msd_self_anion, msd_distinct_CatAn]

def calc_slope_msd(times_array, msd_array, dt_, interval_time, step_size):
    log_time = np.log(times_array[1:])
    log_msd = np.log(msd_array[1:])

    interval_msd = int(interval_time / dt_)
    min_time = 10
    max_time = 1000

    time_range = (None, None)
    min_diff = float('inf')
    best_slope = None

    for i in range(0, len(log_time) - interval_msd, step_size):
        if i + interval_msd > len(log_time):
            break

        window_start = times_array[i]
        window_end = times_array[i + interval_msd]
        if window_start < min_time or window_end > max_time:
            continue

        x = log_time[i:i + interval_msd]
        y = log_msd[i:i + interval_msd]

        if len(x) < 2:
            continue

        slope, intercept = np.polyfit(x, y, 1)

        diff = abs(slope - 1)

        if diff < min_diff:
            min_diff = diff
            best_slope = slope
            time_range = (times_array[i], times_array[i + interval_msd])

    # --- Linear regression instead of simple slope ---
    start_idx = int(time_range[0] / dt_)
    end_idx = int(time_range[1] / dt_)

    x = times_array[start_idx:end_idx]
    y = msd_array[start_idx:end_idx]

    if len(x) < 2:
        raise ValueError("Not enough points for regression")
    print(f"time range for regression: {time_range[0]:.2f} ps to {time_range[1]:.2f} ps slope = {best_slope:.4f}")
    slope, intercept = np.polyfit(x, y, 1)

    return slope, time_range

# ========================= Output files =========================

def write_msd_sigma(times_ps, msd_sigma, output_dir, suffix: int):
    msd_dir = os.path.join(output_dir, "msd_sigma_file")
    os.makedirs(msd_dir, exist_ok=True)

    path = os.path.join(msd_dir, f"msd_sigma_{suffix}.csv")

    np.savetxt(
        path,
        np.column_stack((times_ps, msd_sigma)),
        fmt="%.6f",
        header="time(ps) msd_sigma(A²/ps)",
    )

def write_msd_diff(times_ps, msd_diff, output_dir, num_atoms, species):
    msd_dir = os.path.join(output_dir, "msd_diffusion_files")
    os.makedirs(msd_dir, exist_ok=True)

    path = os.path.join(msd_dir, f"msd_diff_{species}.csv")

    np.savetxt(
        path,
        np.column_stack((times_ps, msd_diff)),
        fmt="%.6f",
        header="time(ps) msd_sigma(A^2)",
    )

# ========================= Fitting window selection (conductivity branch) =========================
def preview_plot_msd(
    msd_data,
    times,
    fname,
    title,
    ylabel,
    time_ranges=None,
    guide_multiplier=1.0,
):
    rc("text", usetex=False)
    rc("font", family="serif")

    font = {"title": 18, "label": 16, "tick": 12}
    colors = ["#3C3846", "#DF543F", "#2286A9", "#FBBF7C"]

    dt_collection = 500
    dt = 0.002
    dt_ = dt_collection * dt

    fig, ax = plt.subplots(figsize=(5.5, 4))

    # Optional diffusive guide
    if time_ranges is not None:
        mid_time = 0.5 * (time_ranges[0] + time_ranges[1])
        start = int(10 ** (np.log10(mid_time) - 0.15) / dt_)
        end   = int(10 ** (np.log10(mid_time) + 0.15) / dt_)

        scale = msd_data[int(mid_time / dt_)] / mid_time
        x_log = times[start:end]
        y_log = x_log * scale * guide_multiplier

        ax.plot(x_log, y_log, "--", linewidth=2, color=colors[2])

    # MSD curve
    ax.plot(times[1:], msd_data[1:], "-", linewidth=1.5, color=colors[0])

    ax.set_xlabel(r"$t$ (ps)", fontsize=font["label"])
    ax.set_ylabel(ylabel, fontsize=font["label"])
    ax.set_title(title, fontsize=font["title"])

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e2,)
    ax.grid(True, linestyle="--")

    # larger ticks (key change)
    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        labelsize=font["tick"],
        length=7,
        width=1.2,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        direction="in",
        length=4,
        width=1.0,
    )

    plt.tight_layout()
    fig.savefig(fname, dpi=160)
    plt.close(fig)


def preview_msd_sigma(times_ps, msd_sigma, time_ranges, output_dir, suffix: int):
    msd_dir = os.path.join(output_dir, "msd_sigma_file")
    os.makedirs(msd_dir, exist_ok=True)

    preview_plot_msd(
        msd_data=msd_sigma,
        times=times_ps,
        fname=os.path.join(msd_dir, f"msd_sigma_preview_{suffix}.png"),
        title=r"MSD $L^{tot} = L^{++}+L^{--}-2\times L^{+-}$",
        ylabel=r"MSD ($\mathrm{\AA}^2/ps$)",
        time_ranges=time_ranges,
        guide_multiplier=2.0,
    )


def preview_msd_diff(times_ps, msd_self, time_ranges, output_dir, key):
    msd_dir = os.path.join(output_dir, "msd_diffusion_files")
    os.makedirs(msd_dir, exist_ok=True)

    preview_plot_msd(
        msd_data=msd_self,
        times=times_ps,
        fname=os.path.join(msd_dir, f"msd_self_{key}.png"),
        title=f"MSD – {key.capitalize()}",
        ylabel=r"MSD ($\mathrm{\AA}^2/ps$)",
        time_ranges=time_ranges,
        guide_multiplier=2.0,
    )

# ========================= transformations =========================

def positions_array(run, atoms, times):
    run_start = 0
    time = 0
    atoms_list = atoms.atoms.split("residue")
    atoms_positions = np.zeros((times, len(atoms_list), 3))
    
    for ts in enumerate(run.trajectory[int(run_start):]):
        system_com = run.atoms.center_of_mass(wrap=True)
        for index, ion in enumerate(atoms_list):
            atoms_positions[time, index, :] = ion.center_of_mass() - system_com
        
        time += 1

    return atoms_positions  



"""
def calc_slope_msd(times_array, msd_array, dt_, interval_time, step_size):
    # Log transformation
    log_time = np.log(times_array[1:])
    log_msd = np.log(msd_array[1:])

    # calculate the time interval
    interval_msd = int(interval_time / dt_)

    max_time = 2000  # ps

    # Initialize a list to store the average slope for each large interval
    time_range = (None, None)
    min_slope_sum = float('inf')

    # Use a sliding window to calculate the average slope for each large interval
    for i in range(0, len(log_time) - interval_msd, step_size):
        if i + interval_msd > len(log_time):  # Ensure not to go out of bounds
            break

        window_end = times_array[i + interval_msd]
        
        if window_end > max_time:
            continue

        local_slope = np.gradient(log_msd[i:i + interval_msd], log_time[i:i + interval_msd])
        slope_difference_sum = np.sum(np.abs(local_slope - 1))
        if slope_difference_sum < min_slope_sum:
            min_slope_sum = slope_difference_sum            
            time_range = (times_array[i], times_array[i + interval_msd])

    # Calculate the final slope
    final_slope = (msd_array[int(time_range[1] / dt_)] - msd_array[int(time_range[0] / dt_)]) / (time_range[1] - time_range[0])
    return final_slope, time_range
"""


"""
def calc_Ltot(run, cations, anions, run_start =0):
    # Split atoms into lists by residue for cations and anions
    cations_list = cations.atoms.split("residue")
    anions_list = anions.atoms.split("residue")

    # compute sum over all charges and positions
    qr = []
    for _ts in tqdm(run.trajectory[run_start:], desc='Calculating conductivity'):
        qr_temp = np.zeros(3)
        for cation in cations_list:
            qr_temp += cation.center_of_mass() * int(1)
        for anion in anions_list:
            qr_temp += anion.center_of_mass() * int(-1)
        qr.append(qr_temp)
    return msd_fft(np.array(qr))
"""  