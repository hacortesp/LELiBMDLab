import os
from tqdm.auto import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Default fitting windows (fractions of len(times))
CAT_START_F   = 0.01
CAT_END_F     = 0.10
AN_START_F    = 0.01
AN_END_F      = 0.10
CROSS_START_F = 0.01
CROSS_END_F   = 0.10
TOTAL_START_F = 0.01
TOTAL_END_F   = 0.10

# Updated windows for SELF terms (fractions)
CAT_SELF_START_F = 0.10
CAT_SELF_END_F   = 0.90
AN_SELF_START_F  = 0.10
AN_SELF_END_F    = 0.90


# ========================= Common numeric kernels =========================
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
    

def compute_all_Lij(cation_positions, anion_positions, times):
    msd_self_cation = calc_Lii_self(cation_positions, times) 
    msd_cation = calc_Lii(cation_positions)
    msd_self_anion = calc_Lii_self(anion_positions, times)
    msd_anion = calc_Lii(anion_positions)
    msd_distinct_CatAn = calc_Lij(cation_positions, anion_positions)
    msd_tot = (msd_cation + msd_anion) - 2 * msd_distinct_CatAn
    return [msd_cation, msd_self_cation, msd_anion, msd_self_anion, msd_distinct_CatAn, msd_tot]


def calc_slope_msd(times_array, msd_array, dt_collection, dt, interval_time=1200, step_size=10):
    # Log transformation
    log_time = np.log(times_array[1:])
    log_msd = np.log(msd_array[1:])

    # calculate the time interval
    dt_ = dt_collection * dt
    interval_msd = int(interval_time / dt_)

    # Initialize a list to store the average slope for each large interval
    time_range = (None, None)
    min_slope_sum = float('inf')

    # Use a sliding window to calculate the average slope for each large interval
    for i in range(0, len(log_time) - interval_msd, step_size):
        if i + interval_msd > len(log_time):  # Ensure not to go out of bounds
            break
        local_slope = np.gradient(log_msd[i:i + interval_msd], log_time[i:i + interval_msd])
        slope_difference_sum = np.sum(np.abs(local_slope - 1))
        if slope_difference_sum < min_slope_sum:
            min_slope_sum = slope_difference_sum
            time_range = (times_array[i], times_array[i + interval_msd])

    # Calculate the final slope
    final_slope = (msd_array[int(time_range[1] / dt_)] - msd_array[int(time_range[0] / dt_)]) / (time_range[1] - time_range[0])

    return final_slope, time_range

# ========================= Output files =========================

def write_msds(num_cation, times_ps, msds_all, output_dir):
    msd_dir = os.path.join(output_dir, "msd_files")
    os.makedirs(msd_dir, exist_ok=True)

    filenames_headers = [
        ("msd_cations.csv", "time(ps) msd_cations(A²/ps)"),
        ("msd_self_cations.csv", "time(ps) msd_self_cations(A²/ps)"),
        ("msd_anions.csv", "time(ps) msd_anions(A²/ps)"),
        ("msd_self_anions.csv", "time(ps) msd_self_anions(A²/ps)"),
        ("msd_cation-anion.csv", "time(ps) msd_cation_anion(A²/ps)"),
        ("msd_total.csv", "time(ps) msd_total(A²/ps)")
    ]

    for i, (fname, header) in enumerate(filenames_headers):
        path = os.path.join(msd_dir, fname)
        np.savetxt(
            path,
            np.column_stack((times_ps, msds_all[i] / 6 /num_cation)),
            fmt="%.6f",
            header=header,
        )

# ========================= Fitting window selection (conductivity branch) =========================
def _preview_plot(
    msd_data,
    times,
    fname,
    title,
    time_ranges=None,
):
    font_list = {"title": 20, "label": 18, "legend": 16, "ticket": 18, "data": 14}
    color_list = ["#DF543F", "#2286A9", "#FBBF7C", "#3C3846"]
    
    dt_collection = 2000
    dt = 0.002

    dt_ = dt_collection * dt
    fig, ax = plt.subplots()

    # only compute and plot the power-law fit if time_ranges was passed
    if time_ranges is not None:
        mid_time = (time_ranges[1] + time_ranges[0]) / 2
        start = int(10 ** (np.log10(mid_time) - 0.15) / dt_)
        end   = int(10 ** (np.log10(mid_time) + 0.15) / dt_)
        scale = (msd_data[int(mid_time / dt_)] + 40) / mid_time

        x_log = times[start:end]
        y_log = x_log * scale

        ax.plot(x_log, y_log, '--', linewidth=2, color="grey")

    # always plot the raw MSD
    ax.plot(times[1:], msd_data[1:], '-', linewidth=1.5, color=color_list[0])

    ax.set_xlabel(r'$t$ (ps)', fontsize=font_list["label"])
    ax.set_ylabel(r'MSD ($\mathrm{\AA}^2$)', fontsize=font_list["label"])
    ax.set_title(title)
    ax.tick_params(axis='both', which='both', direction='in', labelsize=font_list["ticket"])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1e2,)
    ax.grid(True, linestyle='--')
    fig.set_size_inches(5.5, 4)
    plt.tight_layout()
    fig.savefig(fname, dpi=160)
    plt.close(fig)

def _idx_from_ps(times_ps, tmin_ps, tmax_ps):
    if tmin_ps >= tmax_ps:
        tmin_ps, tmax_ps = tmax_ps, tmin_ps
    i0 = int(np.searchsorted(times_ps, tmin_ps, side="left"))
    i1 = int(np.searchsorted(times_ps, tmax_ps, side="right"))
    i0 = max(0, min(i0, len(times_ps)-2))
    i1 = max(i0+1, min(i1, len(times_ps)-1))
    return i0, i1

def _default_window_ps(name, times_ps):
    n = len(times_ps)
    def frac(a,b): return (int(a*n), int(b*n))
    table = {
        "cation":       frac(CAT_START_F, CAT_END_F),
        "cation_self":  frac(CAT_SELF_START_F, CAT_SELF_END_F),
        "anion":        frac(AN_START_F, AN_END_F),
        "anion_self":   frac(AN_SELF_START_F, AN_SELF_END_F),
        "cation_anion": frac(CROSS_START_F, CROSS_END_F),
        "total":        frac(TOTAL_START_F, TOTAL_END_F),
    }
    i0, i1 = table[name]
    return times_ps[i0], times_ps[i1]

def slope_windows(times_ps, msds_all, num_cation, output_dir, label_prefix="msd_"):
    msd_dir = os.path.join(output_dir, "msd_files")
    os.makedirs(msd_dir, exist_ok=True)

    msd_dict = {
        "cation":      msds_all[0] / num_cation / 6,
        "cation_self": msds_all[1] / num_cation / 6,
        "anion":       msds_all[2] / num_cation / 6,
        "anion_self":  msds_all[3] / num_cation / 6,
        "cation_anion":    msds_all[4] / num_cation / 6,
        "total":    msds_all[5] / num_cation / 6,
    }

    out = {}

    for key, series in msd_dict.items():
        tmin_ps, tmax_ps = _default_window_ps(key, times_ps)

        fname = os.path.join(msd_dir, f"msdpreview_{label_prefix}{key}.png")
        _preview_plot(
            series,
            times_ps,
            fname=fname,
            title=f"{key}",
            time_ranges=(tmin_ps, tmax_ps),        
        )

        i0, i1 = _idx_from_ps(times_ps, tmin_ps, tmax_ps)
        out[key] = (tmin_ps, tmax_ps, i0, i1)

    return out

def fit_data(f, start, end, times):
    slope, _, _, _, _ = stats.linregress(times[start:end], f[start:end])
    return slope