import os
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

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

def calc_Lii(atom_positions, times):
    return msd_fft(np.sum(atom_positions, axis=1))

def calc_Lij(cation_positions, anion_positions, times):
    return msd_fft_cross(np.sum(cation_positions, axis=1), np.sum(anion_positions, axis=1))

def compute_all_Lij(cation_positions, anion_positions, times):
    msd_self_cation = calc_Lii_self(cation_positions, times)
    msd_cation = calc_Lii(cation_positions, times)
    msd_self_anion = calc_Lii_self(anion_positions, times)
    msd_anion = calc_Lii(anion_positions, times)
    msd_distinct_CatAn = calc_Lij(cation_positions, anion_positions, times)
    msd_tot = (msd_cation + msd_anion) - 2 * msd_distinct_CatAn
    return [msd_cation, msd_self_cation, msd_anion, msd_self_anion, msd_distinct_CatAn, msd_tot]


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
            np.column_stack((times_ps, msds_all[i]/num_cation)),
            fmt="%.6f",
            header=header,
        )

def _default_window_ps(name, times_ps):
    n = len(times_ps)
    def frac(a,b): return (int(a*n), int(b*n))
    table = {
        "cat":       frac(CAT_START_F, CAT_END_F),
        "cat_self":  frac(CAT_SELF_START_F, CAT_SELF_END_F),
        "an":        frac(AN_START_F, AN_END_F),
        "an_self":   frac(AN_SELF_START_F, AN_SELF_END_F),
        "cross":     frac(CROSS_START_F, CROSS_END_F),
        "total":     frac(TOTAL_START_F, TOTAL_END_F),
    }
    i0, i1 = table[name]
    return times_ps[i0], times_ps[i1]