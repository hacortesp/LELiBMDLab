import os
import math
from statistics import mean
from sys import stderr
import pandas as pd
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.dielectric import DielectricConstant

import LELiBMDLab.tools.constants as const

from LELiBMDLab.tools.msd import (
    calc_Ltot,
    calc_Lii_self,
    calc_Lii,
    calc_Lij,
    calc_slope_msd,
    write_msd_sigma,
    write_msd_diff,
    preview_msd_sigma,
    preview_msd_diff,
    positions_array
)

from LELiBMDLab.tools.coordination import (
    calc_rdf_coord,
    obtain_rdf_coord,
    plot_rdf_coordination,  
    analyze_coordination_structure,
    calc_population_parallel,
    cip_finder
)

from LELiBMDLab.tools.activity import (
    born_radius,
    calc_activity,
    calc_thermodynamic_factor    
)

from LELiBMDLab.tools.permittivity import (
    corrected_cip_dipoles,
    permittivity_corr,    
)

MOL_SELECTION = {
    "EC": "resname _EC and name C2",
    "DMC": "resname DMC and name C2",
    "EMC": "resname EMC and name C2",
    "PC": "resname _PC and name C2",
    "DEC": "resname DEC and name C2",
    "DME": "resname DME and name C3",
    "PF6": "resname _PF and name P1",
    "TFSI": "resname TFS and name N1",
    "FSI": "resname FSI and name N2",
    "BF4": "resname _BF and name B1",
    "ClO4": "resname ClO and name Cl",
    "Li": "resname LIP and name LI", 
}

class TrajAnalysis:
    def __init__(
        self,
        workdir: str,
        dt: float = 0.002,
        dt_collection: int = 500,
        temperature: float = 300.0,
        cation_name: str = "Li",
        anion_name: str = "PF6",
        q_eff: float = 0.80,
    ):
        self.workdir = workdir
        self.dt = dt
        self.dt_collection = dt_collection
        self.temp = temperature
        self.cation_name = MOL_SELECTION[cation_name]
        self.anion_name = MOL_SELECTION[anion_name]
        self.q_eff =  q_eff  
        
        tpr_path = os.path.join(self.workdir, "nvt_prod_wrap.tpr")
        wrap_xtc_path = os.path.join(self.workdir, "nvt_prod_wrap.xtc")
        unwrap_xtc_path = os.path.join(self.workdir, "nvt_prod_unwrap.xtc")

        self.run_wrap = mda.Universe(tpr_path, wrap_xtc_path)
        self.run_unwrap = mda.Universe(tpr_path, unwrap_xtc_path)

        self.cations_unwrap = self.run_unwrap.select_atoms(self.cation_name).residues
        self.anions_unwrap = self.run_unwrap.select_atoms(self.anion_name).residues
        
        self.volume = self.run_unwrap.coord.volume        
        self.num_cation = len(self.cations_unwrap)
        self.num_anion = len(self.anions_unwrap)
        self.num_frames = self.run_unwrap.trajectory.n_frames
        self.run_end = self.num_frames * self.dt_collection

        print("===== Electrolyte information =====")
        print(f"Work directory: {self.workdir}")

        self.V_m3 = self.volume * 1e-30

        # total mass
        m_total_kg = self.run_wrap.atoms.masses.sum() * const.AMU_TO_KG

        # ion mass (cation + anion)
        m_ion_kg = (
            self.cations_unwrap.masses.sum() +
            self.anions_unwrap.masses.sum()
        ) * const.AMU_TO_KG
        # solvent mass (what molality needs)
        m_solvent_kg = m_total_kg - m_ion_kg
  
        rho_kg_m3 = m_total_kg / self.V_m3
        rho_g_cm3 = rho_kg_m3 * 1e-3

        print(f"Volume: {self.V_m3 * 1e6:.4e} cm^3")
        print(f"Density: {rho_g_cm3:.4f} g/cm^3")

        vsol_L = self.volume * 1e-27

        n_cation = self.num_cation / const.NA

        # molarity
        c_m = n_cation / vsol_L
        print(f"Molarity: {c_m:.4f} mol/L")

        # molality (correct)
        b_m = n_cation / m_solvent_kg
        print(f"Molality: {b_m:.4f} mol/kg")

# ========================= conductivity =========================
    def conductivity_split(self, window_ps: float = 6000, return_all: bool = False):
        ps_per_frame = self.dt * self.dt_collection
        frames_per_window = int(window_ps / ps_per_frame)
        total_windows = self.num_frames // frames_per_window

        if total_windows < 1:
            raise ValueError("Trajectory too short")

        sigma_values = [];time_windows = []

        for i in range(total_windows):
            start = i * frames_per_window
            stop = start + frames_per_window
            idx = i + 1 

            times = np.arange(
                0,
                frames_per_window * ps_per_frame,
                ps_per_frame,
                dtype=float
            )
            msd_sigma = calc_Ltot(
                self.run_unwrap,
                self.cations_unwrap,
                self.anions_unwrap,
                start=start,
                stop=stop
            )
            
            slope, time_ranges = calc_slope_msd(
                times,
                msd_sigma,
                ps_per_frame,
                interval_time=100,
                step_size=4
            )
            #print(f'Slope for window {idx}: {slope:4f}')
            sigma_val = self.calc_conductivity(
                slope,
                self.volume,
                self.temp,
                self.q_eff
            )
            
            # save per-window outputs
            write_msd_sigma(times, msd_sigma, self.workdir, suffix=idx)
            preview_msd_sigma(times, msd_sigma, time_ranges, self.workdir, suffix=idx)
            sigma_values.append(sigma_val)
            time_windows.append(time_ranges)

        time_windows = np.array(time_windows, dtype=float)
        values = np.array(sigma_values, dtype=float)

        if values.size < 2:
            raise ValueError("Need at least two values to estimate error")

        mean = np.mean(values)
        std = np.std(values, ddof=1)
        stderr = std / np.sqrt(values.size)

        if return_all:
            return mean, stderr, values, time_windows
        else:
            return mean, stderr

    def transfer_number_split(self, window_ps: float = 6000, onsager_coef: bool = False ):
        ps_per_frame = self.dt * self.dt_collection
        frames_per_window = int(window_ps / ps_per_frame)
        total_windows = self.num_frames // frames_per_window

        if total_windows < 1:
            raise ValueError("Trajectory too short")

        # --- compute COM-corrected positions once ---
        cations_positions = positions_array(
            self.run_unwrap,
            self.cations_unwrap,
            self.num_frames
        )

        anions_positions = positions_array(
            self.run_unwrap,
            self.anions_unwrap,
            self.num_frames
        )

        _, _, sigmas, time_windows = self.conductivity_split(window_ps, return_all=True)

        slope_plusplus_values = []
        slope_minusminus_values = []

        for i in range(total_windows):
            start = i * frames_per_window
            stop = start + frames_per_window
            idx = i + 1

            # --- slice positions ---
            cat_pos = cations_positions[start:stop]
            an_pos = anions_positions[start:stop]

            # --- time array ---
            times = np.arange(
                0,
                frames_per_window * ps_per_frame,
                ps_per_frame,
                dtype=float
            )

            # --- compute MSDs ---
            msd_plusplus = calc_Lii(cat_pos)
            msd_minusminus = calc_Lii(an_pos)
       
            # --- slopes ---
            slope_plusplus, _ = calc_slope_msd(
                times, msd_plusplus, ps_per_frame,
                interval_time=100, step_size=4, forced_time_range=time_windows[i]
            )

            slope_minusminus, _ = calc_slope_msd(
                times, msd_minusminus, ps_per_frame,
                interval_time=100, step_size=4, forced_time_range=time_windows[i]
            )

            slope_plusplus_values.append(slope_plusplus)
            slope_minusminus_values.append(slope_minusminus)

        slope_plusplus_values = np.array(slope_plusplus_values, dtype=float)
        slope_minusminus_values = np.array(slope_minusminus_values, dtype=float)
        sigmas = np.asarray(sigmas, dtype=float)

        t_values, slope_plusminus_values = self.calc_transfer_number(
            slope_plusplus_values,
            slope_minusminus_values,
            self.temp,
            self.volume,
            sigmas,
            self.q_eff
        )

        t_values = np.asarray(t_values, dtype=float)
        slope_plusminus_values = np.asarray(slope_plusminus_values, dtype=float)

        mean = np.mean(t_values)
        std = np.std(t_values, ddof=1)
        stderr = std / np.sqrt(len(t_values))
        
        if onsager_coef:
            return mean, stderr, slope_plusplus_values, slope_minusminus_values, slope_plusminus_values
        else:
            return mean, stderr

    def calc_conductivity(self, slope, v, T, q_eff):
        convert = const.e2c * const.e2c / const.ps2s / const.A2cm * 1000

        sigma = (q_eff**2 * slope) / (2*3) / const.kB / T / v * convert # "mS/cm"

        return sigma
        
    def calc_transfer_number(self, slope_plusplus, slope_minusminus, T, v, sigma, q_eff):
        convert = const.e2c * const.e2c / const.ps2s / const.A2cm * 1000

        # Recover the unweighted total collective MSD slope.
        slope_total = (
            sigma
            / convert
            * 6.0
            * const.kB
            * T
            * v
            / q_eff**2
        )

        # slope_total = slope++ + slope-- - 2 slope+-
        slope_plusminus = (
            slope_total
            - slope_plusplus
            - slope_minusminus
        ) / -2.0

        t = (
            slope_plusplus - slope_plusminus
        ) / (
            slope_plusplus
            + slope_minusminus
            - 2.0 * slope_plusminus
        )

        return t,  slope_plusminus
    
# ========================= Coordination ========================= 

    def coordination_number(
        self,
        center_atom,
        counter_atom,
        write_files:bool = True,
    ):
        group1_name = MOL_SELECTION[center_atom]
        group2_name = MOL_SELECTION[counter_atom]
        
        bins, rdf, coord_number = self.get_rdf_coordination_array(group1_name, group2_name)

        res1 = self.extract_resname(group1_name)
        res2 = self.extract_resname(group2_name)
        tag = f"{res1}_{res2}" 

        if write_files:
            coor_dir = os.path.join(self.workdir, "coordination_files")
            os.makedirs(coor_dir, exist_ok=True)
                
            
            csv_path = os.path.join(coor_dir,  f"coord_{tag}.csv")

            df = pd.DataFrame(
                {
                    "bins": bins,
                    "rdf": rdf,
                    "coordination_number": coord_number,
                }
            )
            df.to_csv(csv_path, index=False)


            plot_rdf_coordination(
                bins,
                rdf,
                coord_number,
                self.workdir,
                filename=f"rdf_coordination_{tag}.png"
            )

        x_val, y_coord, peak_position = obtain_rdf_coord(bins, rdf, coord_number)

        return x_val, y_coord, peak_position

    def get_rdf_coordination_array(self, group1_name, group2_name):
        group1 = self.run_wrap.select_atoms(group1_name)
        group2 = self.run_wrap.select_atoms(group2_name)

        bins, rdf, coord_number = calc_rdf_coord(
            group1,
            group2,
            self.volume
        )
        return bins, rdf, coord_number

    def extract_resname(self, selection: str) -> str:
        for token in selection.split():
            if token.lower() != "resname":
                continue
            idx = selection.split().index(token)
            return selection.split()[idx + 1]
        raise ValueError(f"Cannot extract resname from '{selection}'")
    
    def coordination_classification(
        self,
        run_start: int,
        run_end: int,
        distance: float,
        center_atom: str,
        counter_atom: str
    ):

        center = MOL_SELECTION[center_atom]
        counter = MOL_SELECTION[counter_atom]

        solve_dir = os.path.join(self.workdir, "solvatation_structure")
        os.makedirs(solve_dir, exist_ok=True)

        png_path = os.path.join(solve_dir, "solvatation_structure.png")
        csv_path = os.path.join(solve_dir, "solvatation.csv")

        select_dict = {
            "center": center,
            "counter": counter,
        }

        df = analyze_coordination_structure(
            run=self.run_wrap,
            run_start=run_start,
            run_end=run_end,
            select_dict=select_dict,
            distance=distance,
            center_atom="center",
            counter_atom="counter",
            plot_path=png_path,
        )

        df.to_csv(csv_path, index=False, sep="\t")

        return df

    def ion_cluster_population(self, run_start, run_end, center_atom, counter_atom, distance=3.2, core = 2):

        select_cations = MOL_SELECTION[center_atom]
        select_anions = MOL_SELECTION[counter_atom]

        assoc_dir = os.path.join(self.workdir, "ion_association")
        os.makedirs(assoc_dir, exist_ok=True)

        png_path = os.path.join(assoc_dir, "ion_association_matrix.png")
        csv_path = os.path.join(assoc_dir, "ion_association.csv")

        calc_population_parallel(
            self.run_wrap,
            run_start,
            run_end,
            select_cations,
            select_anions,
            distance,
            core,
            csv_path,
            png_path,
        )

# ========================= Permittivity =========================
    def permittivity_solv(
        self,
        trajdir: str,
        start_ps: float = 12000,
        end_ps: float = 30000,
    ):
        """Calculate solvent permittivity over one trajectory interval."""
        start_ps = float(start_ps)
        end_ps = float(end_ps)

        if not np.isfinite(start_ps) or start_ps < 0:
            raise ValueError(
                "start_ps must be finite and non-negative. "
                f"Received: {start_ps}"
            )
        if not np.isfinite(end_ps) or end_ps <= start_ps:
            raise ValueError(
                "end_ps must be finite and greater than start_ps. "
                f"Received start_ps={start_ps}, end_ps={end_ps}"
            )

        solv_tpr_path = os.path.join(trajdir, "nvt_prod_wrap.tpr")
        solv_xtc_path = os.path.join(trajdir, "nvt_prod_wrap.xtc")

        if not os.path.isfile(solv_tpr_path):
            raise FileNotFoundError(
                f"nvt_prod_wrap.tpr not found in folder: {trajdir}"
            )
        if not os.path.isfile(solv_xtc_path):
            raise FileNotFoundError(
                f"nvt_prod_wrap.xtc not found in folder: {trajdir}"
            )

        run_solv = mda.Universe(solv_tpr_path, solv_xtc_path)
        atoms = run_solv.atoms

        if atoms.n_atoms == 0:
            raise ValueError("Solvent Universe contains no atoms.")
        if not hasattr(atoms, "charges"):
            raise AttributeError("Solvent atoms do not have charges.")

        total_frames = len(run_solv.trajectory)
        if total_frames == 0:
            raise ValueError("Solvent trajectory contains no frames.")

        ps_per_frame = float(run_solv.trajectory.dt)
        if not np.isfinite(ps_per_frame) or ps_per_frame <= 0:
            raise ValueError(
                "Could not determine a valid time interval between solvent "
                f"trajectory frames: {ps_per_frame}"
            )

        run_solv.trajectory[0]
        first_time_ps = float(run_solv.trajectory.ts.time)
        if not np.isfinite(first_time_ps):
            raise ValueError(
                f"Invalid time for first trajectory frame: {first_time_ps}"
            )

        frame_times_ps = (
            first_time_ps
            + np.arange(total_frames, dtype=float) * ps_per_frame
        )
        start_frame = int(
            np.searchsorted(frame_times_ps, start_ps, side="left")
        )
        end_frame = int(
            np.searchsorted(frame_times_ps, end_ps, side="left")
        )

        if start_frame >= total_frames:
            raise ValueError(
                f"start_ps={start_ps:.2f} ps lies outside the trajectory; "
                f"the last frame is at approximately "
                f"{frame_times_ps[-1]:.2f} ps."
            )
        if end_frame > total_frames or (
            end_frame == total_frames and end_ps > frame_times_ps[-1] + ps_per_frame
        ):
            raise ValueError(
                f"end_ps={end_ps:.2f} ps lies outside the trajectory; "
                f"the available half-open interval ends at approximately "
                f"{frame_times_ps[-1] + ps_per_frame:.2f} ps."
            )
        if end_frame <= start_frame:
            raise ValueError(
                "The requested interval contains no stored frames: "
                f"[{start_ps:.2f}, {end_ps:.2f}) ps."
            )

        actual_start_ps = frame_times_ps[start_frame]
        actual_end_ps = (
            frame_times_ps[end_frame]
            if end_frame < total_frames
            else frame_times_ps[-1] + ps_per_frame
        )

        run_solv.trajectory[start_frame]
        volume_solv_m3 = float(run_solv.trajectory.ts.volume) * 1e-30
        if not np.isfinite(volume_solv_m3) or volume_solv_m3 <= 0:
            raise ValueError(f"Invalid solvent volume: {volume_solv_m3} m³")

        eps_solv = float(
            permittivity_corr(
                start=start_frame,
                end=end_frame,
                run=run_solv,
                temperature=self.temp,
                volume_m3=volume_solv_m3,
                cip_dipoles_by_frame=None,
            )
        )
        if not np.isfinite(eps_solv):
            raise ValueError(
                f"Calculated solvent permittivity is not finite: {eps_solv}"
            )

        return eps_solv

    def permittivity_sol(
        self,
        eps_solv: float | None = None,
        start_ps: float = 12000,
        end_ps: float = 30000,
    ):
        """Calculate solution permittivity over one trajectory interval."""
        if eps_solv is None:
            raise ValueError(
                "eps_solv must be provided. Calculate it first with "
                "permittivity_solv(...), or provide a previously calculated "
                "value."
            )

        eps_solv = float(eps_solv)
        start_ps = float(start_ps)
        end_ps = float(end_ps)

        if not np.isfinite(eps_solv) or eps_solv <= 1.0:
            raise ValueError(
                "eps_solv must be finite and larger than 1. "
                f"Received: {eps_solv}"
            )
        if not np.isfinite(start_ps) or start_ps < 0:
            raise ValueError(
                "start_ps must be finite and non-negative. "
                f"Received: {start_ps}"
            )
        if not np.isfinite(end_ps) or end_ps <= start_ps:
            raise ValueError(
                "end_ps must be finite and greater than start_ps. "
                f"Received start_ps={start_ps}, end_ps={end_ps}"
            )

        run_wrap = self.run_wrap
        total_frames = len(run_wrap.trajectory)
        if total_frames == 0:
            raise ValueError("Solution trajectory contains no frames.")

        ps_per_frame = float(run_wrap.trajectory.dt)
        if not np.isfinite(ps_per_frame) or ps_per_frame <= 0:
            ps_per_frame = float(self.dt * self.dt_collection)
        if not np.isfinite(ps_per_frame) or ps_per_frame <= 0:
            raise ValueError(
                "Could not determine a valid time interval between stored "
                f"frames: {ps_per_frame}"
            )

        run_wrap.trajectory[0]
        first_time_ps = float(run_wrap.trajectory.ts.time)
        if not np.isfinite(first_time_ps):
            raise ValueError(
                f"Invalid time for first trajectory frame: {first_time_ps}"
            )

        frame_times_ps = (
            first_time_ps
            + np.arange(total_frames, dtype=float) * ps_per_frame
        )
        start_frame = int(
            np.searchsorted(frame_times_ps, start_ps, side="left")
        )
        end_frame = int(
            np.searchsorted(frame_times_ps, end_ps, side="left")
        )

        if start_frame >= total_frames:
            raise ValueError(
                f"start_ps={start_ps:.2f} ps lies outside the trajectory; "
                f"the last frame is at approximately "
                f"{frame_times_ps[-1]:.2f} ps."
            )
        if end_frame > total_frames or (
            end_frame == total_frames and end_ps > frame_times_ps[-1] + ps_per_frame
        ):
            raise ValueError(
                f"end_ps={end_ps:.2f} ps lies outside the trajectory; "
                f"the available half-open interval ends at approximately "
                f"{frame_times_ps[-1] + ps_per_frame:.2f} ps."
            )
        if end_frame <= start_frame:
            raise ValueError(
                "The requested interval contains no stored frames: "
                f"[{start_ps:.2f}, {end_ps:.2f}) ps."
            )

        actual_start_ps = frame_times_ps[start_frame]
        actual_end_ps = (
            frame_times_ps[end_frame]
            if end_frame < total_frames
            else frame_times_ps[-1] + ps_per_frame
        )

        atoms = run_wrap.atoms
        if atoms.n_atoms == 0:
            raise ValueError("Solution Universe contains no atoms.")
        if not hasattr(atoms, "charges"):
            raise AttributeError("Solution atoms do not have charges.")

        cation_key = self.get_mol_selection_key(self.cation_name)
        anion_key = self.get_mol_selection_key(self.anion_name)
        r_cut = self.determine_cip_cutoff(
            resnames=np.unique(atoms.resnames),
            cation_key=cation_key,
            anion_key=anion_key,
        )

        print(f"Solvent permittivity used for correction: {eps_solv:.4f}")
        print(f"Cation selection: {cation_key}")
        print(f"Anion selection: {anion_key}")
        print(f"Determined CIP cutoff distance: {r_cut:.4f} Å")

        cip_inf = cip_finder(
            start=start_frame,
            end=end_frame,
            run=run_wrap,
            cation=self.cation_name,
            anion=self.anion_name,
            r_cut=r_cut,
        )
        corr_Mcip = corrected_cip_dipoles(
            start=start_frame,
            end=end_frame,
            cip_array=cip_inf,
            anion=self.anion_name,
            eps_solv=eps_solv,
            q_scale=self.q_eff,
        )
        eps_sol = float(
            permittivity_corr(
                start=start_frame,
                end=end_frame,
                run=run_wrap,
                temperature=self.temp,
                volume_m3=self.V_m3,
                cip_dipoles_by_frame=corr_Mcip,
                eps_solv_reference=eps_solv,
                print_decomposition=True,
            )
        )
        if not np.isfinite(eps_sol):
            raise ValueError(
                f"Calculated solution permittivity is not finite: {eps_sol}"
            )

        return eps_sol

    def get_solvent_selections(
        self,
        resnames,
    ):

        solvent_selections = {}

        for res in resnames:

            # Skip cation/anion residues
            if (
                res in self.cation_name
                or res in self.anion_name
            ):
                continue

            # Find corresponding MOL_SELECTION key
            for key, selection in MOL_SELECTION.items():

                if f"resname {res}" in selection:

                    solvent_selections[res] = key
                    break

        return solvent_selections

    def get_mol_selection_key(
        self,
        selection_string,
    ):

        for key, value in MOL_SELECTION.items():

            if value == selection_string:
                return key

        raise ValueError(
            f"No MOL_SELECTION key found for:"
            f" {selection_string}"
        )

    def determine_cip_cutoff(
        self,
        resnames,
        cation_key,
        anion_key,
    ):

        # ---------- cation-anion distance ----------
        x_cat_an, _, x_cat_peak = self.coordination_number(
            cation_key,
            anion_key,
            write_files=False,
        )
        print(f"Cation-anion distance: {x_cat_an:.4f} Å")
        # ---------- cation-solvent distances ----------

        r_cut = x_cat_an

        return r_cut

# ========================= Activity =========================
    def activity(
        self,
        c_molar: float,
        eps_solv: float,
        eps_sol: float,
    ):
        """
        Calculate the Debye-Hückel, Born, and combined mean activity
        coefficients for a 1:1 electrolyte.

        Parameters
        ----------
        c_molar
            Salt concentration in mol L^-1.
        eps_solv
            Dielectric constant of the pure solvent.
        eps_sol
            Dielectric constant of the electrolyte solution.
        """

        SELECTION_TO_ION = {
            "resname LIP and name LI": 0.60,
            "resname _PF and name P1": 2.42,  # PF6-
            "resname TFS and name N1": 3.26,  # TFSI-
            "resname FSI and name N2": 2.90,  # FSI-
            "resname _BF and name B1": 2.30,  # BF4-
            "resname ClO and name Cl": 2.25,  # ClO4-
        }

        # ---------- Ion-size parameter ----------
        try:
            cation_ion = SELECTION_TO_ION[self.cation_name]
            anion_ion = SELECTION_TO_ION[self.anion_name]
        except KeyError as exc:
            raise ValueError(f"Unknown ion: {exc}") from exc

        # Angstrom -> m
        a = (cation_ion + anion_ion) * 1.0e-10

        Rb_plus, Rb_minus = born_radius(
            self.cation_name,
            self.anion_name,
            eps_solv,
        )

        gamma_DH, gamma_B, gamma_DH_B = calc_activity(
            c=c_molar,  # mol L^-1
            eps_solv=eps_solv,
            eps_sol=eps_sol,
            temp=self.temp,
            a=a,
            R_plus=Rb_plus,
            R_minus=Rb_minus,
        )

        return gamma_DH, gamma_B, gamma_DH_B

    def thermodynamic_factor(
        self,
        c_values,
        activity_coefficients_file,
        print_details=False,
    ):
        return calc_thermodynamic_factor(
            c_values=c_values,
            activity_coefficients_file=activity_coefficients_file,
            print_details=print_details,
        )

# ========================= Salt Diffusivity =========================
    def salt_diffusivity_split(self, xi: float, window_ps: float = 6000):
        (
            _,
            _,
            slope_plusplus_values,
            slope_minusminus_values,
            slope_plusminus_values,
        ) = self.transfer_number_split(
            window_ps=window_ps,
            onsager_coef=True,
        )

        D_salt_values = self.calc_salt_diffusivity(
            slope_plusplus_values,
            slope_minusminus_values,
            slope_plusminus_values,
            self.temp,
            xi,
        )

        D_salt_values = np.asarray(D_salt_values, dtype=float)

        if D_salt_values.size < 2:
            raise ValueError("Need at least two values to estimate error")

        mean = np.mean(D_salt_values)
        std = np.std(D_salt_values, ddof=1)
        stderr = std / np.sqrt(D_salt_values.size)

        print("--- salt diffusivity summary ---")
        print(f"ξ      = {xi:.4e}")
        print(f"values = {D_salt_values} cm² s⁻¹")
        print(f"mean   = {mean:.4e} cm² s⁻¹")
        print(f"std    = {std:.4e} cm² s⁻¹")
        print(f"stderr = {stderr:.4e} cm² s⁻¹")

        return mean, stderr

    def calc_salt_diffusivity(
        self,
        slope_plusplus,
        slope_minusminus,
        slope_plusminus,
        T,
        xi,
    ):
        slope_plusplus = np.asarray(slope_plusplus, dtype=float)
        slope_minusminus = np.asarray(slope_minusminus, dtype=float)
        slope_plusminus = np.asarray(slope_plusminus, dtype=float)

        z_plus = self.q_eff
        z_minus = -self.q_eff

        nu_plus = 1
        nu_minus = 1
        nu = nu_plus + nu_minus

        # Number of salt formula units calculated independently
        # from the cation and anion counts.
        num_salt_plus = self.num_cation / nu_plus
        num_salt_minus = self.num_anion / nu_minus

        if not np.isclose(num_salt_plus, num_salt_minus):
            raise ValueError(
                "Cation and anion counts are inconsistent with "
                f"stoichiometry ν+={nu_plus}, ν-={nu_minus}"
            )

        num_salt = 0.5 * (num_salt_plus + num_salt_minus)

        # Angstrom² ps⁻¹ -> m² s⁻¹
        slope_plusplus = (
            slope_plusplus * 1.0e-20 / const.ps2s
        )
        slope_minusminus = (
            slope_minusminus * 1.0e-20 / const.ps2s
        )
        slope_plusminus = (
            slope_plusminus * 1.0e-20 / const.ps2s
        )

        volume_m3 = self.V_m3

        # Molar Onsager coefficients
        denominator_L = (
            6.0
            * const.kB
            * T
            * volume_m3
            * const.NA**2
        )

        L_plusplus = slope_plusplus / denominator_L
        L_minusminus = slope_minusminus / denominator_L
        L_plusminus = slope_plusminus / denominator_L

        print(f"L++ = {L_plusplus}")
        print(f"L-- = {L_minusminus}")
        print(f"L+- = {L_plusminus}")

        # Salt concentration in mol m⁻³
        c = (num_salt / const.NA) / volume_m3
        R = const.NA * const.kB

        determinant = (
            L_plusplus * L_minusminus
            - L_plusminus**2
        )

        numerator = (
            -z_plus
            * z_minus
            * determinant
        )

        denominator = (
            z_plus**2 * L_plusplus
            + z_minus**2 * L_minusminus
            + 2.0 * z_plus * z_minus * L_plusminus
        )

        #if np.any(denominator <= 0.0):
        #    raise ValueError("The Onsager denominator must be positive")

        # Equation 16; initial result in m² s⁻¹
        D_salt = (
            nu
            * R
            * T
            * xi
            / (c * nu_plus * nu_minus)
            * numerator
            / denominator
        )

        # m² s⁻¹ -> cm² s⁻¹
        D_salt *= 1.0e4

        print(f"z+ = {z_plus}, z- = {z_minus}")
        print(f"ν+ = {nu_plus}, ν- = {nu_minus}, ν = {nu}")
        print(
            f"determinant = {determinant}\n"
            f"numerator   = {numerator}\n"
            f"denominator = {denominator}"
        )
        print(f"Number of salt formula units = {num_salt:.0f}")
        print(f"Salt concentration = {c:.4e} mol m⁻³")

        return D_salt




