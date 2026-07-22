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
    calc_activity    
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
        print(f"Density: {rho_g_cm3:.4f} g·cm^-3")

        vsol_L = self.volume * 1e-27

        n_cation = self.num_cation / const.NA

        # molarity
        c_m = n_cation / vsol_L
        print(f"Concentration: {c_m:.2f} mol L^-1")

        # molality (correct)
        b_m = n_cation / m_solvent_kg
        print(f"Molality: {b_m:.2f} mol kg^-1")

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
            print(f'Slope for window {idx}: {slope}')
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

        print("--- summary ---")
        print(f"values = {values}")
        print(f"mean   = {mean:.4f}")
        print(f"std = {std:.4f}")

        if return_all:
            return mean, std, values, time_windows
        else:
            return mean, std

    def transfer_number_split(self, window_ps: float = 6000):
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

        t_values = self.calc_transfer_number(
            slope_plusplus_values,
            slope_minusminus_values,
            self.temp,
            self.volume,
            sigmas,
            self.q_eff
        )

        mean = np.mean(t_values)
        std = np.std(t_values, ddof=1)
        stderr = std / np.sqrt(len(t_values))

        print("--- summary ---")
        print(f"values = {t_values}")
        print(f"mean   = {mean:.4f}")
        print(f"std = {std:.4f}")

        return mean, std

    def calc_conductivity(self, slope, v, T, q_eff):
        convert = const.e2c * const.e2c / const.ps2s / const.A2cm * 1000

        sigma = (q_eff**2 * slope) / (2*3) / const.kB / T / v * convert # "mS/cm"

        return sigma
        
    def calc_transfer_number(self, slope_plusplus, slope_minusminus, T, v, sigma, q_eff):
        convert = const.e2c * const.e2c / const.ps2s / const.A2cm * 1000

        slope_plusminus = (sigma / convert * 6 * const.kB * T * v / q_eff**2 - slope_plusplus - slope_minusminus) / -2

        t = (slope_plusplus - slope_plusminus) / (slope_plusplus + slope_minusminus - 2 * slope_plusminus)   # mS/cm

        return t
    
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

        x_val, y_coord = obtain_rdf_coord(bins, rdf, coord_number)

        return x_val, y_coord

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
    
    def coordination_type(
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
        run_start: int,
        run_end: int,
        trajdir: str,
    ):

        # ---------- solvent trajectory ----------
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

        # ---------- volume ----------
        run_solv.trajectory[run_start]

        V_solv_m3 = (run_solv.trajectory.ts.volume * 1e-30)

        # ---------- corrected solvent permittivity ----------
        eps_solv = permittivity_corr(
            start=run_start,
            end=run_end,
            run=run_solv,
            temperature=self.temp,
            volume_m3=V_solv_m3,
            cip_dipoles_by_frame=None,
        )

        return eps_solv

    def permittivity_sol(
        self,
        run_start: int,
        run_end: int,
        eps_solv: float | None = None,
    ):

        # ---------- require solvent permittivity ----------
        if eps_solv is None:
            raise ValueError(
                "eps_solv must be provided to calculate the solution "
                "permittivity. Calculate it first using "
                "permittivity_solv(...), or provide a previously "
                "calculated value."
            )

        eps_solv = float(eps_solv)

        if eps_solv <= 1.0:
            raise ValueError(
                f"eps_solv must be larger than 1. "
                f"Received: {eps_solv}"
            )

        print(
            f"Solvent permittivity used for the CIP correction: "
            f"{eps_solv:.4f}"
        )

        # ---------- solution trajectory ----------
        run_wrap = self.run_wrap
        V_m3 = self.V_m3

        atoms = run_wrap.atoms

        if atoms.n_atoms == 0:
            raise ValueError(
                "Solution Universe contains no atoms."
            )

        if not hasattr(atoms, "charges"):
            raise AttributeError(
                "Solution atoms do not have charges."
            )

        # ==========================================================
        # Determine CIP cutoff
        # ==========================================================
        resnames = np.unique(atoms.resnames)

        cation_key = self.get_mol_selection_key(self.cation_name)
        anion_key = self.get_mol_selection_key(self.anion_name)

        print(f"Cation selection: {cation_key}")
        print(f"Anion selection: {anion_key}")
    
        r_cut = self.determine_cip_cutoff(
            resnames=resnames,
            cation_key=cation_key,
            anion_key=anion_key,
        )   

        print(f"Determined CIP cutoff distance: {r_cut:.4f} Å")

        r_cut = 6.0
     
        # ==========================================================
        # Find CIPs
        # ==========================================================
        cip_inf = cip_finder(
            start=run_start,
            end=run_end,
            run=run_wrap,
            cation=self.cation_name,
            anion=self.anion_name,
            r_cut=r_cut,
        )

        # ==========================================================
        # Calculate xi-corrected CIP dipoles
        # ==========================================================
        corr_Mcip = corrected_cip_dipoles(
            start=run_start,
            end=run_end,
            cip_array=cip_inf,
            anion=self.anion_name,
            eps_solv=eps_solv,
            q_scale=self.q_eff,
        )
        
        # ==========================================================
        # Calculate solution permittivity
        #npj Comput Mater 9, 175 (2023)
        # M_sol =
        #     M_solv(alpha-corrected)
        #     +
        #     M_CIP(xi-corrected)
        # ==========================================================
        eps_sol = permittivity_corr(
            start=run_start,
            end=run_end,
            run=run_wrap,
            temperature=self.temp,
            volume_m3=V_m3,
            cip_dipoles_by_frame=corr_Mcip,
            eps_solv_reference=eps_solv,
            print_decomposition=True,
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
        x_cat_an, _ = self.coordination_number(
            cation_key,
            anion_key,
            write_files=False,
        )
        print(f"Cation-anion distance: {x_cat_an:.4f} Å")
        # ---------- cation-solvent distances ----------
        solvent_selections = self.get_solvent_selections(
            resnames
        )

        solvent_distances = {}

        for res, solvent_key in solvent_selections.items():

            x_solv, _ = self.coordination_number(
                cation_key,
                solvent_key,
                write_files=False,
            )

            print(f"Cation-{solvent_key} distance: {x_solv:.4f} Å")
            solvent_distances[solvent_key] = x_solv

        # ---------- select cutoff ----------
        larger_solvents = {
            solvent_key: dist
            for solvent_key, dist in solvent_distances.items()
            if dist > x_cat_an
        }

        if larger_solvents:

            largest_solvent = max(
                larger_solvents,
                key=larger_solvents.get,
            )

            r_cut = larger_solvents[largest_solvent]

        else:
            r_cut = x_cat_an

        return r_cut

# ========================= Activity =========================
    def activity(
        self,        
        run_start: int,
        run_end: int,
        solv_dir: str
    ):
        if solv_dir is None:
            raise ValueError("solv_dir must be provided")
        
        if os.path.abspath(solv_dir) == os.path.abspath(self.workdir):
            raise ValueError("solv_dir must be different from self.workdir")

        SELECTION_TO_ION = {
        "resname LIP and name LI": 0.60,
        "resname _PF and name P1": 2.42,   # PF6-
        "resname TFS and name N1": 3.26,   # TFSI-
        "resname FSI and name N2": 2.90,   # FSI-
        "resname _BF and name B1": 2.30,   # BF4-
        "resname ClO and name Cl": 2.25    # ClO4-
        }

        # ---------- ion-size parameter ----------
        try:
            cation_ion = SELECTION_TO_ION[self.cation_name]
            anion_ion = SELECTION_TO_ION[self.anion_name]
        except KeyError as e:
            raise ValueError(f"Unknown ion: {e}")
        
        a = (cation_ion + anion_ion) * 1e-10
        """
        eps_sol = self.permittivity(
            run_start,
            run_end,
            temperature=self.temp,
            trajdir=self.workdir,
        )

        """

        eps_solv = self.permittivity(
            run_start,
            run_end,
            trajdir=solv_dir
        )


        #eps_solv = 3.1075 

        eps_sol = 5.3375 
        

        solv_tpr_path = os.path.join(solv_dir, "nvt_prod_wrap.tpr")
        solv_xtc_path = os.path.join(solv_dir, "nvt_prod_wrap.xtc")

        solv_wrap = mda.Universe(solv_tpr_path, solv_xtc_path)
        
        m_solv_kg = solv_wrap.atoms.masses.sum() * const.AMU_TO_KG
        rho_solv = m_solv_kg / self.V_m3 

        n_salt = self.num_cation / const.NA

        c = n_salt / m_solv_kg        
       
        Rb_plus, Rb_minus = born_radius(self.cation_name, self.anion_name, eps_solv)
        
        gamma_DH, gamma_B, gamma_DH_B = calc_activity(
            c,
            rho_solv,
            eps_solv,
            eps_sol,
            temp=self.temp,
            a=a,
            R_plus=Rb_plus,
            R_minus=Rb_minus,
        )

        print(f"Concentration (mol kg^-1): {c:.4f}")
        print(f"rho solvent (kg m^-3): {rho_solv:.2f}")
        print(f"Dielectric constant solvent: {eps_solv:.2f}")
        print(f"Dielectric constant solution: {eps_sol:.2f}")
        print(f"Ion-size parameter a (m): {a:.2e}")
        print(f"Born radius cation Rb+ (m): {Rb_plus:.2e}")
        print(f"Born radius anion Rb- (m): {Rb_minus:.2e}")
  
        return gamma_DH, gamma_B, gamma_DH_B















