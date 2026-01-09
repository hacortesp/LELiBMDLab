import os
import math
import pandas as pd
import numpy as np
import MDAnalysis as mda


from LEMDLab.tools.msd import (
    compute_all_Lij,
    calc_Ltot,
    calc_slope_msd,
    write_msds,
    slope_windows,
    fit_data
)

"""
from LEMDLab.tools.ion_dynamics import (
    process_traj,
    calc_tau3,
    calc_delta_n_square,
    calc_tau1,
    ms_endtoend_distance,
    fit_rouse_model,
    calc_msd_M2
)
from LEMDLab.tools.coordination import (
    calc_rdf_coord,
    obtain_rdf_coord,
    plot_rdf_coordination,
    num_of_neighbor,
    pdb2mol,
    get_cluster_index,
    find_poly_match_subindex,
    get_cluster_withcap,
    merge_xyz_files,
    analyze_coordination_structure,
    calc_population_parallel
)
"""

kb = 1.38E-23
q =  1.60E-19
convertion_factor = 1e22

class TrajAnalysis:
    def __init__(
        self,
        workdir: str,
        tpr_file: str = "nvt_prod_wrap.tpr",
        xtc_wrap_file: str = "nvt_wrap.xtc",
        xtc_unwrap_file: str = "nvt_unwrap.xtc",
        dt: float = 0.002,
        dt_collection: int = 2000,
        temperature: float = 300.0,
        cation_name: str = "resname LIP and name LI",
        anion_name: str = "resname _PF and name P1",
        q_eff: float = 0.85,
    ):
        self.workdir = workdir
        self.xtc_wrap_file = xtc_wrap_file
        self.xtc_unwrap_file = xtc_unwrap_file
        self.dt = dt
        self.dt_collection = dt_collection
        self.temp = temperature
        self.cation_name = cation_name
        self.anion_name = anion_name
        self.q_eff =  q_eff * q
        self.kbT = kb * self.temp
        
        tpr_path = os.path.join(self.workdir, tpr_file)
        wrap_xtc_path = os.path.join(self.workdir, xtc_wrap_file)
        unwrap_xtc_path = os.path.join(self.workdir, xtc_unwrap_file)

        self.run_wrap = mda.Universe(tpr_path, wrap_xtc_path)
        self.run_unwrap = mda.Universe(tpr_path, unwrap_xtc_path)

        self.cations_unwrap = self.run_unwrap.select_atoms(cation_name)
        self.anions_unwrap = self.run_unwrap.select_atoms(anion_name)
        
        self.volume = self.run_unwrap.coord.volume        
        self.num_cation = len(self.cations_unwrap)
        self.num_anion = len(self.anions_unwrap)
        self.num_frames = self.run_unwrap.trajectory.n_frames
        self.run_end = self.num_frames * self.dt_collection


    def conductivity(self): 
        self.times = np.arange(0, self.run_end* self.dt, self.dt * self.dt_collection, dtype=float)

        print("Number of cations: ", self.num_cation)
        print("Number of anions:  ", self.num_anion)

        cation_positions, anion_positions = self.coord_arr()

        msds_all = compute_all_Lij(cation_positions, anion_positions, self.times)

        write_msds(self.num_cation, self.times, msds_all, self.workdir)
        
        windows = slope_windows(self.times, msds_all, self.num_cation, self.workdir)
        i0, i1 = windows["total"][2], windows["total"][3]
        print("Fitting Ltot in time range: ", self.times[i0], " to ", self.times[i1])
        msd_total_norm = msds_all[5] / 6.0 / self.kbT / self.volume
        l_total = fit_data(msd_total_norm, i0, i1, self.times)*(self.q_eff**2)*convertion_factor
        
        l_totPEMD = calc_Ltot(self.run_unwrap, self.cations_unwrap, self.anions_unwrap)
        slope, time_ranges = self.get_slope_msd(l_totPEMD)
        print("slope Ltot from calc_Ltot:", slope, "in time range:", time_ranges)
        cond = self.calc_conductivity(slope * 0.85**2, self.volume, self.temp)
        print(f"conductivity = {cond:.2f} mS/cm")

        return l_total

    def coord_arr(self):
        cation_list = self.cations_unwrap.atoms.split("residue")
        anions_list = self.anions_unwrap.atoms.split("residue")

        cation_positions = np.zeros((self.num_frames, self.num_cation, 3), dtype=float)
        anion_positions = np.zeros((self.num_frames, self.num_anion, 3), dtype=float)

        for iframe, ts in enumerate(self.run_unwrap.trajectory):
            for i, cation in enumerate(cation_list):
                cation_positions[iframe, i] = cation.center_of_mass()
            for i, anion in enumerate(anions_list):
                anion_positions[iframe, i] = anion.center_of_mass()

        return cation_positions, anion_positions
    
    def get_slope_msd(self, msd_array, interval_time=1200, step_size=10):
        slope, time_range = calc_slope_msd(
            self.times,
            msd_array,
            self.dt_collection,
            self.dt,
            interval_time,
            step_size
        )
        return slope, time_range
    
    def calc_conductivity(self, slope, v, T):
        # Calculate conductivity from the slope
        A2cm = 1e-8  # Angstroms to cm
        ps2s = 1e-12  # picoseconds to seconds
        e2c = 1.60217662e-19  # elementary charge to Coulomb
        kb = 1.38064852e-23  # Boltzmann Constant, J/K
        convert = e2c * e2c / ps2s / A2cm * 1000

        cond = slope / 6 / kb / T / v * convert   # "mS/cm"

        return cond