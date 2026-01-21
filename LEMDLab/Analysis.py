import os
import math
import pandas as pd
import numpy as np
import MDAnalysis as mda


from LEMDLab.tools.msd import (
    calc_Ltot,
    calc_Lii_self,
    calc_Lii,
    calc_slope_msd,
    write_msd_sigma,
    write_msd_diff,
    preview_msd_sigma,
    preview_msd_diff,
    positions_array
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
from LEMDLab.tools.ion_dynamics import (
    process_traj,
    calc_tau3,
    calc_delta_n_square,
    calc_tau1,
    ms_endtoend_distance,
    fit_rouse_model,
    calc_msd_M2
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
        self.q_eff =  q_eff
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

# ========================= conductivity =========================
    def conductivity(self): 
        self.times = np.arange(0, self.run_end* self.dt, self.dt * self.dt_collection, dtype=float)

        msd_sigma = calc_Ltot(self.run_unwrap, self.cations_unwrap, self.anions_unwrap)
        slope, time_ranges = calc_slope_msd(
            self.times,
            msd_sigma,
            self.dt_collection,
            self.dt,
            interval_time=1200,
            step_size=10,
            )
        sigma_val = self.calc_conductivity(slope, self.volume, self.temp, self.q_eff)
        write_msd_sigma(self.times, msd_sigma, self.workdir)
        preview_msd_sigma(self.times, msd_sigma, time_ranges, self.workdir)
        return sigma_val    

    def difusivity(self):
        species = {
            "cation": {
                "positions": positions_array(
                    self.run_unwrap,
                    self.cations_unwrap,
                    self.num_frames
                ),
                "num": self.num_cation,

            },
            "anion": {
                "positions": positions_array(
                    self.run_unwrap,
                    self.anions_unwrap,
                    self.num_frames
                ),
                "num": self.num_anion,
            },
        }

        diffusivities = {}

        for name, data in species.items():
            msd = calc_Lii_self(data["positions"], self.times)

            slope, time_ranges = calc_slope_msd(
                self.times,
                msd,
                self.dt_collection,
                self.dt,
                interval_time=9600,
                step_size=10,
            )

            # Diffusivity (inside the loop, as requested)
            D = self.calc_self_diffusion(slope)

            # Store result
            diffusivities[name] = D

            write_msd_diff(
                self.times,
                msd,
                self.workdir,
                data["num"],
                species=name,
            )

            preview_msd_diff(
                self.times,
                msd,
                time_ranges,
                self.workdir,
                name,   # "cation" or "anion"
            )

        return diffusivities
    

    def transfer_number(self):

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

        slope_plusplus, time_range_plusplus = calc_slope_msd(
            self.times,
            calc_Lii(cations_positions),
            self.dt_collection,
            self.dt,
            interval_time=1200,
            step_size=10,
            )
        
        slope_minusminus, time_range_minusminus = calc_slope_msd(
            self.times,
            calc_Lii(anions_positions),
            self.dt_collection,
            self.dt,
            interval_time=1200,
            step_size=10,
            )

        return self.calc_transfer_number(
            slope_plusplus,
            slope_minusminus,
            self.temp,
            self.volume,
            self.conductivity(),
            self.q_eff
        )


    def calc_conductivity(self, slope, v, T, q_eff):
        # Calculate conductivity from the slope
        A2cm = 1e-8  # Angstroms to cm
        ps2s = 1e-12  # picoseconds to seconds
        e2c = 1.60217662e-19  # elementary charge to Coulomb
        kb = 1.38064852e-23  # Boltzmann Constant, J/K
        convert = e2c * e2c / ps2s / A2cm * 1000

        sigma = (q_eff**2 * slope) / 6 / kb / T / v * convert # "mS/cm"

        return sigma
    
    def calc_self_diffusion(self, slope):
        # Constants for unit conversion from Angstroms squared to centimeters squared, and picoseconds to seconds
        A2cm = 1e-8  # Angstroms to cm
        ps2s = 1e-12  # picoseconds to seconds
        convert = (A2cm ** 2) / ps2s   # conversion factor for cm^2/s

        # Calculate the self-diffusion coefficient, D, using the slope of the MSD curve
        D = slope * convert / 6  # factor of 6 for three-dimensional diffusion

        return D
    
    def calc_transfer_number(self, slope_plusplus, slope_minusminus, T, v, sigma, q_eff):
        A2cm = 1e-8  # Angstroms to cm
        ps2s = 1e-12  # picoseconds to seconds
        e2c = 1.60217662e-19  # elementary charge to Coulomb
        kb = 1.38064852e-23  # Boltzmann Constant, J/K
        convert = e2c * e2c / ps2s / A2cm * 1000

        slope_plusminus = (sigma / convert * 6 * kb * T * v / q_eff**2 - slope_plusplus - slope_minusminus) / -2

        t = (slope_plusplus - slope_plusminus) / (slope_plusplus + slope_minusminus - 2 * slope_plusminus)   # mS/cm

        return t
    
# ========================= Coordination ========================= 

    def coordination_number(self, group1_name, group2_name):
        bins, rdf, coord_number = self.get_rdf_coordination_array(group1_name, group2_name)

        res1 = self.extract_resname(group1_name)
        res2 = self.extract_resname(group2_name)
        tag = f"{res1}__{res2}" 

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
        counter_atom: str,
        plot: bool = False,
    ):
       

        select_dict = {
            "center": center_atom,
            "counter": counter_atom,
        }

        return analyze_coordination_structure(
            run=self.run_wrap,
            run_start=run_start,
            run_end=run_end,
            select_dict=select_dict,
            distance=distance,
            center_atom="center",
            counter_atom="counter",
            plot=plot,
        )
