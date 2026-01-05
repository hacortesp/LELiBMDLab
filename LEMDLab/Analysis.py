import os
import math
import pandas as pd
import numpy as np
import MDAnalysis as mda

from tqdm.auto import tqdm
from cclib.io import ccopen
from functools import partial
from functools import lru_cache
from multiprocessing import Pool
from typing import Optional, Dict

from LEMDLab.tools.msd import (
    calc_slope_msd,
    create_position_arrays,
    calc_Lii_self,
    calc_Lii,
    calc_self_diffusion_coeff,
    plot_msd,
    calc_cond_msd,
    calc_conductivity
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

class TrajAnalysis:

    def __init__(
        self,
        workdir: str,
        tpr_file: str = "nvt_prod.tpr",
        xtc_wrap_file: str = "nvt_wrap.xtc",
        xtc_unwrap_file: str = "nvt_unwrap.xtc",
        run_start: int = 0,
        run_end: int = 3000,
        dt: float = 0.002,
        dt_collection: int = 4,
        temperature: float = 300.0,
        select_dict: Optional[Dict[str, str]] = None,
        cation_name: str = "resname LIP and name LI",
        anion_name: str = "resname _PF and name P1",
        polymer_name: str = "resname _EC and name C2",
    ):

        self.xtc_wrap_file = xtc_wrap_file
        self.xtc_unwrap_file = xtc_unwrap_file
        self.run_start = run_start
        self.run_end = run_end
        self.dt = dt
        self.dt_collection = dt_collection
        self.temp = temperature
        
        self.select_dict = select_dict or {
            self.cation_name: f"resname {self.cation_name}",
            self.anion_name: f"resname {self.anion_name}",
            self.polymer_name: f"resname {self.polymer_name}",
        }
        self.cation_name = cation_name
        self.anion_name = anion_name
        self.polymer_name = polymer_name

        tpr_path = os.path.join(workdir, tpr_file)
        wrap_xtc_path = os.path.join(workdir, xtc_wrap_file)
        unwrap_xtc_path = os.path.join(workdir, xtc_unwrap_file)

        self.run_wrap = mda.Universe(tpr_path, wrap_xtc_path)
        self.run_unwrap = mda.Universe(tpr_path, unwrap_xtc_path)
        self.times = self.times_range()

        self.cations_unwrap = self.run_unwrap.select_atoms(self.select_dict.get(cation_name))
        self.anions_unwrap = self.run_unwrap.select_atoms(self.select_dict.get(anion_name))
        self.polymers_unwrap = self.run_unwrap.select_atoms(self.select_dict.get(polymer_name))
        self.volume = self.run_unwrap.coord.volume
        self.box_size = self.run_unwrap.dimensions[0]
        self.num_cation = len(self.cations_unwrap)
        self.num_o_polymer = len(self.polymers_unwrap)
        self.num_chain = len(np.unique(self.polymers_unwrap.resids))
        #self.num_o_chain = int(self.num_o_polymer // self.num_chain)


    def times_range(self) -> np.ndarray:
        n_frames = len(self.run_unwrap.trajectory[self.run_start:])
        dt_frame = self.dt * self.dt_collection

        return np.arange(
            0.0,
            n_frames * dt_frame,
            dt_frame,
            dtype=float,
        )

    def get_cond_array(self):

        return calc_cond_msd(
            self.run_unwrap,
            self.cations_unwrap,
            self.anions_unwrap,
            self.run_start,
        )
    
    def get_slope_msd(self, msd_array, interval_time=10000, step_size=10):

        slope, time_range = calc_slope_msd(
            self.times,
            msd_array,
            self.dt_collection,
            self.dt,
            interval_time,
            step_size
        )
        return slope, time_range

    def conductivity(self, save_csv=False, plot=False, ):

        msd_array = self.get_cond_array()
        slope, time_ranges = self.get_slope_msd(msd_array)

        if save_csv:
            df = pd.DataFrame({
                "time": self.times,
                "msd": msd_array
            })
            df.to_csv('msd.csv', index=False)

        if plot:
            plot_msd(
                msd_array,
                self.times,
                self.dt_collection,
                self.dt,
                time_ranges=time_ranges,
            )

        return calc_conductivity(
            slope,
            self.volume,
            self.temp
        )


    def get_ions_positions_array(self, mols_unwrap=None):

        mols_positions = create_position_arrays(
            self.run_unwrap,
            mols_unwrap,
            self.times,
            self.run_start,
        )
        return mols_positions


    def get_Lii_self_array(self, atom_positions):

        n_atoms = np.shape(atom_positions)[1]
        return calc_Lii_self(atom_positions, self.times) / n_atoms


    # calculate the self diffusion coefficient
    def diffusion_coefficient(self, atoms, save_csv=False, plot=False, ):

        atoms_unwrap = self.run_unwrap.select_atoms(atoms)
        atoms_positions = self.get_ions_positions_array(atoms_unwrap)

        msd_array = self.get_Lii_self_array(atoms_positions)
        slope, time_ranges = self.get_slope_msd(msd_array)

        if save_csv:
            df = pd.DataFrame({
                "time": self.times,
                "msd": msd_array
            })
            df.to_csv('msd.csv', index=False)

        if plot:
            plot_msd(
                msd_array,
                self.times,
                self.dt_collection,
                self.dt,
                time_ranges=time_ranges,
            )

        D = calc_self_diffusion_coeff(slope)

        return D