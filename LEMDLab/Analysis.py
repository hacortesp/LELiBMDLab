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
    compute_all_Lij,
    write_msds
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

kb, q = 1.38E-23, 1.60E-19



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
        self.q_eff = q_eff


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


    def conductivity(self):
        kbT = kb * self.temp
        run_end = self.num_frames * self.dt_collection
        times = np.arange(0, run_end* self.dt, self.dt * self.dt_collection, dtype=float)

        print("Number of cations: ", self.num_cation)
        print("Number of anions:  ", self.num_anion)

        cation_list = self.cations_unwrap.atoms.split("residue")
        anions_list = self.anions_unwrap.atoms.split("residue")

        cation_positions = np.zeros((self.num_frames, self.num_cation, 3), dtype=float)
        anion_positions = np.zeros((self.num_frames, self.num_anion, 3), dtype=float)

        for iframe, ts in enumerate(self.run_unwrap.trajectory):
            for i, cation in enumerate(cation_list):
                cation_positions[iframe, i] = cation.center_of_mass()

            for i, anion in enumerate(anions_list):
                anion_positions[iframe, i] = anion.center_of_mass()

        msds_all = compute_all_Lij(cation_positions, anion_positions, times)
        write_msds(self.num_cation, times, msds_all, self.workdir)
