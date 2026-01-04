import os
import math
import pandas as pd
import numpy as np
import MDAnalysis as mda
import LEMDLab.tools as lib

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
    plot_msd
)
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


class TrajAnalysis:

    def __init__(
        self,
        run_wrap: mda.Universe,
        run_unwrap: mda.Universe,
        run_start: int,
        run_end: int,
        dt: float,
        dt_collection: int,
        temperature: float,
        select_dict: Optional[Dict[str, str]] = None,
        cation_name: str = "cation",
        anion_name: str = "anion",
        polymer_name: str = "polymer",
    ):

        self.run_wrap = run_wrap
        self.run_unwrap = run_unwrap
        self.run_start = run_start
        self.run_end = run_end
        self.dt = dt
        self.dt_collection = dt_collection
        self.temp = temperature
        self.times = self.times_range(self.run_end)
        self.select_dict = select_dict or {}
        self.cation_name = cation_name
        self.anion_name = anion_name
        self.polymer_name = polymer_name
        self.cations_unwrap = run_unwrap.select_atoms(self.select_dict.get(cation_name))
        self.anions_unwrap = run_unwrap.select_atoms(self.select_dict.get(anion_name))
        self.polymers_unwrap = run_unwrap.select_atoms(self.select_dict.get(polymer_name))
        self.volume = self.run_unwrap.coord.volume
        self.box_size = self.run_unwrap.dimensions[0]
        self.num_cation = len(self.cations_unwrap)
        self.num_o_polymer = len(self.polymers_unwrap)
        self.num_chain = len(np.unique(self.polymers_unwrap.resids))
        self.num_o_chain = int(self.num_o_polymer // self.num_chain)