import os
import math
import pandas as pd
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.dielectric import DielectricConstant


from LELiBMDLab.tools.msd import (
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

from LELiBMDLab.tools.coordination import (
    calc_rdf_coord,
    obtain_rdf_coord,
    plot_rdf_coordination,  
    analyze_coordination_structure,
    calc_population_parallel
)


from LELiBMDLab.tools.activity import (
    born_radius,
    calc_activity    
)


kb = 1.38E-23 # J K^-1
q =  1.60E-19 # C
NA = 6.02214076e23 # mol^-1
AMU_TO_KG = 1.66053906660e-27
convertion_factor = 1e22


class TrajAnalysis:
    def __init__(
        self,
        workdir: str,
        dt: float = 0.002,
        dt_collection: int = 2000,
        temperature: float = 300.0,
        cation_name: str = "resname LIP and name LI",
        anion_name: str = "resname _PF and name P1",
        q_eff: float = 0.85,
    ):
        self.workdir = workdir
        self.dt = dt
        self.dt_collection = dt_collection
        self.temp = temperature
        self.cation_name = cation_name
        self.anion_name = anion_name
        self.q_eff =  q_eff  
        
        tpr_path = os.path.join(self.workdir, "nvt_prod_wrap.tpr")
        wrap_xtc_path = os.path.join(self.workdir, "nvt_prod_wrap.xtc")
        unwrap_xtc_path = os.path.join(self.workdir, "nvt_prod_unwrap.xtc")

        self.run_wrap = mda.Universe(tpr_path, wrap_xtc_path)
        self.run_unwrap = mda.Universe(tpr_path, unwrap_xtc_path)

        self.cations_unwrap = self.run_unwrap.select_atoms(cation_name)
        self.anions_unwrap = self.run_unwrap.select_atoms(anion_name)
        
        self.volume = self.run_unwrap.coord.volume        
        self.num_cation = len(self.cations_unwrap)
        self.num_anion = len(self.anions_unwrap)
        self.num_frames = self.run_unwrap.trajectory.n_frames
        self.run_end = self.num_frames * self.dt_collection

        print("===== Electrolyte information =====")
        print(f"Work directory: {self.workdir}")
        V_m3 = self.volume * 1e-30
        m_sol_kg = self.run_wrap.atoms.masses.sum() * AMU_TO_KG
        rho_kg_m3 = m_sol_kg / V_m3  #kg·m^⁻3
        rho_g_cm3 = rho_kg_m3 * 1e-3
        print(f"Density: {rho_g_cm3:.4f} g·cm^-3")

        vsol_L = self.volume * 1e-27
        c_m = self.num_cation / NA / vsol_L
        print(f"Concentration: {c_m:.2f} mol L^-1")

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
        counter_atom: str
    ):
       
        solve_dir = os.path.join(self.workdir, "solvatation_structure")
        os.makedirs(solve_dir, exist_ok=True)

        png_path = os.path.join(solve_dir, "solvatation_structure.png")
        csv_path = os.path.join(solve_dir, "solvatation.csv")

        select_dict = {
            "center": center_atom,
            "counter": counter_atom,
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

        select_cations = center_atom
        select_anions = counter_atom

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

# ========================= Activity =========================
    def activity(
        self,        
        run_start: int,
        run_end: int,
        solv_dir: str
    ):
        SELECTION_TO_ION = {
        "resname LIP and name LI": 0.60,
        "resname _PF and name P1": 2.42,
        # add more as needed
        }

        # ---------- ion-size parameter ----------
        try:
            cation_ion = SELECTION_TO_ION[self.cation_name]
            anion_ion = SELECTION_TO_ION[self.anion_name]
        except KeyError as e:
            raise ValueError(f"Unknown ion: {e}")
        
        a = (cation_ion + anion_ion) * 1e-10

        eps_sol = self.permittivity(
            run_start,
            run_end,
            temperature=self.temp,
            trajdir=self.workdir,
        )  

        eps_solv = self.permittivity(
            run_start,
            run_end,
            temperature=self.temp,
            trajdir=solv_dir,
        )

        solv_tpr_path = os.path.join(solv_dir, "nvt_prod_wrap.tpr")
        solv_xtc_path = os.path.join(solv_dir, "nvt_prod_wrap.xtc")

        solv_wrap = mda.Universe(solv_tpr_path, solv_xtc_path)
        
        V_m3_solv = solv_wrap.coord.volume * 1e-30
        m_solv_kg = solv_wrap.atoms.masses.sum() * AMU_TO_KG
        rho_solv = m_solv_kg / V_m3_solv #kg·m^⁻3

        z_plus = 1 * self.q_eff
        z_minus = -1 * self.q_eff

        n_salt = self.num_cation / NA

        c = n_salt / m_solv_kg        
       
        Rb_plus, Rb_minus = born_radius(z_plus, z_minus, eps_solv)
        
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


    def permittivity(
        self,
        run_start: int,
        run_end: int,
        temperature: float,
        trajdir: str | None = None,
    ):
        make_whole = True
        charge_tolerance = 1e-6

        # ---------- resolve universe ----------
        if trajdir is None or trajdir == self.workdir:
            run_wrap = self.run_wrap
        else:
            solv_tpr_path = os.path.join(trajdir, "nvt_prod_wrap.tpr")
            solv_xtc_path = os.path.join(trajdir, "nvt_prod_wrap.xtc")
            if not os.path.isfile(solv_tpr_path):
                raise FileNotFoundError(solv_tpr_path)
            if not os.path.isfile(solv_xtc_path):
                raise FileNotFoundError(solv_xtc_path)

            run_wrap = mda.Universe(solv_tpr_path, solv_xtc_path)

        atoms = run_wrap.atoms

        if atoms.n_atoms == 0:
            raise ValueError("Universe contains no atoms.")

        if not hasattr(atoms, "charges"):
            raise AttributeError(
                "Atoms do not have charges. Ensure the topology includes charges."
            )

        neutral_atomgroups = [
            res.atoms
            for res in atoms.residues
            if abs(res.atoms.total_charge()) <= charge_tolerance
        ]

        if not neutral_atomgroups:
            raise ValueError("No neutral molecules found.")

        atomgroup = neutral_atomgroups[0]
        for ag in neutral_atomgroups[1:]:
            atomgroup += ag

        diel = DielectricConstant(
            atomgroup,
            temperature=temperature,
            make_whole=make_whole,
        )

        diel.run(start=run_start, stop=run_end)

        return diel.results.eps_mean
  
