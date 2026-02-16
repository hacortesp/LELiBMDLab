import argparse
from timeit import default_timer as timer
from LELiBMDLab.SimulationCell import Builder
from LELiBMDLab.MDSimulation import MDsim
from LELiBMDLab.Analysis import TrajAnalysis


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse input arguments for the builder script."
    )

    parser.add_argument(
        "-box",
        type=float,
        default=40.0,
        metavar="L",
        help="Simulation box side length in Å (cubic box, default: 40).",
    )

    parser.add_argument(
        "-cation_name",
        type=str,
        default="Li",
        help="Cation name key in mol_dict (default: Li).",
    )

    parser.add_argument(
        "-anion_name",
        type=str,
        default="PF6",
        help="Anion name (default: PF6).",
    )

    parser.add_argument(
        "-salt-conc",
        type=float,
        default=1.0,
        help="Salt concentration in mol/L (default: 1.0).",
    )

    parser.add_argument(
        "-solvents",
        nargs="+",
        type=str,
        default=["EC","DMC"],
        help="List of solvents (default: EC DMC).",
    )

    parser.add_argument(
        "-solvent-fracs",
        nargs="+",
        type=float,
        default=[0.5, 0.5],
        help="Relative solvent volume fractions (default: 0.5 0.5).",
    )

    parser.add_argument(
        "-workdir",
        type=str,
        default="./simulation_test",
        help="Working directory for simulation files (default: workdir).",
    )

    parser.add_argument(
        "-charge-scale",
        type=float,
        default=1.0,
        help="Charge scaling factor (default: 1.0).",
    )

    parser.add_argument(
        "-temperature",
        type=float,
        default=298.0,
        help="Temperature in Kelvin (default: 298.0).",
    )

    parser.add_argument(
        "-gromacs-exec",
        type=str,
        default="gmx",
        help="GROMACS executable (default: gmx).",
    )
    
    parser.add_argument(
        "-parallel",
        type=bool,
        help="Whether to run GROMACS in parallel (default: False).",
    )

    parser.add_argument(
        "-dt",
        type=float,
        default=0.002,
        help="Timestep in ps (default: 0.002).",
    )

    parser.add_argument(
        "-dt_collection",
        type=float,
        default=2000,
        help="Data collection interval (default: 2000).",
    )
    return parser.parse_args()

args = parse_args()

start = timer()


mol_dict = {
    "EC": "resname _EC and name C2",
    "DMC": "resname DMC and name C2",
    "EMC": "resname EMC and name C2",
    "PC": "resname _PC and name C2",
    "DEC": "resname DEC and name C2",
    "DME": "resname DME and name C3",
    "PF6": "resname _PF and name P1",
    "TFSI": "resname TFS and name N1",
    "Li": "resname LIP and name LI", 
}

try:
    cation_sel = mol_dict[args.cation_name]
except KeyError:
    raise KeyError(f"Unknown cation_name '{args.cation_name}'. Available: {list(mol_dict)}")

try:
    anion_sel = mol_dict[args.anion_name]
except KeyError:
    raise KeyError(f"Unknown anion_name '{args.anion_name}'. Available: {list(mol_dict)}")


Builder(
    box=args.box,
    salt=args.anion_name,
    salt_conc=args.salt_conc,
    solvents=args.solvents,
    solvent_fracs=args.solvent_fracs,
    workdir=args.workdir,

)

MDsim(
    charge_scale=args.charge_scale,
    workdir=args.workdir,
    parallel=args.parallel,
    temperature=args.temperature,
    gmx_exec=args.gromacs_exec,
)

"""
analysis = TrajAnalysis(
    workdir=args.workdir,
    dt=args.dt,
    dt_collection=args.dt_collection,
    temperature=args.temperature,
    cation_name=cation_sel,
    anion_name=anion_sel,
    q_eff=args.charge_scale,
)

print("\n===== Calculating activity =====")
gamma_DH, gamma_B, gamma_DH_B = analysis.activity(
    run_start = 1500, 
    run_end = 1550,
    solv_dir="solvents"
)

print(f"Results {gamma_DH:.2f}, {gamma_B:.2f}, {gamma_DH_B:.2f}\n")


print("\n===== Calculating permittivity =====")
epsilon = analysis.permittivity(
    run_start = 1500, 
    run_end = 1550,
    temperature= 300
)

print(f"Dielectric constant = {epsilon:.2f}\n")


analysis = TrajAnalysis(
    workdir=args.workdir,
    tpr_file=args.tpr_file,
    xtc_wrap_file=args.xtc_wrap_file,
    xtc_unwrap_file=args.xtc_unwrap_file,
    dt=args.dt,
    dt_collection=args.dt_collection,
    temperature=args.temperature,
    cation_name=cation_sel,
    anion_name=anion_sel,
)


print("\n===== Calculating solvatation structure =====")

df = analysis.coordination_type(
    run_start = 1500, 
    run_end = 3000, 
    distance = 3.3, 
    center_atom = mol_dict["Li"], 
    counter_atom = mol_dict["PF6"]
)
print("Writing files in solvatation_structure folder")
print("Consider:")
print("*Solvent-separated ion pairs (SSIP)")
print("*Contact ion pairs (CIP)")
print("*Aggregate (AGG)")

for _, row in df.iterrows():
    print(f"{row['Solvation structure'].upper():>4s}  = {row['Percentage']}")



print("\n===== Calculating coordination =====")
# 1) Cation–Anion
distance, coord = analysis.coordination_number(cation_sel, anion_sel)
print(
    f"{args.cation_name}–{args.anion_name}, "
    f"coordination number: {float(coord):.2f}, "
    f"distance: {float(distance):.2f} Å"
)

# 2) Cation–Solvents
for solvent in args.solvents:
    if solvent not in mol_dict:
        raise KeyError(
            f"Unknown solvent '{solvent}'. Available: {list(mol_dict)}"
        )

    solvent_sel = mol_dict[solvent]

    distance, coord = analysis.coordination_number(cation_sel, solvent_sel)

    print(
        f"{args.cation_name}–{solvent}, "
        f"coordination number: {float(coord):.2f}, "
        f"distance: {float(distance):.2f} Å"
    )
print("Writing files in coordination_files folder")

print("\n===== Calculating ion association =====")

analysis.ion_cluster_population(
    run_start = 1500, 
    run_end = 3000, 
    center_atom = mol_dict["Li"], 
    counter_atom = mol_dict["PF6"],
    distance=2.7,
    core = 2,
)
print("Writing files in ion_association folder")

print("===== Calculating conductivity =====")
cond = analysis.conductivity()
print(f"Ionic conductivity   = {cond:.6f} mS/cm\n")

print("\n===== Calculating Self-diffusion coefficients =====")
diff = analysis.difusivity()
for species, D in diff.items():
    print(f"  {species.capitalize():6s}: {D:.3e} cm^2/s")

print("\n===== Calculating transfer number =====")
t = analysis.transfer_number()
print(f"t+ = {t:.2f}")



Builder(
    box=args.box,
    salt=args.anion_name,
    salt_conc=args.salt_conc,
    solvents=args.solvents,
    solvent_fracs=args.solvent_fracs,
    workdir=args.workdir,

)

MDsim(
    charge_scale=args.charge_scale,
    workdir=args.workdir,
    parallel=args.parallel,
    temperature=args.temperature,
    gmx_exec=args.gromacs_exec,
)
"""


end = timer()
elapsed_time = (end - start) / 60
print(f"\nTime elapsed = {elapsed_time:.4f} minutes")

