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
        "-equil",
        type=bool,
        default=True,
        help="Enable equilibration before production.",
    )    
    parser.add_argument(
        "-dt",
        type=float,
        default=0.002,
        help="Timestep in ps.",
    )
    parser.add_argument(
        "-nvt_time_ps",
        type=int,
        default=30000,
        help="Production time in ps",
    )
    parser.add_argument(
        "-dt_collection",
        type=float,
        default=500,
        help="Data collection interval",
    )

    return parser.parse_args()

args = parse_args()

start = timer()


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
    equilibration=args.equil,
    temperature=args.temperature,
    gmx_exec=args.gromacs_exec,
    nvt_time_ps=args.nvt_time_ps,
    time_step_ps=args.dt,
)
"""


analysis = TrajAnalysis(
    workdir=args.workdir,
    dt=args.dt,
    dt_collection=args.dt_collection,
    temperature=args.temperature,
    cation_name=args.cation_name,
    anion_name=args.anion_name,
    q_eff=args.charge_scale,
)


print("\n===== Calculating permittivity =====")
epsilon = analysis.permittivity(
    run_start = 29990, 
    run_end = 30000,
    trajdir = args.workdir,
)
print(f"Dielectric constant = {epsilon:.4f}\n")



print("\n===== Calculating conductivity =====")
cond, err = analysis.conductivity_split()
print(f"Ionic conductivity = {cond:.4f} ± {err:.4f} mS/cm\n")


print("\n===== Calculating transfer number =====")
t, err = analysis.transfer_number_split()
print(f"t+ = {t:.4f}  ± {err:.4f}")

print("\n===== Calculating permittivity =====")
epsilon = analysis.permittivity(
    run_start = 29500, 
    run_end = 30000,
    trajdir = args.workdir,
)
print(f"Dielectric constant = {epsilon:.4f}\n")

print("\n===== Calculating activity =====")
gamma_DH, gamma_B, gamma_DH_B = analysis.activity(
    run_start = 17000, 
    run_end = 18000,
    solv_dir="/home/hacortes/Li-EMDLab/1MLiClO4_DMC_T298K/solvent"
)
print(f"Results {gamma_DH}, {gamma_B}, {gamma_DH_B}\n")


print("\n===== Calculating solvatation structure =====")
df = analysis.coordination_type(
    run_start = 29500, 
    run_end = 30000, 
    distance = 4.075, 
    center_atom = args.cation_name, 
    counter_atom = args.anion_name
)
print("Writing files in ion_association folder")
print("Writing files in solvatation_structure folder")
print("Consider:")
print("*Solvent-separated ion pairs (SSIP)")
print("*Contact ion pairs (CIP)")
print("*Aggregate (AGG)")
for _, row in df.iterrows():
    print(f"{row['Solvation structure'].upper():>4s}  = {row['Percentage']}")


print("\n===== Calculating coordination =====")
distance, coord = analysis.coordination_number(args.cation_name, args.anion_name)
print(
    f"{args.cation_name}–{args.anion_name}, "
    f"coordination number: {float(coord):.3f}, "
    f"distance: {float(distance):.3f} Å"
)
for solvent in args.solvents:
    distance, coord = analysis.coordination_number(args.cation_name, solvent)
    print(
        f"{args.cation_name}–{solvent}, "
        f"coordination number: {float(coord):.3f}, "
        f"distance: {float(distance):.3f} Å"
    )
print("Writing files in coordination_files folder")


print("\n===== Calculating ion association =====")
analysis.ion_cluster_population(
    run_start = 29500, 
    run_end = 30000, 
    center_atom = args.cation_name, 
    counter_atom = args.anion_name,
    distance=4.075,
    core = 2,
)
"""


end = timer()
elapsed_time = (end - start) / 60
print(f"\nTime elapsed = {elapsed_time:.4f} minutes")

