import argparse
from timeit import default_timer as timer
from LEMDLab.SimulationCell import Builder
from LEMDLab.MDSimulation import MDsim


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse input arguments for the builder script."
    )

    parser.add_argument(
        "-box",
        nargs=3,
        type=float,
        default=[40.0, 40.0, 40.0],
        metavar=("Lx", "Ly", "Lz"),
        help="Simulation box dimensions in Å (default: 40 40 40).",
    )

    parser.add_argument(
        "-salt",
        type=str,
        default="LiPF6",
        help="Salt name (default: LiPF6).",
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
        default=["EC", "DMC"],
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
        default=False,
        help="Whether to run GROMACS in parallel (default: False).",
    )
    return parser.parse_args()


args = parse_args()

start = timer()

#constructor DOES the work
Builder(
    box=args.box,
    salt=args.salt,
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


end = timer()
elapsed_time = (end - start) / 60
print(f"\nTime elapsed = {elapsed_time:.4f} minutes")

