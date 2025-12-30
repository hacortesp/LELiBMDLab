from __future__ import annotations

import os
import subprocess
from pathlib import Path


class GROMACSrun:
    """
    Write and execute GROMACS runs (EM → NPT eq → NPT production).
    """

    def __init__(
        self,
        workdir: str | Path,
        temperature: float,
        *,
        parallel: bool = False,
        gmx_exec: str = "gmx",
        gmx_mpi_exec: str = "gmx_mpi",
        ntasks_env: str = "SLURM_NTASKS",
    ) -> None:
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)

        self.temp = temperature
        self.parallel = parallel
        self.gmx_exec = gmx_exec
        self.gmx_mpi_exec = gmx_mpi_exec
        self.ntasks_env = ntasks_env

    # ========================================================
    # Public API
    # ========================================================

    def write_all(self) -> None:
        self.write_em_mdp()
        self.write_npt_eq_mdp()
        self.write_nvt_prod_mdp()

    def run(self) -> None:
        commands = (
            self._commands_parallel()
            if self.parallel
            else self._commands_local()
        )
        self._run_commands(commands)

    # ========================================================
    # Command builders
    # ========================================================

    def _commands_local(self) -> list[str]:
        return [
            f"{self.gmx_exec} grompp -f em.mdp -c conf.pdb -p topol.top -o em.tpr",
            f"{self.gmx_exec} mdrun -v -deffnm em",

            f"{self.gmx_exec} grompp -f npt_eq.mdp -c em.gro -p topol.top -o npt_eq.tpr -maxwarn 1",
            f"{self.gmx_exec} mdrun -deffnm npt_eq",

            self._command_extract_volume("npt_eq.edr", "volume.xvg"),

            f"{self.gmx_exec} grompp -f nvt_prod.mdp -c npt_eq.gro -p topol.top -o nvt_prod.tpr -maxwarn 1",
            f"{self.gmx_exec} mdrun -deffnm nvt_prod",
        ]

    def _commands_parallel(self) -> list[str]:
        ntasks = os.environ.get(self.ntasks_env)
        if ntasks is None:
            raise EnvironmentError(f"{self.ntasks_env} not set")

        return [
            f"mpirun -np 1 {self.gmx_mpi_exec} grompp -f em.mdp -c conf.pdb -p topol.top -o em.tpr",
            f"mpirun -np {ntasks} {self.gmx_mpi_exec} mdrun -deffnm em",

            f"mpirun -np 1 {self.gmx_mpi_exec} grompp -f npt_eq.mdp -c em.gro -p topol.top -o npt_eq.tpr -maxwarn 1",
            f"mpirun -np {ntasks} {self.gmx_mpi_exec} mdrun -deffnm npt_eq",

            self._command_extract_volume("npt_eq.edr", "volume.xvg"),

            f"mpirun -np 1 {self.gmx_mpi_exec} grompp -f npt_data.mdp -c npt_eq.gro -p topol.top -o npt_data.tpr -maxwarn 1",
            f"mpirun -np {ntasks} {self.gmx_mpi_exec} mdrun -deffnm npt_data",
        ]

    # ========================================================
    # Execution
    # ========================================================

    def _run_commands(self, commands: list[str]) -> None:
        for cmd in commands:
            print(f"\n>>> {cmd}")
            subprocess.run(
                cmd,
                shell=True,
                check=True,
                cwd=self.workdir,
            )

    def _command_extract_volume(
        self,
        edr_file: str = "npt_eq.edr",
        output_file: str = "volume.xvg",
    ) -> str:
        gmx = self.gmx_mpi_exec if self.parallel else self.gmx_exec
        return (
            f"echo 21 | {gmx} energy "
            f"-f {edr_file} -o {output_file}"
    )


    def write_em_mdp(self) -> None:
        self._write(
            "em.mdp",
            """\
integrator      = steep
emtol           = 1000.0
emstep          = 0.01
nsteps          = 50000

cutoff-scheme   = Verlet
ns_type         = grid
nstlist         = 1
coulombtype     = PME
rcoulomb        = 1.0
rvdw            = 1.0
pbc             = xyz
""",
        )

    def write_npt_eq_mdp(self) -> None:
        self._write(
            "npt_eq.mdp",
            f"""\
integrator              = md          
dt                      = 0.002       
nsteps                  = 1000 ;2000000     
tinit                   = 0

nstxout                 = 0           
nstvout                 = 0           
nstlog                  = 1000        
nstcalcenergy           = 100         
nstenergy               = 1000        
nstxout-compressed      = 2000        
compressed-x-precision  = 1000

cutoff-scheme           = Verlet
nstlist                 = 10          
ns_type                 = grid
rlist                   = 1.4

coulombtype             = PME
rcoulomb                = 1.4
vdwtype                 = Cut-off
rvdw                    = 1.4
DispCorr                = EnerPres    
constraints             = all-bonds
constraint-algorithm    = Lincs
lincs-order             = 4
lincs-iter              = 1

pbc                     = xyz

tcoupl                  = v-rescale
tc-grps                 = System      
tau-t                   = 0.1
ref-t                   = {self.temp}         

Pcoupl                  = Berendsen
Pcoupltype              = isotropic
tau-p                   = 1.0
compressibility         = 5e-5
ref-p                   = 1.0

gen_vel                 = yes
gen_temp                = {self.temp}
gen_seed                = -1
continuation            = no         
""",
        )

    def write_nvt_prod_mdp(self) -> None:
        self._write(
            "nvt_prod.mdp",
            f"""\
integrator              = md
dt                      = 0.002      ; 2 fs
nsteps                  = 1000    ; 6000000 12 ns

nstlog                  = 1000
nstcalcenergy           = 100
nstenergy               = 1000
nstxout                 = 0
nstvout                 = 0
nstxout-compressed      = 2000
compressed-x-precision  = 1000

cutoff-scheme           = Verlet
nstlist                 = 10
ns_type                 = grid
rlist                   = 1.4

coulombtype             = PME
rcoulomb                = 1.4
vdwtype                 = Cut-off
rvdw                    = 1.4
DispCorr                = EnerPres

pbc                     = xyz

tcoupl                  = v-rescale
tc-grps                 = System
tau-t                   = 0.1
ref-t                   = {self.temp}

Pcoupl                  = no
gen-vel                 = no

constraints             = all-bonds
constraint-algorithm    = Lincs
lincs-order             = 4
lincs-iter              = 1
""",
        )

    def _write(self, filename: str, content: str) -> None:
        path = self.workdir / filename
        path.write_text(content.strip() + "\n")
        print(f"Wrote {path}")
