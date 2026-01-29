import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.dielectric import DielectricConstant


def calc_permittivity(
    universe,
    temperature,
    atom_selection,
    start=None,
    stop=None,
    make_whole=True
):

    # Resolve atom group
    if atom_selection is None:
        atomgroup = universe.atoms
    elif isinstance(atom_selection, str):
        atomgroup = universe.select_atoms(atom_selection)
    else:
        atomgroup = atom_selection

    if atomgroup.n_atoms == 0:
        raise ValueError("Atom selection resulted in an empty AtomGroup.")

    diel = DielectricConstant(
        atomgroup,
        temperature=temperature,
        make_whole=make_whole
    )

    diel.run(start=start, stop=stop)

    # Return scalar dielectric constant
    return diel.results.eps_mean




 