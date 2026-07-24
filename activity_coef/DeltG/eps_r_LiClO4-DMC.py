# file: plot_epsr_liclo4_dmc.py

import numpy as np
import matplotlib.pyplot as plt


def eps_r(
    c: np.ndarray,
    eps_solv: float = 3.1075, #1.6
    A: float = -9.53,
    B: float = -11.87,
    D: float = 5.20,
    E: float = 0.43,
) -> np.ndarray:
    """
    Relative permittivity from Eq. 7 for LiClO4 in DMC.

    Parameters
    ----------
    c : np.ndarray
        Salt concentration [mol/L].
    eps_solv : float
        Static permittivity of pure DMC at 298 K.
    A, B, D, E : float
        Fit parameters from Supplementary Table 10.

    Returns
    -------
    np.ndarray
        Relative permittivity ε_r.
    """
    return eps_solv - A * c + B * c**2 + D * c**3 - E * c**4


def main() -> None:
    c = np.linspace(0.0, 2.0, 500)
    eps = eps_r(c)

    plt.figure(figsize=(7, 5))
    plt.plot(c, eps, linewidth=2)

    plt.xlabel("LiClO₄ concentration [mol/L]")
    plt.ylabel("Relative permittivity ε_r")
    plt.title("ε_r of LiClO₄ in DMC (Eq. 7, Table 10)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()