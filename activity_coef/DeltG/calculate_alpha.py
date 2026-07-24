from math import sqrt

experimental_eps = {
    "DEC": 2.80,
    "DMC": 3.10,
    "DME": 7.20,
    "EC": 89.00,
    "EMC": 2.90,
    "PC": 64.90,
}

md_eps = {
    "DEC": 1.34,
    "DMC": 1.58,
    "DME": 8.93,
    "EC": 102.95,
    "EMC": 1.45,
    "PC": 66.36,
}


def calculate_alpha(eps_exp: float, eps_md: float) -> float:
    if eps_md <= 1:
        raise ValueError(
            f"eps_md must be > 1, got {eps_md}"
        )

    return sqrt(
        (eps_exp - 1.0)
        / (eps_md - 1.0)
    )


print(
    f"{'Solvent':<6} "
    f"{'eps_exp':>10} "
    f"{'eps_md':>10} "
    f"{'alpha':>10}"
)

print("-" * 40)

for solvent in experimental_eps:

    alpha = calculate_alpha(
        experimental_eps[solvent],
        md_eps[solvent],
    )

    print(
        f"{solvent:<6} "
        f"{experimental_eps[solvent]:10.2f} "
        f"{md_eps[solvent]:10.2f} "
        f"{alpha:10.4f}"
    )