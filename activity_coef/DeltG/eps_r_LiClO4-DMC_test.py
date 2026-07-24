# file: plot_epsr_liclo4_dmc.py

import numpy as np
import matplotlib.pyplot as plt


EPS_SOLV = 3.1075

A = -9.53
B = -11.87
D = 5.0
E = 0.43


def eps_r(c: np.ndarray) -> np.ndarray:
    return (
        EPS_SOLV
        - A * c
        + B * c**2
        + D * c**3
        - E * c**4
    )


c = np.linspace(0.0, 2.0, 500)
eps = eps_r(c)

# extracted Figure 3a data
c_exp = np.array([
    0.09626508015204127,
    0.49912861886089455,
    0.9991361307672662,
    1.502223524285993,
    1.9957107014618165,
])

eps_exp = np.array([
    3.7596340199215845,
    5.43298628325897,
    5.3849100824807365,
    5.625591562626759,
    7.307657637355199,
])

print("eps_r(1 M) =", eps_r(np.array([1.0]))[0])
print("eps_r(2 M) =", eps_r(np.array([2.0]))[0])

plt.figure(figsize=(7, 5))

plt.plot(c, eps, label="Eq. 7 fit")
plt.scatter(c_exp, eps_exp, label="Extracted Fig. 3a data")

plt.xlabel("LiClO4 concentration [mol kg$^{-1}$]")
plt.ylabel(r"$\varepsilon_r$")
plt.title("LiClO4 in DMC")

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()