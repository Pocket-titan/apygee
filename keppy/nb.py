# %%
import numpy as np
from keppy.orbit import Orbit
from keppy.kepler import angle_between
from keppy.utils import plot_vector, rotate_vector, scale_vector
from keppy.constants import (
    MERCURY,
    VENUS,
    EARTH,
    MARS,
    JUPITER,
    SATURN,
    URANUS,
    NEPTUNE,
    MU_SUN,
)

import matplotlib.pyplot as plt

np.set_printoptions(precision=2, suppress=True)

custom_styles = [
    "seaborn-v0_8-darkgrid",
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": (0.2, 0.2, 0.2),
        "axes.linewidth": 1,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
    },
]

plt.style.use([*custom_styles])

earth = Orbit([EARTH.a], mu=MU_SUN)
mars = Orbit([MARS.a], mu=MU_SUN)

# %%
# Plotting the solar system
plt.scatter(0, 0, s=10)

MERCURY.plot()
VENUS.plot()
EARTH.plot()
MARS.plot()
JUPITER.plot()
SATURN.plot()
URANUS.plot()
NEPTUNE.plot()


# %%
# Hohmann transfer from Earth to Mars
ax = plt.subplot()
ax.scatter(0, 0, s=100)

earth.plot(theta=0, ax=ax, plot_velocity=True)

mars.plot(theta=np.pi, ax=ax, plot_velocity=True)

hohmann = earth.hohmann_transfer(mars)
hohmann.plot(thetas=np.linspace(0, np.pi, num=100), ax=ax)

dt = (hohmann.T / 2) / 3600 / 24
print(f"Δt = {dt:.2f} days")

dv = np.linalg.norm(hohmann.at_theta(0).v_vec - earth.at_theta(0).v_vec) / 1000
print(f"Δv = {dv:.2f} km/s")

hohmann_dv = dv
# %%
# Bielliptic (or double Hohmann) transfer from Earth to Mars
ax = plt.subplot()
ax.scatter(0, 0, s=100)

earth.plot(theta=0, ax=ax)
mars.plot(ax=ax)

[transfer1, transfer2] = earth.bielliptic_transfer(mars, mars.ra * 1.5)
transfer1.plot(theta=np.pi, thetas=np.linspace(0, np.pi, num=100), ax=ax)
transfer2.plot(theta=2 * np.pi, thetas=np.linspace(np.pi, 2 * np.pi, num=100), ax=ax)

# %%
# Coplanar transfer
ax = plt.subplot()
ax.scatter(0, 0, s=100)

theta_dep = np.pi / 4
theta_arr = 8 * np.pi / 5

earth.plot(theta=theta_dep, ax=ax)
mars.plot(theta=theta_arr, ax=ax)

transfer = earth.coplanar_transfer(mars, theta_dep, theta_arr)
transfer.plot(thetas=np.linspace(theta_dep, theta_arr, num=100) - transfer.omega, ax=ax)

transfer.plot(
    thetas=np.linspace(theta_arr, theta_dep + 2 * np.pi) - transfer.omega,
    ls="--",
    ax=ax,
)

# %%
# Impulsive shot
ax = plt.subplot()
ax.scatter(0, 0, s=100)

earth.plot(ax=ax)

transfer = earth.impulsive_shot(dv=[1000, 1000, 0], theta=0)
transfer.plot(ax=ax)

# %%
ax = plt.gca()
plot_vector(earth.at_theta(0).v_vec)
plot_vector(transfer.at_theta(0).v_vec)


# %%
theta = 0

orbit = Orbit([100_000_000_000, 0.2, 0.3, 0.2, 0.8], mu=MU_SUN)
transfer = orbit.impulsive_shot(4000, x=np.pi / 2)

# orbit.plot(theta=theta, plot_velocity=True)
transfer.plot(plot_velocity=True)

# %%
Orbit.from_cart([10000, 10000, 10000, 1, 1, 0], mu=MU_SUN)
# %%
