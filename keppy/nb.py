# %%
import numpy as np
from keppy.orbit import Orbit
from keppy.plot import AngleAnnotation
from keppy.kepler import angle_between, kep_to_cart, cart_to_kep
from keppy.utils import rotate_vector, scale_vector
from keppy.plot import plot_vector, plot_angle
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
    MU_EARTH,
)

import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=True)

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
plt.scatter(0, 0, s=100)

theta_dep = np.pi / 4
theta_arr = 8 * np.pi / 5

earth.plot(theta=theta_dep)
mars.plot(theta=theta_arr)

transfer = earth.coplanar_transfer(mars, theta_dep, theta_arr)
transfer.plot(thetas=np.linspace(theta_dep, theta_arr, num=100) - transfer.omega)

transfer.plot(
    thetas=np.linspace(theta_arr, theta_dep + 2 * np.pi) - transfer.omega,
    ls="--",
)


# %%
# Impulsive shot
plt.scatter(0, 0, s=100)

earth.plot()

transfer1 = earth.impulsive_shot(dv=[2000, 10_000, 4000], theta=0.0)
transfer1.plot()

transfer2 = earth.impulsive_shot(dv=10_000, x=np.pi / 2, theta=np.pi / 2)
transfer2.plot()


# %%
theta = 1.1780972450961724
dv = 1000
x = 3.141592653589793
orbit = Orbit([2e6], mu=MU_EARTH)

transfer = orbit.impulsive_shot(dv=dv, x=x, theta=theta)
orbit.plot(theta=theta * 2, show=["v"])
transfer.plot(theta=transfer.theta, show="v")

print(transfer.r_vec)
print(orbit.at_theta(theta).r_vec)


# %%
fig = plt.figure()
ax = fig.add_subplot()
# ax = fig.add_subplot(projection="3d")

orbit = Orbit([10_000, 0.3, 0.9, 2.2, np.pi / 2], mu=MU_EARTH)
orbit.plot(theta=np.pi / 2)
# %%
