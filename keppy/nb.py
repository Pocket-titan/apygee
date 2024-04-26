# %%
import numpy as np
from keppy.orbit import Orbit
from keppy.plot import AngleAnnotation
from keppy.kepler import angle_between, kep_to_cart, cart_to_kep, euler_rotation_matrix
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
orbit.plot(theta=theta, show=["r"])
transfer.plot(show=["r"])

v1 = orbit.at_theta(theta).v_vec
v2 = transfer.at_theta(transfer.theta).v_vec

print(v1, v2)
# assert np.isclose(np.linalg.norm(v2 - v1), dv)

# transfer.at_theta(theta - transfer.omega).r_vec, orbit.at_theta(theta).r_vec
# print(transfer.theta), (theta - transfer.omega), 2* np.pi


# %%
h_vec = np.array([0, 0, 1e3])

K = np.array([0, 0, 1])
N_vec = np.cross(K, h_vec)
N = np.linalg.norm(N_vec, axis=-1)

N_vec, N

orbit.e_vec, orbit.e

np.arccos(0)

# %%

kep = np.array([2000000.0, 1.2, np.pi / 2, 2 * np.pi, 0.0, 0])
orbit = Orbit(kep, mu=MU_EARTH)
orbit.plot(theta=np.pi * 0, thetas=np.linspace(-np.pi * 4 / 10, np.pi * 4 / 10, num=10))


# %%
cart = kep_to_cart(kep, mu=MU_EARTH)
plt.scatter(cart[0], cart[1])

# %%
kep2 = cart_to_kep2(cart, mu=MU_EARTH)
orbit2 = Orbit(kep2, mu=MU_EARTH)

orbit2


# convert_cartesian_to_keplerian_elements(cart, MU_EARTH)


# %%
import numpy as np
from keppy.kepler import dot


def cart_to_kep2(cart, mu):
    shape = np.shape(cart)
    cart = np.asarray(cart, dtype=np.float64).reshape((-1, 6))
    eps = np.finfo(np.float64).eps

    if isinstance(mu, (int, float)):
        mu = np.full((cart.shape[0],), mu, dtype=np.float64)
    else:
        mu = np.asarray(mu, dtype=np.float64).ravel()

    r_vec = cart[:, :3]
    v_vec = cart[:, 3:]
    r = np.linalg.norm(r_vec, axis=-1)

    h_vec = np.cross(r_vec, v_vec, axis=-1)
    h = np.linalg.norm(h_vec, axis=-1)
    p = h**2 / mu

    K = np.array([0, 0, 1])
    N_vec = np.cross(K, h_vec, axis=-1)
    N = np.linalg.norm(N_vec, axis=-1)

    # Eccentricity
    e_vec = np.cross(v_vec, h_vec, axis=-1) / mu[:, None] - r_vec / r[:, None]
    e = np.linalg.norm(e_vec, axis=-1)
    par = np.abs(e - 1.0) < eps
    circ = np.abs(e) < eps

    # Semi-major axis
    a = np.zeros_like(p)
    a[par] = p
    a[~par] = p[~par] / (1 - e[~par] ** 2)

    # Inclination
    i = np.arccos(h_vec[:, -1] / h)
    equa = np.abs(i) < eps
    N_vec[equa] = np.array([1, 0, 0])
    N[equa] = 1.0

    # Right ascension of the ascending node
    Omega = np.arccos(N_vec[:, 0] / N)
    Omega[N_vec[:, 1] < 0] = 2 * np.pi - Omega[N_vec[:, 1] < 0]

    # Argument of periapsis
    omega = np.zeros_like(e)
    omega[circ] = 0.0
    omega[~circ] = np.arccos(
        np.clip(dot(e_vec[~circ] / e[~circ, None], N_vec[~circ] / N[~circ, None]), -1, 1)
    )
    iy = ~circ & equa & (N_vec[~circ & equa, 1] < 0)
    iz = ~circ & ~equa & (N_vec[~circ & ~equa, 2] < 0)
    omega[iy | iz] = 2 * np.pi - omega[iy | iz]

    # True anomaly
    e_vec[circ] = N_vec[circ]
    theta = np.arccos(
        np.clip(dot(r_vec / r[:, None], e_vec / np.linalg.norm(e_vec, axis=-1)), -1, 1)
    )
    ii = np.isclose(N_vec, [1, 0, 0], atol=eps).all(axis=-1)
    iy = circ & ii & (r_vec[:, 1] < 0)
    iz = circ & ~ii & (r_vec[:, 2] < 0)
    ih = ~circ & (h_vec[:, 2] < 0)
    theta[iy | iz | ih] = 2 * np.pi - theta[iy | iz | ih]

    kep = np.stack([a, e, i, Omega, omega, theta], axis=-1)
    return kep.reshape(shape)


def convert_cartesian_to_keplerian_elements(cart, mu):
    eps = 20.0 * np.finfo(float).eps
    kep = np.zeros(7)

    r_vec = cart[:3]
    v_vec = cart[3:]

    h_vec = np.cross(r_vec, v_vec)
    p = np.linalg.norm(h_vec) ** 2 / mu

    k = np.array([0, 0, 1])
    n = np.cross(k, h_vec)
    n /= np.linalg.norm(n)

    # Eccentricity
    e_vec = np.cross(v_vec, h_vec) / mu - r_vec / np.linalg.norm(r_vec)
    e = np.linalg.norm(e_vec)
    kep[0] = e

    # Semimajor axis
    if np.abs(e - 1.0) < eps:
        a = p
    else:
        a = p / (1 - e**2)
    kep[1] = a

    # Inclination
    i = np.arccos(h_vec[2] / np.linalg.norm(h_vec))
    kep[3] = i

    argument_of_periapsis_quandrant_condition = e_vec[2]

    if np.abs(i) < eps:
        n = np.array([1, 0, 0])
        argument_of_periapsis_quandrant_condition = e_vec[1]

    # Right ascension of the ascending node
    Omega = np.arccos(n[0])
    if n[1] < 0:
        Omega = 2 * np.pi - Omega
    kep[4] = Omega

    true_anomaly_quandrant_condition = np.dot(r_vec, v_vec)

    if np.abs(e) < eps:
        e_vec = n
        kep[5] = 0.0

        if np.all(n == np.array([1, 0, 0])):
            true_anomaly_quandrant_condition = r_vec[1]
        else:
            true_anomaly_quandrant_condition = r_vec[2]
    else:
        edotn = np.dot(e_vec / np.linalg.norm(e_vec), n)

        if edotn < -1:
            kep[5] = np.pi
        elif edotn > 1:
            kep[5] = 0.0
        else:
            kep[5] = np.arccos(edotn)

        if argument_of_periapsis_quandrant_condition < 0:
            kep[5] = 2 * np.pi - kep[5]

    rdote = np.dot(r_vec / np.linalg.norm(r_vec), e_vec / np.linalg.norm(e_vec))

    if np.abs(1 - rdote) < eps:
        rdote = 1.0
    elif np.abs(1 + rdote) < eps:
        rdote = -1.0
    elif np.abs(rdote) < eps:
        rdote = 0.0

    kep[6] = np.arccos(rdote)

    if true_anomaly_quandrant_condition < 0:
        kep[6] = 2 * np.pi - kep[6]

    return kep
