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
def convert_cartesian_to_kep(cart, mu):
    eps = 20.0 * np.finfo(float).eps
    kep = np.zeros(6)

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
    kep[1] = e

    # Semimajor axis
    if np.abs(e - 1.0) < eps:
        a = p
    else:
        a = p / (1 - e**2)
    kep[0] = a

    # Inclination
    i = np.arccos(h_vec[2] / np.linalg.norm(h_vec))
    kep[2] = i

    argument_of_periapsis_quandrant_condition = e_vec[2]

    if np.abs(i) < eps:
        n = np.array([1, 0, 0])
        argument_of_periapsis_quandrant_condition = e_vec[1]

    # Right ascension of the ascending node
    Omega = np.arccos(n[0])
    if n[1] < 0:
        Omega = 2 * np.pi - Omega
    kep[3] = Omega

    true_anomaly_quandrant_condition = np.dot(r_vec, v_vec)

    if np.abs(e) < eps:
        e_vec = n
        omega = 0.0

        if np.all(n == np.array([1, 0, 0])):
            true_anomaly_quandrant_condition = r_vec[1]
        else:
            true_anomaly_quandrant_condition = r_vec[2]
    else:
        edotn = np.dot(e_vec / np.linalg.norm(e_vec), n)

        if edotn < -1:
            omega = np.pi
        elif edotn > 1:
            omega = 0.0
        else:
            omega = np.arccos(edotn)

        if argument_of_periapsis_quandrant_condition < 0:
            omega = 2 * np.pi - omega

    kep[4] = omega

    rdote = np.dot(r_vec / np.linalg.norm(r_vec), e_vec / np.linalg.norm(e_vec))

    if np.abs(1 - rdote) < eps:
        rdote = 1.0
    elif np.abs(1 + rdote) < eps:
        rdote = -1.0
    elif np.abs(rdote) < eps:
        rdote = 0.0

    kep[5] = np.arccos(rdote)

    if true_anomaly_quandrant_condition < 0:
        kep[5] = 2 * np.pi - kep[5]

    return kep


def cart_to_kep_orb(cart, mu, degrees=False, e_lim=1e-9, i_lim=np.pi * 1e-9):
    if not isinstance(cart, np.ndarray):
        raise TypeError("Input type {} not supported: must be {}".format(type(cart), np.ndarray))
    if cart.shape[0] != 6:
        raise ValueError(
            f"Input data must have at least 6 variables along axis 0:\
             input shape is {cart.shape}"
        )

    if len(cart.shape) < 2:
        input_is_vector = True
        try:
            cart.shape = (6, 1)
        except ValueError as e:
            print(f"Error {e} while trying to cast vector into single column.")
            print(f"Input array shape: {cart.shape}")
            raise
    else:
        input_is_vector = False

    ez = np.array([0, 0, 1], dtype=cart.dtype)

    o = np.empty(cart.shape, dtype=cart.dtype)

    r = cart[:3, :]
    v = cart[3:, :]
    rn = np.linalg.norm(r, axis=0)
    vn = np.linalg.norm(v, axis=0)

    vr = np.sum((r / rn) * v, axis=0)

    epsilon = vn**2 * 0.5 - mu / rn

    # ## ECCENTRICITY ###
    e = 1.0 / mu * ((vn**2 - mu / rn) * r - (rn * vr) * v)
    o[1, :] = np.linalg.norm(e, axis=0)

    # ## SEMI MAJOR AXIS ###
    # possible cases
    # e < 1
    # and
    # e >= 1
    e_hyp = o[1, :] >= 1
    o[0, :] = -mu / (2.0 * epsilon)
    o[0, e_hyp] = -o[0, e_hyp]

    # ## ANUGLAR MOMENTUM ###
    h = np.cross(r, v, axisa=0, axisb=0, axisc=0)
    hn = np.linalg.norm(h, axis=0)
    o[2, :] = np.arccos(h[2, :] / hn)

    # possible cases
    eg = o[1, :] > e_lim  # e grater
    # i grater (include retrograde planar orbits)
    ig = np.logical_and(o[2, :] > i_lim, o[2, :] < np.pi - i_lim)
    el = np.logical_not(eg)  # e less equal
    il = np.logical_not(ig)  # i less equal

    # e > elim & i > ilim
    eg_ig = np.logical_and(eg, ig)

    # e > elim & i <= ilim
    eg_il = np.logical_and(eg, il)

    # e <= elim & i > ilim
    el_ig = np.logical_and(el, ig)

    # e <= elim & i <= ilim
    el_il = np.logical_and(el, il)

    # ## ASCENDING NODE ###
    # ascending node pointing vector
    n = np.empty_like(h)
    nn = np.empty_like(hn)
    n[:, ig] = np.cross(ez, h[:, ig], axisa=0, axisb=0, axisc=0)
    nn[ig] = np.linalg.norm(n[:, ig], axis=0)

    # ensure [0,2pi]
    ny_neg = np.logical_and(n[1, :] < 0.0, ig)
    o[4, ig] = np.arccos(n[0, ig] / nn[ig])
    o[4, ny_neg] = 2.0 * np.pi - o[4, ny_neg]

    # non inclined: no ascending node
    o[4, il] = 0

    # ## ARGUMENT OF PERIAPSIS ###
    # circular orbits: no argument of periapsis
    o[3, el] = 0

    # elliptical and hyperbolic orbits
    # two cases
    cos_om = np.empty_like(hn)
    # first case: eg and ig (n-vector)
    # use vector angle between the two
    cos_om[eg_ig] = np.sum(
        n[:, eg_ig] * e[:, eg_ig],
        axis=0,
    ) / (nn[eg_ig] * o[1, eg_ig])

    # second case: eg and il (no n-vector)
    # use e vector angle
    cos_om[eg_il] = e[0, eg_il] / o[1, eg_il]

    # remove unused array positions
    cos_om = cos_om[eg]

    # do not fail due to number precision fluctuation
    cos_om[cos_om > 1.0] = 1.0
    cos_om[cos_om < -1.0] = -1.0

    o[3, eg] = np.arccos(cos_om)

    # first case: e and n vector angle
    ez_neg = np.logical_and(e[2, :] < 0.0, eg_ig)
    o[3, ez_neg] = 2.0 * np.pi - o[3, ez_neg]

    # second case: ex component
    # prograde
    print(o[2, :])
    ey_neg = np.logical_and(o[2, :] < np.pi * 0.5, eg_il)
    ey_neg2 = np.logical_and(ey_neg, e[1, :] < 0.0)
    o[3, ey_neg2] = 2.0 * np.pi - o[3, ey_neg2]

    # retrograde
    ey_neg = np.logical_and(o[2, :] > np.pi * 0.5, eg_il)
    ey_neg2 = np.logical_and(ey_neg, e[1, :] >= 0.0)
    o[3, ey_neg2] = 2.0 * np.pi - o[3, ey_neg2]

    # ## TRUE ANOMALY ###
    cos_nu = np.empty_like(hn)

    # three cases
    # elliptical and hyperbolic: (angle from periapsis using e and r)
    cos_nu[eg] = np.sum(e[:, eg] * r[:, eg], axis=0) / (o[1, eg] * rn[eg])

    # circular and inclined: (angle from periapsis using n and r)
    # if e=0 and omega := 0, with inclination +y -> +z perihelion is ascending
    # node
    cos_nu[el_ig] = np.sum(
        (n[:, el_ig] / nn[el_ig]) * (r[:, el_ig] / rn[el_ig]),
        axis=0,
    )

    # circular and planar: (use angle of position vector)
    cos_nu[el_il] = r[0, el_il] / rn[el_il]

    # do not fail due to number precision fluctuation
    cos_nu[cos_nu > 1.0] = 1.0
    cos_nu[cos_nu < -1.0] = -1.0

    o[5, :] = np.arccos(cos_nu)

    # ensure [0,2pi]
    # elliptical and hyperbolic
    tmp_ind_ = np.logical_and(vr < 0.0, eg)
    o[5, tmp_ind_] = 2.0 * np.pi - o[5, tmp_ind_]

    # circular and inclined
    tmp_ind_ = np.logical_and(r[2, :] < 0.0, el_ig)
    o[5, tmp_ind_] = 2.0 * np.pi - o[5, tmp_ind_]

    # circular and planar
    # prograde
    tmp_ind_ = np.logical_and(o[2, :] < np.pi * 0.5, el_il)
    tmp_ind2_ = np.logical_and(tmp_ind_, r[1, :] < 0.0)
    o[5, tmp_ind2_] = 2.0 * np.pi - o[5, tmp_ind2_]

    # if retrograde, its reversed
    tmp_ind_ = np.logical_and(o[2, :] > np.pi * 0.5, el_il)
    tmp_ind2_ = np.logical_and(tmp_ind_, r[1, :] >= 0.0)
    o[5, tmp_ind2_] = 2.0 * np.pi - o[5, tmp_ind2_]
    print("o:", o[5, :])

    tmp = o[3, :].copy()
    o[3, :] = o[4, :]
    o[4, :] = tmp

    # # OUTPUT FORMATTING ##
    if degrees:
        o[2:, :] = np.degrees(o[2:, :])

    if input_is_vector:
        cart.shape = (6,)
        o.shape = (6,)

    return o


def compute_semi_latus_rectum(e, a, eps):
    return a * (1 - e**2)


def convert_keplerian_to_cartesian_elements(kep, mu):
    eps = np.finfo(float).eps

    a = kep[0]
    e = kep[1]
    i = kep[2]
    Omega = kep[3]
    omega = kep[4]
    theta = kep[5]

    cosi = np.cos(i)
    sini = np.sin(i)
    coso = np.cos(omega)
    sino = np.sin(omega)
    cosO = np.cos(Omega)
    sinO = np.sin(Omega)
    costheta = np.cos(theta)
    sintheta = np.sin(theta)

    p = compute_semi_latus_rectum(e, a, eps)

    position_perifocal = np.array(
        [
            p * costheta / (1 + e * costheta),
            p * sintheta / (1 + e * costheta),
        ]
    )

    velocity_perifocal = np.array(
        [
            -np.sqrt(mu / p) * sintheta,
            np.sqrt(mu / p) * (e + costheta),
        ]
    )

    R = np.array(
        [
            [
                cosO * coso - sinO * sino * cosi,
                -cosO * sino - sinO * coso * cosi,
            ],
            [
                sinO * coso + cosO * sino * cosi,
                -sinO * sino + cosO * coso * cosi,
            ],
            [
                sino * sini,
                coso * sini,
            ],
        ]
    )

    cart = np.zeros(6)

    cart[:3] = np.dot(R, position_perifocal)
    cart[3:] = np.dot(R, velocity_perifocal)

    return cart


# %%
from keppy.kepler import cart_to_kep, kep_to_cart

kep = np.array([2000000.0, 0, 0, 0.0, np.pi, 2 * np.pi])
orbit = Orbit(kep, mu=MU_EARTH)


kep = orbit.kep
print("kep: ", kep)
cart = np.concatenate([orbit.r_vec, orbit.v_vec])
print("cart: ", cart)
kep2 = cart_to_kep(cart, mu=orbit.mu)
print("kep2: ", kep2)
kep3 = convert_cartesian_to_kep(cart, mu=orbit.mu)
print("kep3: ", kep3)
kep4 = cart_to_kep_orb(cart, mu=orbit.mu)
print("kep4: ", kep4)
orbit2 = Orbit(kep2, mu=orbit.mu)
cart2 = np.concatenate([orbit2.r_vec, orbit2.v_vec])
print("cart2: ", cart2)


orbit.plot(theta=orbit.theta)
plt.scatter(cart[0], cart[1])

orbit2.plot(theta=orbit2.theta)
plt.scatter(cart2[0], cart2[1])

# %%
cart = kep_to_cart([2e6, 0, 0, 0, 0, 3 * np.pi / 2], mu=MU_EARTH)
kep = cart_to_kep(cart, mu=MU_EARTH)

# %%
np.array([0.9, 0, 0]) == [1, 0, 0]

# %%
np.abs(cart - cart2).sum()
