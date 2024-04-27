import pytest
import numpy as np

from itertools import product
from keppy.constants import MU_EARTH
from keppy.orbit import Orbit
from keppy.kepler import cart_to_kep, kep_to_cart


def test_conversion(orbit):
    kep = orbit.kep
    print("kep: ", kep)
    cart = np.concatenate([orbit.r_vec, orbit.v_vec])
    print("cart: ", cart)
    kep2 = cart_to_kep(cart, mu=orbit.mu)
    print("kep2: ", kep2)
    orbit2 = Orbit(kep2, mu=orbit.mu)
    cart2 = np.concatenate([orbit2.r_vec, orbit2.v_vec])
    print("cart2: ", cart2)
    print(np.abs(cart - cart2).sum())
    # assert np.allclose(kep, kep2)
    assert np.allclose(cart, cart2, atol=1e-6)


# @pytest.mark.parametrize(
#     "x,y,z,vx,vy,vz,mu",
#     [
#         [1400e3, 1400e3, 0, -10e3, 10e3, 0, MU_EARTH],  # circular
#         [-700e3, 700e3, 0, -23e3, 6e3, 0, MU_EARTH],  # elliptic
#         [-5000e3, 5000e3, 1100e3, -10e3, 4e3, 1e3, MU_EARTH],  # parabolic
#         [-15000e3, 80000e3, 5800e3, 1.3e3, -10e3, -0.6e3, MU_EARTH],  # hyperbolic
#     ],
# )
# def test_cart_to_kep(x, y, z, vx, vy, vz, mu):
#     cart = np.array([x, y, z, vx, vy, vz])
#     print(cart)
#     kep = cart_to_kep(cart, mu=mu)
#     print(kep)
#     cart2 = kep_to_cart(kep, mu=mu)
#     print(cart2)
#     assert np.allclose(cart, cart2)


# @pytest.mark.parametrize(
#     "a,e,i,Omega,omega,theta,mu",
#     product(
#         [2000e3],
#         [0, 0.5, 1, 1.5],
#         [0, np.pi / 2, np.pi],
#         [0, np.pi / 2, np.pi, 2 * np.pi],
#         [0, np.pi / 2, np.pi],
#         [0, np.pi / 2, np.pi],
#         [MU_EARTH],
#     ),
# )
# def test_kep_to_cart(a, e, i, Omega, omega, theta, mu):
#     if np.isclose(e, 1) and np.isclose(theta, np.pi):
#         with pytest.raises(AssertionError):
#             kep = np.array([a, e, i, Omega, omega, theta])
#             cart = kep_to_cart(kep, mu=mu)
#             kep2 = cart_to_kep(cart, mu=mu)
#             assert np.allclose(kep[1:3], kep2[1:3])
#         return

#     kep = np.array([a, e, i, Omega, omega, theta])
#     cart = kep_to_cart(kep, mu=mu)
#     kep2 = cart_to_kep(cart, mu=mu)
#     print(kep)
#     assert np.allclose(kep[:3], kep2[:3])

#     # angles = np.mod(np.sum(kep[2:], axis=-1), 2 * np.pi)
#     # angles2 = np.mod(np.sum(kep2[2:], axis=-1), 2 * np.pi)
#     # assert np.allclose(angles, angles2)
