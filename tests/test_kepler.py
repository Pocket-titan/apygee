import pytest
import numpy as np

from itertools import product
from keppy.constants import MU_EARTH
from keppy.kepler import cart_to_kep, kep_to_cart


def test_cart_one():
    cart = np.array([10, 1, 1, 2, 2, 2])
    kep = cart_to_kep(cart, mu=1)
    cart2 = kep_to_cart(kep, mu=1)
    assert np.allclose(cart, cart2)


def test_cart_two():
    cart = np.array([[1, 10, 1, 2, 2, 2], [1, 1, 5, 2, 2, 2]])
    kep = cart_to_kep(cart, mu=1)
    cart2 = kep_to_cart(kep, mu=1)
    assert np.allclose(cart, cart2)


def test_kep_one():
    kep = np.array([10, 0.2, np.pi / 7, 0, 0, np.pi / 4])
    cart = kep_to_cart(kep, mu=1)
    kep2 = cart_to_kep(cart, mu=1)
    assert np.allclose(kep, kep2)


def test_kep_two():
    kep = np.array(
        [
            [10, 0.2, np.pi / 7, 0, 0, np.pi / 4],
            [15, 0.95, np.pi / 2, 0, 0, np.pi / 9],
        ]
    )
    cart = kep_to_cart(kep, mu=1)
    kep2 = cart_to_kep(cart, mu=1)
    assert np.allclose(kep, kep2)


@pytest.mark.parametrize(
    "x,y,z,vx,vy,vz,mu",
    [
        [1400e3, 1400e3, 0, -10e3, 10e3, 0, MU_EARTH],  # circular
        [-700e3, 700e3, 0, -23e3, 6e3, 0, MU_EARTH],  # elliptic
        [-5000e3, 5000e3, 1100e3, -10e3, 4e3, 1e3, MU_EARTH],  # parabolic
        [-15000e3, 80000e3, 5800e3, 1.3e3, -10e3, -0.6e3, MU_EARTH],  # hyperbolic
    ],
)
def test_cart_to_kep(x, y, z, vx, vy, vz, mu):
    cart = np.array([x, y, z, vx, vy, vz])
    kep = cart_to_kep(cart, mu=mu)
    cart2 = kep_to_cart(kep, mu=mu)
    assert np.allclose(cart, cart2)


@pytest.mark.parametrize(
    "a,e,i,Omega,omega,theta,mu",
    product(
        [2000e3],
        [0, 0.5, 1, 1.5],
        [0, np.pi / 2, np.pi],
        [0, np.pi / 2, np.pi, 2 * np.pi],
        [0, np.pi / 2, np.pi],
        [0, np.pi / 2, np.pi],
        [MU_EARTH],
    ),
)
def test_kep_to_cart(a, e, i, Omega, omega, theta, mu):
    kep = np.array([a, e, i, Omega, omega, theta])
    print(kep)
    cart = kep_to_cart(kep, mu=mu)
    kep2 = cart_to_kep(cart, mu=mu)
    print(kep2)
    kep[2] = np.mod(kep[2], 2 * np.pi)
    kep[3] = np.mod(kep[3], 2 * np.pi)
    kep[4] = np.mod(kep[4], np.pi)
    kep[5] = np.mod(kep[5], 2 * np.pi)
    assert np.allclose(kep, kep2)
