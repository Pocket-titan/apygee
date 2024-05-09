import pytest
import numpy as np
from apygee.constants import MU_EARTH
from apygee.orbit import Orbit
from itertools import product


@pytest.fixture(autouse=True, scope="session")
def setup():
    np.set_printoptions(precision=3, suppress=True)


def kep_isclose(kep1: np.array, kep2: np.array) -> bool:
    [a1, e1, i1, Omega1, omega1, theta1] = kep1
    [a2, e2, i2, Omega2, omega2, theta2] = kep2

    if not np.isclose(e1, e2, atol=1e-9):
        print(f"{e1 = }, {e2 = }")
        return False

    if e1 > 1 and e2 > 1:
        [a1, a2] = -np.abs([a1, a2])

    if not np.isclose(a1, a2, atol=1e-6):
        print(f"{a1 = }, {a2 = }")
        return False

    # If not planar/equatorial
    if not np.isclose(i1, [0, np.pi, 2 * np.pi]).any():
        if not np.isclose(i1, i2, atol=1e-9):
            print(f"{i1 = }, {i2 = }")
            return False

        if not np.isclose(np.mod(Omega1, 2 * np.pi), np.mod(Omega2, 2 * np.pi), atol=1e-9):
            print(f"{Omega1 = }, {Omega2 = }")
            return False

        if not np.isclose(np.mod(omega1 + theta1, 2 * np.pi), np.mod(omega2 + theta2, 2 * np.pi)):
            print(f"{omega1 = }, {omega2 = }")
            print(f"{theta1 = }, {theta2 = }")
            return False

    return True


def cart_isclose(cart1: np.array, cart2: np.array) -> bool:
    if not np.allclose(cart1[:3], cart2[:3], atol=1e-6):
        return False

    if not np.allclose(cart1[3:], cart2[3:], atol=1e-6):
        return False

    return True


# All kepler elements (selection of good edge cases)
def filter_kep(kep: list[float]) -> bool:
    [a, e, i, Omega, omega, theta] = kep

    # No apoapsis for parabolic or hyperbolic orbits
    if e >= 1 and theta == np.pi:
        return False

    return True


@pytest.fixture(
    params=filter(
        filter_kep,
        product(
            aa := np.linspace(1, 100, num=2) * 2e6,
            ee := [0, 1, 1.5],
            ii := np.array([0, 0.5, 1]) * np.pi,
            OO := np.array([0, 0.5, 1, 1.5, 2]) * np.pi,
            oo := np.array([0, 0.5, 1, 1.5, 2]) * np.pi,
            tt := np.array([0, 0.5, 1, 1.5, 2]) * np.pi,
        ),
    ),
    scope="session",
)
def kep(request):
    return request.param


# With Orbit wrapper
@pytest.fixture(scope="session")
def orbit(kep, mu):
    return Orbit(kep, mu=mu)


# Individual kepler elements (higher fidelity)
@pytest.fixture(params=np.linspace(1, 1000, num=10) * 2e6, scope="session")
def a(request):
    return request.param


@pytest.fixture(params=np.linspace(0, 2, num=10), scope="session")
def e(request):
    return request.param


@pytest.fixture(params=np.arange(0, 8 + 1) / 8 * np.pi, scope="session")
def i(request):
    return request.param


@pytest.fixture(params=np.arange(0, 16 + 1) / 8 * np.pi, scope="session")
def Omega(request):
    return request.param


@pytest.fixture(params=np.arange(0, 16 + 1) / 8 * np.pi, scope="session")
def omega(request):
    return request.param


@pytest.fixture(params=np.arange(0, 16 + 1) / 8 * np.pi, scope="session")
def theta(request):
    return request.param


# All cartesian elements (selection of good edge cases)
def filter_cart(cart: list[float]) -> bool:
    [x, y, z, vx, vy, vz] = cart

    if not np.count_nonzero(cart[:3]) >= 2:
        return False

    if not np.count_nonzero(cart[3:]) >= 2:
        return False

    if (np.abs(cart[:3]) == 0).all():
        return False

    if (np.abs(cart[3:]) > 0).all():
        return False

    if vx == vy or vx == vz or vy == vz:
        return False

    return True


cart_cases = [
    [-767008.0, -324286.0, -284495.0, -15923.0, -6732.0, -28396.0],  # hyperbolic
    [-1654945.0, -798742.0, -789409.0, 7850.0, -6849.0, -9528.0],  # circular
    [-1818595.0, 832294.0, -0.0, -6962.0, 18712.0, -0.0],  # parabolic
    [-3000000.0, 0.0, -0.0, -0.0, -0.0, 8151.0],  # elliptic
    [-2500000.0, 0.0, 4330127.0, -0.0, -11811.0, 0.0],  # elliptic
    [-10000000.0, 0.0, 17320508.0, -0.0, -7732.0, 0.0],  # hyperbolic
]


@pytest.fixture(
    params=filter(
        filter_cart,
        [
            *cart_cases,
            *product(
                xx := np.concatenate([[0], np.linspace(2000, 40_000, num=3)]) * 1e3,
                yy := np.concatenate([[0], np.linspace(2000, 40_000, num=3)]) * 1e3,
                zz := np.concatenate([[0], np.linspace(2000, 40_000, num=3)]) * 1e3,
                vx := np.concatenate([[0], np.linspace(15, 50, num=3)]) * 1e3,
                vy := np.concatenate([[0], np.linspace(15, 50, num=3)]) * 1e3,
                vz := np.concatenate([[0], np.linspace(15, 50, num=3)]) * 1e3,
            ),
        ],
    ),
    scope="session",
)
def cart(request):
    return request.param


# Individual cartesian elements (higher fidelity)
@pytest.fixture(params=np.linspace(2000, 40_000, num=10) * 1e3, scope="session")
def x(request):
    return request.param


@pytest.fixture(params=np.linspace(2000, 40_000, num=10) * 1e3, scope="session")
def y(request):
    return request.param


@pytest.fixture(params=np.linspace(2000, 40_000, num=10) * 1e3, scope="session")
def z(request):
    return request.param


@pytest.fixture(params=np.linspace(10, 1000, num=10) * 1e3, scope="session")
def vx(request):
    return request.param


@pytest.fixture(params=np.linspace(10, 1000, num=10) * 1e3, scope="session")
def vy(request):
    return request.param


@pytest.fixture(params=np.linspace(10, 1000, num=10) * 1e3, scope="session")
def vz(request):
    return request.param


# Gravitational parameter
@pytest.fixture(params=[MU_EARTH], scope="session")
def mu(request):
    return request.param
