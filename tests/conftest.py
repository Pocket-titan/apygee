import pytest
import numpy as np
from keppy.constants import (
    MU_SUN,
    MU_EARTH,
    MU_VENUS,
    MU_MERCURY,
    MU_MARS,
    MU_JUPITER,
    MU_SATURN,
    MU_URANUS,
    MU_NEPTUNE,
)
from keppy.orbit import Orbit
from itertools import product

np.set_printoptions(precision=3, suppress=True)


as_ = np.linspace(1, 100, num=2) * 2e6
es = [0, 1, 1.5]
is_ = np.array([0, 0.5, 1]) * np.pi
Os = np.array([0, 0.5, 1, 1.5, 2]) * np.pi
os = np.array([0, 0.5, 1, 1.5, 2]) * np.pi
ts = np.array([0, 0.5, 1, 1.5, 2]) * np.pi
mus = [MU_EARTH]


# Orbit
@pytest.fixture(params=product(as_, es, is_, Os, os, ts, mus))
def orbit(request):
    return Orbit(request.param[:-1], mu=request.param[-1])


# Kepler elements
@pytest.fixture(params=np.linspace(1, 1000, num=10) * 2e6)
def a(request):
    return request.param


@pytest.fixture(params=np.linspace(0, 2, num=10))
def e(request):
    return request.param


@pytest.fixture(params=np.arange(0, 8 + 1) / 8 * np.pi)
def i(request):
    return request.param


@pytest.fixture(params=np.arange(0, 16 + 1) / 8 * np.pi)
def Omega(request):
    return request.param


@pytest.fixture(params=np.arange(0, 16 + 1) / 8 * np.pi)
def omega(request):
    return request.param


@pytest.fixture(params=np.arange(0, 16 + 1) / 8 * np.pi)
def theta(request):
    return request.param


# Cartesian elements
@pytest.fixture(params=np.linspace(2000, 40_000, num=2) * 1e3)
def x(request):
    return request.param


@pytest.fixture(params=np.linspace(2000, 40_000, num=2) * 1e3)
def y(request):
    return request.param


@pytest.fixture(params=np.linspace(2000, 40_000, num=2) * 1e3)
def z(request):
    return request.param


@pytest.fixture(params=np.linspace(10, 1000, num=2) * 1e3)
def vx(request):
    return request.param


@pytest.fixture(params=np.linspace(10, 1000, num=2) * 1e3)
def vy(request):
    return request.param


@pytest.fixture(params=np.linspace(10, 1000, num=2) * 1e3)
def vz(request):
    return request.param


# Gravitational parameter
@pytest.fixture(
    params=[
        MU_SUN,
        MU_EARTH,
        MU_MERCURY,
        MU_JUPITER,
    ]
)
def mu(request):
    return request.param
