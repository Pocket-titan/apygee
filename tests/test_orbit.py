import numpy as np
import pytest

from itertools import product
from keppy.constants import MU_EARTH
from keppy.orbit import Orbit


def test_circular(a, theta):
    orbit = Orbit([a, 0], mu=MU_EARTH)
    assert np.isclose(orbit.orbital_distance(theta), a)


@pytest.mark.parametrize(
    "dv,x",
    product([1000, 10_000], np.array([0, 0.5, 1, 1.5, 2]) * np.pi),
)
def test_impulsive_shot(theta, dv, x):
    orbit = Orbit([2e6], mu=MU_EARTH)
    transfer = orbit.impulsive_shot(dv=dv, x=x, theta=theta)
    # print(orbit)
    # print(transfer)
    # print(transfer.r_vec)
    # print(orbit.at_theta(theta).r_vec)
    res = np.mod(transfer.theta - transfer.omega + theta, np.pi)
    print(np.isclose(res, 0) or np.isclose(res, np.pi))
    assert np.allclose(transfer.r_vec, orbit.at_theta(theta).r_vec)
