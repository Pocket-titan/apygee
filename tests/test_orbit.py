import numpy as np
import pytest

from itertools import product
from keppy.orbit import Orbit


def test_orbit(orbit):
    rv1 = np.concatenate([orbit.at_theta(0).r_vec, orbit.at_theta(0).v_vec])
    rv2 = np.concatenate([orbit.at_theta(2 * np.pi).r_vec, orbit.at_theta(2 * np.pi).v_vec])
    assert np.allclose(rv1, rv2, atol=1e-6)


def test_circular(a, theta, mu):
    orbit = Orbit([a, 0], mu=mu)
    assert np.isclose(orbit.orbital_distance(theta), a)


@pytest.mark.parametrize(
    "dv,x",
    product([1000, 10_000], np.array([0, 0.5, 1, 1.5, 2]) * np.pi),
)
def test_impulsive_shot(theta, dv, x, mu):
    orbit = Orbit([2e6], mu=mu)
    transfer = orbit.impulsive_shot(dv=dv, x=x, theta=theta)

    assert np.allclose(transfer.r_vec, orbit.at_theta(theta).r_vec)

    v1 = orbit.at_theta(theta).v_vec
    v2 = transfer.at_theta(transfer.theta).v_vec
    assert np.isclose(np.linalg.norm(v2 - v1), dv)


def test_time(orbit):
    if orbit.type in ["parabolic", "hyperbolic"]:
        return

    assert np.isclose(orbit.at_theta(0).at_time(orbit.T / 2).theta, np.pi, atol=1e-6)
    assert np.isclose(orbit.at_theta(np.pi).time_since(0), orbit.T / 2, atol=1e-6)
