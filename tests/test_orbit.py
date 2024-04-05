import numpy as np

from keppy.orbit import Orbit
from keppy.constants import MU_EARTH


def test_circular():
    orbit = Orbit([10_000], mu=MU_EARTH)
    radii = np.fromiter(
        map(
            lambda x: orbit.orbital_distance(x),
            np.array([0, 1 / 4, 1 / 2, 3 / 4, 1, 3 / 2, 2, 3, 4, 4.5]) * np.pi,
        ),
        dtype=np.float64,
    )
    assert np.allclose(radii - 10_000, 0)
