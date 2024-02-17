import numpy as np
from numpy.typing import ArrayLike
from utils import shorten_fstring_number


class Orbit:
    """
    A class that represents a Keplerian orbit.

    Attributes
    ----------
    kep : np.ndarray
        keplerian elements: [a, e, i, ω, Ω, θ]
    mu : float
        gravitational parameter
    a : float
        semi-major axis
    e : float
        eccentricity
    i : float
        inclination
    omega : float
        argument of periapsis
    Omega : float
        longitude of the ascending node
    theta : float
        true anomaly
    """

    kep: np.ndarray

    def __init__(self, kep: ArrayLike, mu: float) -> None:
        """
        Parameters
        ----------
        kep : array_like
            keplerian elements: [a, e, i, ω, Ω, θ]
        mu : float
            gravitational parameter
        """
        self.kep = np.asarray(kep).reshape(-1).astype(np.float64)

        if self.kep.shape[0] != 6:
            self.kep = np.pad(self.kep, (0, 6 - self.kep.shape[0]))

        self.mu = mu

    def __str__(self) -> str:
        kep = map(
            shorten_fstring_number,
            [
                f"a={self.a:.2e}",
                f"e={self.e:.2f}",
                f"i={self.i:.2f}",
                f"ω={self.omega:.2f}",
                f"Ω={self.Omega:.2f}",
                f"θ={self.theta:.2f}",
            ],
        )

        return f"Orbit([{', '.join(kep)}], μ={shorten_fstring_number(f"{self.mu:.2e}")})"

    def __repr__(self) -> str:
        return str(self)

    @property
    def a(self) -> float:
        "float : semi-major axis"
        return self.kep[0]

    @property
    def e(self) -> float:
        "float : eccentricity"
        return self.kep[1]

    @property
    def i(self) -> float:
        "float : inclination"
        return self.kep[2]

    @property
    def omega(self) -> float:
        "float : argument of periapsis"
        return self.kep[3]

    @property
    def Omega(self) -> float:
        "float : longitude of the ascending node"
        return self.kep[4]

    @property
    def theta(self) -> float:
        "float : true anomaly"
        return self.kep[5]
