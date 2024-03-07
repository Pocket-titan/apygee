# %%
from numpy.typing import ArrayLike
from typing import Literal, Self

import matplotlib.pyplot as plt
import numpy as np

from utils import complementary_color, darken, deep_diff, omit, shorten_fstring_number
from kepler import (
    E_from_theta,
    F_from_theta,
    M_from_E,
    M_from_F,
    barkers_equation,
    cart_to_kep,
    euler_rotation_matrix,
    hyperbolic_mean_motion,
    inverse_barkers_equation,
    kep_to_cart,
    mean_anomaly,
    mean_motion,
    orbital_distance,
    orbital_plane,
    orbital_velocity,
    eccentric_anomaly,
    hyperbolic_anomaly,
    t_from_M,
    theta_from_E,
    theta_from_F,
)


class Orbit:
    """
    A class that represents a Keplerian orbit.

    Attributes
    ----------
    kep : np.ndarray
        keplerian elements: `[a, e, i, ω, Ω, θ]`
    mu : float
        gravitational parameter
    a : float
        semi-major axis in meters. if `e=1`, `a` is instead taken as the orbital parameter `p`
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
            keplerian elements: [a, e, i, ω, Ω, θ].
            only the first element (a : semi-major axis) is required; the rest is optional and will be 0 if not provided.
        mu : float
            gravitational parameter
        """
        self.kep = np.asarray(kep).reshape(-1).astype(np.float64)

        assert self.kep.size > 0, "kep must have at least one element"
        if self.kep.size < 6:
            self.kep = np.pad(self.kep, (0, 6 - self.kep.size))

        self.mu = mu
        self._check()

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

        return f"Orbit([{', '.join(kep)}], μ={shorten_fstring_number(f"{self.mu:.2e}")}, type='{self.type}')"

    def __repr__(self) -> str:
        return str(self)

    def _check(self):
        assert self.e >= 0, "eccentricity must be non-negative"
        assert self.mu >= 0, "gravitational parameter must be non-negative"

        # In the consistent equations, hyperbolas have negative semi-major axis
        if self.type == "hyperbolic" and self.a > 0:
            self.a *= -1
        if (self.type == "elliptic" or self.type == "circular") and self.a < 0:
            self.a *= -1

    # Factory methods
    @classmethod
    def from_cart(cls, cart: ArrayLike, mu: float) -> None:
        """
        Parameters
        ----------
        cart : array_like
            cartesian state vector: [x, y, z, vx, vy, vz]
        """
        return cls(cart_to_kep(cart), mu)

    # Public methods
    def orbital_distance(self, theta: float = None) -> float:
        """
        Calculate the (scalar) distance from the central body at the given true anomaly.

        Parameters
        ----------
        theta : float, optional
            true anomaly. if `None`, uses the current value of `self.theta`

        Returns
        -------
        float
            distance from the central body
        """
        if theta is None:
            theta = self.theta

        return orbital_distance(self.h, self.mu, self.e, theta)

    def orbital_velocity(self, theta: float = None) -> float:
        """
        Calculate the (scalar) velocity at the given true anomaly.

        Parameters
        ----------
        theta : float, optional
            true anomaly. if `None`, uses the current value of `self.theta`

        Returns
        -------
        float
            velocity
        """
        if theta is None:
            theta = self.theta

        return orbital_velocity(self.h, self.mu, self.e, theta)

    def trajectory(
        self,
        thetas: ArrayLike = None,
        include_velocity=False,
    ) -> np.ndarray:
        """
        Calculate cartesian state vectors of the trajectory along the given thetas (true anomalies).

        Parameters
        ----------
        thetas : array_like, optional
            true anomalies for which to return the positions. defaults to 1000 points on [0, 2π]
        include_velocity : bool, optional
            whether to include velocity in the return value

        Returns
        -------
        cart : np.ndarray
            2d array of cartesian state vectors: [x, y, z], or [x, y, z, vx, vy, vz] if include_velocity is `True`
        """
        if thetas is None:
            if self.type == "elliptic" or self.type == "circular":
                thetas = np.linspace(0, 2 * np.pi, num=1000)
            elif self.type == "parabolic":
                thetas = np.linspace(-np.pi * 9 / 10, np.pi * 9 / 10, num=1000)
            elif self.type == "hyperbolic":
                theta_asymptote = np.arccos(-1 / self.e)
                thetas = np.linspace(-theta_asymptote, theta_asymptote, 1000)[1:-1]
        else:
            thetas = np.asarray(thetas, dtype=np.float64).reshape(-1)
        kep = np.repeat(self.kep.reshape((1, -1)), thetas.size, axis=0)
        kep[:, 5] = thetas.ravel()
        cart = kep_to_cart(kep, self.mu).reshape((-1, 6))

        if include_velocity:
            return cart

        return cart[..., :3]

    def intersects(self, orbit: Self) -> tuple[np.ndarray, np.ndarray] | Literal[False]:
        if self.type not in ["circular", "elliptic"] or orbit.type not in ["circular", "elliptic"]:
            raise NotImplementedError(
                "Intersections are only implemented for circular and elliptic orbits"
            )

        if not self.is_coplanar_with(orbit):
            print("Orbits are not coplanar")
            return False

        [a1, e1] = [self.a, self.e]
        [a2, e2] = [orbit.a, orbit.e]
        domega = orbit.omega - self.omega

        beta1 = a1 * e2 * (1 - e1**2)
        beta2 = a2 * e1 * (1 - e2**2)
        A = beta1 * e2 + beta2 * e1 - (beta1 * e1 + beta2 * e2) * np.cos(domega)
        B = e1**2 + e2**2 - 2 * e1 * e2 * np.cos(domega) - (e1 * e2 * np.sin(domega)) ** 2
        Delta = np.sin(domega) ** 2 * (
            (beta1**2 - 2 * np.cos(domega) * beta1 * beta2 + beta2**2) * (e1 * e2) ** 2
            - (beta1 * e1 - beta2 * e2) ** 2
        )

        if Delta < 0:
            return False

        rp = (A + np.sqrt(Delta)) / B
        rm = (A - np.sqrt(Delta)) / B

        if not (
            ((a1 * (1 - e1) <= rp <= a1 * (1 + e1)) or (a2 * (1 - e2) <= rp <= a2 * (1 + e2)))
            and ((a1 * (1 - e1) <= rm <= a1 * (1 + e1)) or (a2 * (1 - e2) <= rm <= a2 * (1 + e2)))
        ):
            return False

        cp = (a1 * (1 - e1**2) - rp) / (rp * e1)
        sp = (a2 * (1 - e2**2) - rp) / (rp * e2) * (1 / np.sin(domega)) - cp * (1 / np.tan(domega))

        print(np.sin(domega), np.tan(domega))

        cm = (a1 * (1 - e1**2) - rm) / (rm * e1)
        sm = (a2 * (1 - e2**2) - rm) / (rm * e2) * (1 / np.sin(domega)) - cm * (1 / np.tan(domega))

        return (rp * np.array([cp, sp, 0]), rp * np.array([cm, sm, 0]))

    def isclose(self, orbit: Self) -> bool:
        return np.isclose(self.r_vec, orbit.r_vec)

    def is_coplanar_with(self, orbit: Self) -> bool:
        """
        Parameters
        ----------
        orbit : Orbit
            orbit to compare with

        Returns
        -------
        bool
            whether the two orbits are coplanar
        """
        return np.allclose(self.orbital_plane, orbit.orbital_plane)

    def at_theta(self, theta: float) -> Self:
        """
        Parameters
        ----------
        theta : float
            true anomaly

        Returns
        -------
        Orbit
            the orbit at the given true anomaly
        """
        return Orbit([*self.kep[:5], theta], mu=self.mu)

    def at_time(self, t: float, tau: float = 0.0) -> Self:
        """
        Parameters
        ----------
        t : float
            time at which to calculate the orbital position
        tau : float, optional
            epoch of periapsis passage (`tau = 0.0` by default)

        Returns
        -------
        Orbit
            the orbit at the given time
        """
        if self.type == "circular":
            angular_velocity = np.sqrt(self.mu / self.a**3)
            return self.at_theta(self.theta + angular_velocity * t)

        n = self.mean_motion
        M = mean_anomaly(n, t, tau)

        if self.type == "elliptic":
            E = eccentric_anomaly(M, self.e)
            theta = theta_from_E(E, self.e)

        if self.type == "parabolic":
            theta = barkers_equation(M)

        if self.type == "hyperbolic":
            F = hyperbolic_anomaly(M, self.e)
            theta = theta_from_F(F, self.e)

        return self.at_theta(theta)

    def time_since(self, tau: float = 0.0) -> float:
        """
        Parameters
        ----------
        tau : float, optional
            epoch (`tau = 0.0` = periapsis by default)

        Returns
        -------
        float
            time that has passed since the given epoch
        """
        if self.type == "circular":
            angular_velocity = np.sqrt(self.mu / self.a**3)
            return self.theta / angular_velocity

        n = self.mean_motion

        if self.type == "elliptic":
            E = E_from_theta(self.theta, self.e)
            M = M_from_E(E, self.e)

        if self.type == "parabolic":
            M = inverse_barkers_equation(self.theta)

        if self.type == "hyperbolic":
            F = F_from_theta(self.theta, self.e)
            M = M_from_F(F, self.e)

        return t_from_M(M, n, tau)

    # Plotting methods
    def plot(
        self,
        theta: float = None,
        thetas: ArrayLike = None,
        ax: plt.Axes = None,
        plot_vectors=False,
        plot_foci=False,
        rc={},
        **kwargs,
    ):
        own_style = omit(
            deep_diff(plt.rcParamsDefault, plt.rcParams),
            plt.style.core.STYLE_BLACKLIST,
        )
        custom_styles = [
            "seaborn-v0_8-darkgrid",
            {"axes.spines.left": True, "axes.spines.bottom": True, "axes.linewidth": 1},
        ]

        # There's this weird behaviour where creating a new figure inside a context manager like `plt.style.context` will
        # lead subsequent `plot` calls to try to reuse the existing figure, even when inside a notebook context. Creating
        # the figure outside the context manager seems to fix this.
        if ax is None:
            fig = plt.figure()

        with plt.style.context([*custom_styles, own_style, rc]):
            if ax is None:
                ax = fig.gca()

            ax.set_aspect("equal")

            if thetas is None or np.size(thetas) > 0:
                cart = self.trajectory(thetas)

                line = ax.plot(
                    cart[:, 0],
                    cart[:, 1],
                    *([cart[:, 2]] if ax.name == "3d" else []),
                    **kwargs,
                )

                c = line[-1].get_color()
            else:
                c = (0, 1, 1)

            ax.ticklabel_format(style="scientific", axis="both", scilimits=(0, 0))

            if theta is not None:
                assert np.shape(theta) == ()
                [x, y, z] = kep_to_cart([*self.kep[:5], float(theta)], mu=self.mu)[..., :3]
                cc = complementary_color(c)

                ax.scatter([x], [y], *([[z]] if ax.name == "3d" else []), zorder=3, color=cc)
                ax.plot(
                    [0, x],
                    [0, y],
                    *([[0, z]] if ax.name == "3d" else []),
                    zorder=0,
                    color=cc,
                    ls="--",
                )

            if plot_vectors:
                [xp, yp, zp] = kep_to_cart([*self.kep[:5], 0], mu=self.mu)[:3]
                [xa, ya, za] = kep_to_cart([*self.kep[:5], np.pi], mu=self.mu)[:3]

                colors = ["#CE56DB", "#DB956B", "#555EDB", "#8ADB40", "#4BDBDB"]

                if np.abs([self.i, self.omega, self.Omega]).sum() > 0:
                    R_inv = np.linalg.inv(euler_rotation_matrix(self.i, self.omega, self.Omega))
                    asc_node = R_inv @ np.array([xp, yp, zp])

                ax.plot(
                    [0, xp],
                    [0, yp],
                    *([[0, zp]] if ax.name == "3d" else []),
                    color=(0.4, 0.4, 0.4),
                )
                ax.plot(
                    [0, xa],
                    [0, ya],
                    *([[0, za]] if ax.name == "3d" else []),
                    color=(0.5, 0.5, 0.5),
                    ls="--",
                )

                if abs(self.omega) > 0:
                    ax.plot(
                        [0, asc_node[0]],
                        [0, asc_node[1]],
                        *([[0, asc_node[2]]] if ax.name == "3d" else []),
                        color=colors[2],
                        ls="--",
                    )

                if ax.name == "3d" and abs(self.i) > 0:
                    [xa, ya, za] = kep_to_cart([*self.kep[:5], np.pi / 2], mu=self.mu)[:3]
                    ax.plot([xa, xa], [ya, ya], [za, 0], color=colors[0], ls="--")

                    ref_orbit = Orbit([self.a, self.e, 0, self.omega, self.Omega, 0], self.mu)
                    ref_plane = ref_orbit.trajectory()
                    ax.plot(ref_plane[:, 0], ref_plane[:, 1], ref_plane[:, 2], color=colors[0])

                if ax.name == "3d" and abs(self.Omega) > 0:
                    pass

            if plot_foci:
                [xf1, yf1, zf1] = kep_to_cart(
                    [self.a * self.e, 0, *self.kep[2:5], np.pi], mu=self.mu
                )[:3]
                [xf2, yf2, zf2] = kep_to_cart(
                    [2 * self.a * self.e, 0, *self.kep[2:5], np.pi], mu=self.mu
                )[:3]

                ax.scatter(
                    [xf2],
                    [yf2],
                    *([[zf2]] if ax.name == "3d" else []),
                    color=darken(c, 0.25),
                    s=30,
                    zorder=2,
                )
                ax.scatter(
                    [xf1],
                    [yf1],
                    *([[zf1]] if ax.name == "3d" else []),
                    color=c,
                    s=50,
                    zorder=2,
                )
                ax.scatter(
                    [0],
                    [0],
                    *([[0]] if ax.name == "3d" else []),
                    color=darken(c, 0.25),
                    s=30,
                    zorder=2,
                )

    def visualize(self, fig=None, **kwargs):
        import plotly.express as px
        import plotly.graph_objects as go
        from IPython.display import display
        from ipywidgets import FloatSlider, HBox, IntSlider, VBox, interactive_output

        colors = [
            "#66C2A5",
            "#FC8D62",
            "#8DA0CB",
            "#E78AC3",
            "#A6D854",
            "#FFD92F",
            "#E5C494",
            "#B3B3B3",
        ]

        if fig is None:
            fig = go.FigureWidget(layout=dict(height=500))

        def calc_orbit(kep):
            r = Orbit(kep, self.mu).trajectory()
            return dict(x=r[:, 0], y=r[:, 1], z=r[:, 2])

        def calc_position(kep):
            r = kep_to_cart(kep, self.mu)
            return dict(x=[r[0]], y=[r[1]], z=[r[2]])

        [a0, e0, i0, omega0, Omega0, theta0] = self.kep

        orbit = fig.add_trace(px.line_3d(**calc_orbit(self.kep)).data[0])

        pos0 = calc_position(self.kep)
        position = fig.add_trace(px.scatter_3d(**pos0).data[0])
        r = fig.add_trace(
            px.line_3d(x=[0, *pos0["x"]], y=[0, *pos0["y"]], z=[0, *pos0["z"]]).data[0]
        )

        orbit.data[0]["line"]["color"] = colors[0]
        position.data[1]["marker"]["color"] = colors[1]
        r.data[2]["line"]["color"] = colors[1]

        def _round(nr):
            fac = 10 ** (len(str(abs(int(nr)))) - 2)
            return np.ceil(abs(nr / fac)) * np.sign(nr) * fac

        n = 1.75
        rngs = [
            [np.min(x) * n, np.max(x) * n]
            for x in [orbit.data[0].x, orbit.data[0].y, orbit.data[0].z]
        ]
        _max = np.max(np.abs([_round(x) for x in np.ravel(rngs)]))

        fig.update_layout(
            margin=dict(l=0, r=0, b=0, t=0),
            scene=dict(
                xaxis=dict(range=[-_max, _max]),
                yaxis=dict(range=[-_max, _max]),
                zaxis=dict(range=[-_max, _max]),
                aspectratio=dict(x=1, y=1, z=1),
            ),
            xaxis_fixedrange=True,
            yaxis_fixedrange=True,
        )

        fig.update_traces(line=dict(width=5), marker=dict(size=10))

        a = FloatSlider(
            value=a0,
            min=a0 / 2,
            max=2 * a0,
            step=a0 / 100,
            description="a",
            readout_format=".2e",
        )
        e = FloatSlider(value=e0, min=0, max=1, step=0.01, description="e")
        i = IntSlider(value=np.rad2deg(i0), min=0, max=360, step=1, description="i")
        omega = IntSlider(value=np.rad2deg(omega0), min=0, max=360, step=1, description="ω")
        Omega = IntSlider(value=np.rad2deg(Omega0), min=0, max=360, step=1, description="Ω")
        theta = IntSlider(value=np.rad2deg(theta0), min=0, max=360, step=1, description="θ")

        ui_left = VBox([a, e, i])
        ui_right = VBox([omega, Omega, theta])
        ui = HBox([ui_left, ui_right])

        def update(
            a=a0,
            e=e0,
            i=np.rad2deg(i0),
            omega=np.rad2deg(omega0),
            Omega=np.rad2deg(Omega0),
            theta=np.rad2deg(theta0),
        ):
            with fig.batch_update():
                [i, omega, Omega, theta] = np.deg2rad([i, omega, Omega, theta])

                orb = calc_orbit([a, e, i, omega, Omega, theta])
                orbit.data[0].x = orb["x"]
                orbit.data[0].y = orb["y"]
                orbit.data[0].z = orb["z"]

                pos = calc_position([a, e, i, omega, Omega, theta])
                position.data[1].x = pos["x"]
                position.data[1].y = pos["y"]
                position.data[1].z = pos["z"]

                r.data[2].x = [0, *pos["x"]]
                r.data[2].y = [0, *pos["y"]]
                r.data[2].z = [0, *pos["z"]]

        _out = interactive_output(
            update,
            dict(
                a=a,
                e=e,
                i=i,
                omega=omega,
                Omega=Omega,
                theta=theta,
            ),
        )

        display(ui, fig)

    # Transfer methods
    def impulsive_shot(
        self,
        dv0: float,
        delta0: float = None,
        gamma1: float = None,
        theta: float = None,
    ) -> Self:
        """
        Parameters
        ----------

        Returns
        -------

        Raises
        ------
        """
        if delta0 is None and gamma1 is None:
            raise ValueError("Either `d0` or `gamma1` must be specified.")
        if theta is None:
            theta = self.theta

        r0 = self.orbital_distance(theta)
        v0 = self.orbital_velocity(theta)

        if gamma1 is None:
            gamma1 = np.arctan(dv0 * np.sin(delta0) / (v0 + dv0 * np.cos(delta0)))
        if delta0 is None:
            delta0 = np.arcsin(v0 / dv0 * np.sin(gamma1)) + gamma1

        v1 = np.sqrt(v0**2 + dv0**2 + 2 * v0 * dv0 * np.cos(delta0))
        eps = v1**2 / 2 - self.mu / r0
        h = r0 * v1 * np.cos(gamma1)
        p = h**2 / self.mu

        a = -self.mu / (2 * eps)
        e = np.sqrt(1 - p / a)

        if e > 1:
            a *= -1

        # NOTE: how mathematically justified is clipping here?
        theta0 = np.arccos(np.clip(1 / e * (a * (1 - e**2) / r0 - 1), 0, 1))
        return Orbit([a, e, *self.kep[2:5], theta0], mu=self.mu)

    def coplanar_transfer(self, orbit: Self, theta_dep: float, theta_arr: float) -> Self:
        """
        If `theta_arr` > `theta_dep`, the transfer orbit is prograde. Otherwise, it's retrograde.
        Example: setting `theta_dep` to π and `theta_arr` to 0 will result in a *retrograde* transfer orbit.
        Setting `theta_dep` to π and `theta_arr` to 2π will result in a *prograde* transfer orbit.

        Parameters
        ----------

        Returns
        -------

        Raises
        ------
        """
        if not self.is_coplanar_with(orbit):
            raise Exception("Orbits are not coplanar")
        if any([x.type not in ["circular", "elliptic"] for x in [self, orbit]]):
            raise Exception("Both orbits must be circular or elliptic")
        if not np.isclose(self.mu, orbit.mu):
            raise Exception("Gravitational parameters must be equal")

        omega = theta_dep
        dtheta = theta_arr - theta_dep
        r_arr = orbit.orbital_distance(theta_arr)

        if orbit.a > self.a:
            # Transferring from inner to outer orbit
            rp = self.orbital_distance(theta_dep)
            e = (rp - r_arr) / (r_arr * np.cos(dtheta) - rp)
            p = rp * (1 + e)
            ra = p / (1 - e)
        else:
            # Transferring from outer to inner orbit
            ra = self.orbital_distance(theta_dep)
            e = (ra - r_arr) / (r_arr * np.cos(dtheta + np.pi) + ra)
            p = ra * (1 - e)
            rp = p / (1 + e)
            omega += np.pi

        a = (ra + rp) / 2

        return Orbit([a, e, self.i, omega, self.Omega, theta_dep], mu=self.mu)

    def hohmann_transfer(self, orbit: Self) -> Self:
        """
        Parameters
        ----------
        orbit : Orbit
            orbit to transfer to

        Returns
        -------
        Orbit
            the transfer ellipse

        Raises
        ------
        Exception
            if the orbits are not coplanar, or not either circular or elliptic

        See Also
        --------
        coplanar_transfer
        """
        if orbit.a > self.a:
            return self.coplanar_transfer(orbit, 0, np.pi)

        return self.coplanar_transfer(orbit, np.pi, 2 * np.pi)

    def biellptic_transfer(self, orbit: Self, ra: float) -> tuple[Self, Self]:
        """
        Parameters
        ----------

        Returns
        -------

        Raises
        ------

        See Also
        --------
        hohmann_transfer
        """
        assert orbit.a > self.a, "Orbit must be transferring from inner to outer orbit"
        assert ra > orbit.ra > self.ra, "`ra` must be larger than both orbits' apoapses"

        rp = self.rp
        a = (rp + ra) / 2
        e = (ra - rp) / (ra + rp)
        first_transfer = Orbit([a, e, *self.kep[2:6]], mu=self.mu)
        second_transfer = first_transfer.coplanar_transfer(orbit, np.pi, 2 * np.pi)

        return (first_transfer, second_transfer)

    # Computed properties
    @property
    def type(self) -> str:
        "{'circular', 'elliptic', 'parabolic', 'hyperbolic'} : type of orbit"
        if np.isclose(self.e, 0):
            return "circular"
        if np.isclose(self.e, 1):
            return "parabolic"
        if 0 <= self.e < 1:
            return "elliptic"
        if self.e > 1:
            return "hyperbolic"
        raise Exception("Unknown orbit type, should never happen!")

    @property
    def h_vec(self) -> np.ndarray:
        "np.ndarray : specific angular momentum vector"
        cart = kep_to_cart(self.kep, self.mu)
        r = cart[..., :3]
        v = cart[..., 3:]
        return np.cross(r, v, axisa=-1, axisb=-1, axisc=-1)

    @property
    def e_vec(self) -> np.ndarray:
        "np.ndarray : eccentricity vector"
        cart = kep_to_cart(self.kep, self.mu)
        r = cart[..., :3]
        v = cart[..., 3:]
        return (
            np.cross(v, self.h_vec, axisa=-1, axisb=-1, axisc=-1) / self.mu
        ) - r / np.linalg.norm(r, axis=-1, keepdims=True)

    @property
    def h(self) -> float:
        "float : specific angular momentum"
        return np.linalg.norm(self.h_vec, axis=-1)

    @property
    def p(self) -> float:
        "float : semi-latus rectum"
        return self.h**2 / self.mu

    @property
    def rp(self) -> float:
        "float : periapsis distance"
        return self.p / (1 + self.e)

    @property
    def ra(self) -> float:
        "float : apoapsis distance"
        return self.p / (1 - self.e)

    @property
    def r_vec(self) -> np.ndarray:
        "np.ndarray : position vector"
        return kep_to_cart(self.kep, self.mu)[..., :3]

    @property
    def r(self) -> float:
        "float : position magnitude"
        return self.p / (1 + self.e * np.cos(self.theta))

    @property
    def v_vec(self) -> np.ndarray:
        "np.ndarray : velocity vector"
        return kep_to_cart(self.kep, self.mu)[..., 3:]

    @property
    def v(self) -> float:
        "float : velocity magnitude"
        return np.sqrt(self.mu * (2 / self.r - 1 / self.a))

    @property
    def T(self) -> float:
        "float : orbital period"
        if self.type in ["circular", "elliptic"]:
            return 2 * np.pi * np.sqrt(self.a**3 / self.mu)
        if self.type == "parabolic":
            return np.sqrt(16 / 9 * self.p**3 / self.mu)

        return np.inf

    @property
    def orbital_plane(self) -> np.ndarray:
        "np.ndarray : orbital plane normal vector"
        return orbital_plane(self.r_vec, self.v_vec)

    @property
    def mean_motion(self) -> float:
        "float : mean motion"
        if self.type in ["circular", "elliptic"]:
            return mean_motion(self.a, self.mu)
        if self.type == "hyperbolic":
            return hyperbolic_mean_motion(self.a, self.mu)
        return np.inf

    # Properties
    @property
    def a(self) -> float:
        "float : semi-major axis"
        return self.kep[0]

    @a.setter
    def a(self, value: float) -> None:
        self.kep[0] = value

    @property
    def e(self) -> float:
        "float : eccentricity"
        return self.kep[1]

    @e.setter
    def e(self, value: float) -> None:
        self.kep[1] = value

    @property
    def i(self) -> float:
        "float : inclination"
        return self.kep[2]

    @i.setter
    def i(self, value: float) -> None:
        self.kep[2] = value

    @property
    def omega(self) -> float:
        "float : argument of periapsis"
        return self.kep[3]

    @omega.setter
    def omega(self, value: float) -> None:
        self.kep[3] = value

    @property
    def Omega(self) -> float:
        "float : longitude of the ascending node"
        return self.kep[4]

    @Omega.setter
    def Omega(self, value: float) -> None:
        self.kep[4] = value

    @property
    def theta(self) -> float:
        "float : true anomaly"
        return self.kep[5]

    @theta.setter
    def theta(self, value: float) -> None:
        self.kep[5] = value


from scipy.optimize import differential_evolution

orbit1 = Orbit([10_000, 0.9, 0, 0.4], mu=3.98e14)
orbit2 = Orbit([15_000, 0, 0, 0], mu=3.98e14)


def func(x):
    r1 = orbit1.at_theta(x[0]).r_vec
    r2 = orbit2.at_theta(x[1]).r_vec
    return np.linalg.norm(r1 - r2)


res = differential_evolution(func, bounds=[(0, 2 * np.pi), (0, 2 * np.pi)])
res

# %%

ax = plt.gca()
orbit1.plot(ax=ax, theta=3.386, label="Orbit 1")
orbit2.plot(ax=ax, theta=2.898, label="Orbit 2")
ax.legend()

# %%
orbit1.intersects(orbit2)

# %%
orbit = Orbit([20_000e3, 1.2], mu=3.98e14).at_theta(np.pi / 2)
orbit.time_since(), orbit.T / 2

F = F_from_theta(orbit.theta, orbit.e)
M = M_from_F(F, orbit.e)
t_from_M(M, hyperbolic_mean_motion(orbit.a, orbit.mu), 0)

# %%
n = hyperbolic_mean_motion(orbit.a, orbit.mu)
M = mean_anomaly(n, 778.4340677759866, 0)
hyperbolic_anomaly(M, orbit.e)
# %%
orbit1 = Orbit([10_000, 0, 0.6, 0, 0, 0], mu=3.98e14)
orbit2 = Orbit([20_000, 0, 0.6, 0, 0, 0], mu=3.98e14)

theta_dep = 0
theta_arr = np.pi

transfer = orbit1.coplanar_transfer(orbit2, theta_dep, theta_arr)

ax = plt.gca()
orbit1.plot(ax=ax, label="Orbit 1")
orbit2.plot(ax=ax, label="Orbit 2")
transfer.plot(thetas=np.linspace(theta_dep, theta_arr, 100), ax=ax, label="Transfer")

[x, y, _z] = orbit1.at_theta(theta_dep).r_vec
ax.scatter([x], [y], color="r", label="Departure")

# ax.legend()
# %%
[transfer1, transfer2] = orbit1.biellptic_transfer(orbit2, 30_000)

ax = plt.gca()
orbit1.plot(ax=ax, label="Orbit 1")
orbit2.plot(ax=ax, label="Orbit 2")
transfer1.plot(ax=ax, thetas=np.linspace(0, np.pi, 100), label="Transfer 1")
transfer2.plot(ax=ax, thetas=np.linspace(np.pi, 2 * np.pi, 100), label="Transfer 2")

# %%
orbit = Orbit([10_000, 1.2], mu=3.98e14)
n = hyperbolic_mean_motion(orbit.a, orbit.mu)
M = mean_anomaly(n, 1000, 0)
hyperbolic_anomaly(M, orbit.e)

# %%
orbit = Orbit([10_000, 0.7], mu=3.98e14)
n = mean_motion(orbit.a, orbit.mu)
M = mean_anomaly(n, 1000, 0)
eccentric_anomaly(M, orbit.e)
