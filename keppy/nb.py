# %%
import numpy as np
from keppy.orbit import Orbit
from keppy.kepler import angle_between
from keppy.utils import plot_vector, rotate_vector, scale_vector, plot_angle
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
orbit.plot(theta=theta)
transfer.plot(theta=transfer.theta)

print(transfer.r_vec)
print(orbit.at_theta(theta).r_vec)
# %%
orbit = Orbit([10_000, 0.3, 0.2, 0.4, 0.9, 1.2], mu=MU_EARTH)
orbit.plot(theta=np.pi / 2, plot_vectors=True)


# %%
def plot_2d_angle():
    pass


def plot_2d_vector():
    pass


def plot_3d_angle():
    pass


def plot_3d_vector():
    pass


# %%
from numpy.typing import ArrayLike
from keppy.kepler import cart_to_kep, kep_to_cart, euler_rotation_matrix
from keppy.utils import darken, complementary_color, omit, deep_diff


def plot(
    self,
    theta: float = None,
    thetas: ArrayLike = None,
    ax: plt.Axes = None,
    plot_vectors=False,
    plot_velocity=False,
    plot_foci=False,
    rc={},
    **kwargs,
) -> None:
    """
    Plot the orbit.

    Parameters
    ----------
    theta : float, optional
        true anomaly at which to plot the position
    thetas : array_like, optional
        true anomalies for which to plot the orbit. if `None`, defaults to [0, 2π]
    ax : plt.Axes, optional
        axis to plot on. if `None`, creates a new figure
    plot_vectors : bool, optional
        whether to plot the eccentricity and inclination vectors
    plot_foci : bool, optional
        whether to plot the foci of the ellipse
    rc : dict, optional
        matplotlib rc parameters
    kwargs : dict, optional
        additional keyword arguments to pass to the main `plt.plot` call responsible
        for plotting the orbit
    """
    current_style = omit(plt.rcParams, plt.style.core.STYLE_BLACKLIST)
    own_style = omit(
        deep_diff(plt.rcParamsDefault, plt.rcParams),
        plt.style.core.STYLE_BLACKLIST,
    )
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

    plt.style.use([*custom_styles, own_style, rc])

    ax = ax or plt.gca()
    ax.set_aspect("equal")
    ax.ticklabel_format(style="scientific", axis="both", scilimits=(0, 0))

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

    if theta is not None:
        assert np.shape(theta) == ()
        [x, y, z] = kep_to_cart([*self.kep[:5], float(theta)], mu=self.mu)[..., :3]
        cc = complementary_color(c)

        ax.scatter(
            [x],
            [y],
            *([[z]] if ax.name == "3d" else []),
            zorder=3,
            color=cc,
        )
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

        # Draw the ascending node if we have one
        if np.abs(self.i) > 0:
            R_inv = np.linalg.inv(euler_rotation_matrix(self.i, self.omega, self.Omega).squeeze())
            asc_node = R_inv @ np.array([xp, yp, zp])

            ref_dir = np.array([1, 0, 0])
            asc_node = rotate_vector(ref_dir, [0, 0, 1], self.Omega) * 7000

            print(np.linalg.norm(asc_node), self.rp)

            arg = rotate_vector(asc_node, self.h_vec, self.omega)

            # Plot argument of periapsis
            if abs(self.omega) > 0:
                ax.plot(
                    [0, asc_node[0]],
                    [0, asc_node[1]],
                    *([[0, asc_node[2]]] if ax.name == "3d" else []),
                    color=colors[2],
                    ls="--",
                )

                ax.plot(
                    [0, arg[0]],
                    [0, arg[1]],
                    *([[0, arg[2]]] if ax.name == "3d" else []),
                    color=colors[2],
                    ls="--",
                )

        if ax.name == "3d":
            pass
        else:
            pass

        if ax.name == "3d" and abs(self.i) > 0:
            [xa, ya, za] = kep_to_cart([*self.kep[:5], np.pi / 2], mu=self.mu)[:3]
            ax.plot([xa, xa], [ya, ya], [za, 0], color=colors[0], ls="--")

            ref_orbit = Orbit([self.a, self.e, 0, self.omega, self.Omega, 0], self.mu)
            ref_plane = ref_orbit.trajectory()
            ax.plot(
                ref_plane[:, 0],
                ref_plane[:, 1],
                ref_plane[:, 2],
                color=colors[0],
            )

        if ax.name == "3d" and abs(self.Omega) > 0:
            pass

    if plot_foci:
        ax.scatter(
            [0],
            [0],
            *([[0]] if ax.name == "3d" else []),
            color=darken(c, 0.25),
            s=30,
            zorder=2,
        )

        if self.type == "elliptic":
            [xf1, yf1, zf1] = kep_to_cart([self.a * self.e, 0, *self.kep[2:5], np.pi], mu=self.mu)[:3]  # fmt: off
            [xf2, yf2, zf2] = kep_to_cart([2 * self.a * self.e, 0, *self.kep[2:5], np.pi], mu=self.mu)[:3]  # fmt: off

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

    if plot_velocity:
        if ax.name == "3d":
            pass
        else:
            orbit = self.at_theta(theta) if theta is not None else self
            scaled_v = scale_vector(orbit.v_vec[:2], 0.5)

            with np.errstate(invalid="ignore", divide="ignore"):
                scale = scaled_v / orbit.v_vec[:2]
                scale = scale[np.isfinite(scale)][0]

            plot_vector(
                scaled_v,
                ax=ax,
                annotations={"v": "v"},
                origin=orbit.r_vec,
            )
            plot_vector(
                orbit.vr * scale,
                ax=ax,
                origin=orbit.r_vec,
                annotations={"v": "v_r"},
            )
            plot_vector(
                orbit.vt * scale,
                ax=ax,
                origin=orbit.r_vec,
                annotations={"v": r"v_\theta"},
            )

    plt.style.use(current_style)


# orbit = Orbit([10_000, 0.3, 0.2, 0.4, 0.9, 1.2], mu=MU_EARTH)
orbit = Orbit([10_000, 0.3, 0.2, 2.2, np.pi / 2], mu=MU_EARTH)
plot(orbit, theta=0.5, plot_vectors=True)


# %%
orbit.plot()

theta = np.pi / 2
self = orbit

if not np.isclose(self.i, 0):
    asc_node = self.at_theta(-self.omega).r_vec
    periapsis = self.at_theta(0).r_vec

    apoapsis = self.at_theta(np.pi).r_vec
    plot_vector(apoapsis, annotations=r"$r_a$")

    plot_vector(periapsis, annotations=r"$r_p$")
    plot_vector(asc_node, annotations=r"$☊$")
    plot_angle(asc_node, periapsis, r"$\omega$", size=0.75 * self.rp)

    if theta is not None:
        plot_vector(self.at_theta(theta).r_vec, annotations=r"$r$")
        plot_angle(periapsis, self.at_theta(theta).r_vec, r"$\theta$", size=0.75 * self.rp)

    if not np.isclose(self.Omega, 0):
        ref_dir = np.array([1, 0, 0]) * 0.75 * self.rp / 2
        plot_angle(ref_dir, asc_node, r"$\Omega$", size=0.75 * self.rp)
        plot_vector(ref_dir, annotations=r"$♈︎$")


# %%
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.patches import Circle, PathPatch, Arc, Ellipse
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.text import Annotation
from matplotlib.path import Path


class Annotation3D(Annotation):
    def __init__(self, text, xyz, *args, **kwargs):
        super().__init__(text, xy=(0, 0), *args, **kwargs)
        self._xyz = xyz

    def draw(self, renderer):
        x2, y2, z2 = proj_transform(*self._xyz, self.axes.M)
        self.xy = (x2, y2)
        super().draw(renderer)


class Arrow3D(FancyArrowPatch):
    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)


def plot_3d_vector(v, origin=[0, 0, 0], ax=None):
    if ax is None:
        fig = plt.gcf()

        if len(fig.axes) > 0 and fig.gca().name == "3d":
            ax = fig.gca()
        else:
            ax = fig.add_subplot(projection="3d")

    origin = np.asarray(origin).ravel()[:3]
    v = np.asarray(v).ravel()[:3]

    bg = ax.get_facecolor()

    annotation = Annotation3D("a", (0, 0, 0), bbox=dict(facecolor=bg, edgecolor=bg, pad=0.3))
    ax.add_artist(annotation)

    arrow = Arrow3D(*origin, *(origin + v), mutation_scale=20, edgecolor=bg)
    ax.add_artist(arrow)


fig = plt.figure()
ax = fig.add_subplot(projection="3d")


# patch = Circle((0, 0), 1)
# patch = Ellipse((0, 0), 2, 3)

theta1 = 180
theta2 = 0

patch = Arc((0, 0), 2, 2)
ax.add_patch(patch)
patch._path = Path.arc(theta1, theta2)
art3d.pathpatch_2d_to_3d(patch, z=0, zdir="z")

patch._segment3d = rotate_vector(patch._segment3d, [1, 0, 0], np.pi / 4)


ax.set_xlim3d(-2, 2)
ax.set_ylim3d(-2, 2)
ax.set_zlim3d(-2, 2)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")


ax.view_init(elev=40, azim=10)

plot_3d_vector([2, 0, 0])
plot_3d_vector([-2, 0, 0])


# %%
fig = plt.figure()
ax = fig.add_subplot(projection="3d")

v1 = np.asarray([4, -1, 0])
v2 = np.asarray([-2, 1, 1])

v = v2 - v1
mid = v1 + v / 2
angle1 = angle_between(v1, v2)

a = np.array([-1, 0, 0])
b = mid / np.linalg.norm(mid)
axis = np.cross(a, b)
angle2 = angle_between(a, b)

theta1 = 360 - np.rad2deg(angle1)
theta2 = 0

patch = Arc((0, 0), 2, 2)
ax.add_patch(patch)
patch._path = Path.arc(theta1, theta2)
art3d.pathpatch_2d_to_3d(patch, z=0, zdir="z")
patch._segment3d = rotate_vector(patch._segment3d, axis, -angle2)

plot_3d_vector(v1)
plot_3d_vector(v2)

ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-2, 2)

plot_arc3d(v1, v2, radius=1)
ax.view_init(elev=-20, azim=10)

# %%
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection


def plot_arc3d(v1, v2, radius=0.2, ax=None):
    """Plot arc between two given vectors in 3D space."""

    v1 = np.asarray(v1).ravel()[:3]
    v2 = np.asarray(v2).ravel()[:3]

    if ax is None:
        fig = plt.gcf()

        if len(fig.axes) > 0 and fig.gca().name == "3d":
            ax = fig.gca()
        else:
            ax = fig.add_subplot(projection="3d")

    v = v1 - v2
    v_pts = np.apply_along_axis(
        lambda x: v2 + v * x, 1, np.linspace(0, 1, num=50)[None, :].repeat(3, 0).T
    )
    v_phis = np.arctan2(v_pts[:, 1], v_pts[:, 0])
    v_thetas = np.arccos(v_pts[:, 2] / np.linalg.norm(v_pts, axis=-1))

    v_points_arc = np.concatenate(
        [
            np.stack(
                [
                    radius * np.sin(v_thetas) * np.cos(v_phis),
                    radius * np.sin(v_thetas) * np.sin(v_phis),
                    radius * np.cos(v_thetas),
                ],
                axis=-1,
            ),
        ],
        axis=0,
    )

    points_collection = Line3DCollection([v_points_arc], alpha=1)
    # points_collection.set_facecolor(colour)
    points_collection.set_edgecolor("k")
    ax.add_collection3d(points_collection)

    return fig


# %%
radius = 0.2
v1 = np.asarray([2, -1, 0])
v2 = np.asarray([2, 1, 0])

v = v1 - v2
v_pts = np.apply_along_axis(
    lambda x: v2 + v * x, 1, np.linspace(0, 1, num=15)[None, :].repeat(3, 0).T
)
v_phis = np.arctan2(v_pts[:, 1], v_pts[:, 0])
v_thetas = np.arccos(v_pts[:, 2] / np.linalg.norm(v_pts, axis=-1))

v_points_arc = np.concatenate(
    [
        np.stack(
            [
                radius * np.sin(v_thetas) * np.cos(v_phis),
                radius * np.sin(v_thetas) * np.sin(v_phis),
                radius * np.cos(v_thetas),
            ],
            axis=-1,
        ),
        [[0, 0, 0]],
    ],
    axis=0,
)

fig = plt.figure()
ax = fig.add_subplot(projection="3d")

ax.scatter(*v_points_arc.T)
plot_3d_vector(v1)
plot_3d_vector(v2)
