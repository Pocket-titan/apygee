from typing import TypedDict, Unpack
from numpy.typing import ArrayLike

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.patches import Arc, FancyArrowPatch
from matplotlib.transforms import Bbox, IdentityTransform, TransformedBbox
from mpl_toolkits.mplot3d.art3d import Patch3D
from mpl_toolkits.mplot3d.proj3d import proj_transform

from apygee.utils import mix_colors, omit


def plot_dummy_line(pts: ArrayLike, ax=None):
    """Expands limits so `pts` is in view, and returns the next color in the cycle"""
    ax = ax or plt.gca()

    [dummy] = ax.plot(*pts)
    color = dummy.get_color()
    dummy.remove()

    return color


def get_3d_axes():
    fig = plt.gcf()

    if len(fig.axes) > 0 and fig.gca().name == "3d":
        ax = fig.gca()
    else:
        ax = fig.add_subplot(projection="3d")

    return ax


class VectorKwargs(TypedDict):
    origin: ArrayLike
    text: str | dict
    color: str
    plot_components: bool
    arrow_kwargs: dict
    text_kwargs: dict


def plot_vector(v: ArrayLike, ax=None, **kwargs: Unpack[VectorKwargs]):
    ax = ax or plt.gca()

    if ax.name == "3d":
        plot_vector_3d(v, ax=ax, **kwargs)
    else:
        plot_vector_2d(v, ax=ax, **kwargs)


class AngleKwargs(TypedDict):
    origin: ArrayLike
    text: str
    radius: float
    unit: str
    angle_kwargs: dict
    text_kwargs: dict


def plot_angle(v1: ArrayLike, v2: ArrayLike, ax=None, **kwargs: Unpack[AngleKwargs]):
    ax = ax or plt.gca()

    if ax.name == "3d":
        plot_angle_3d(v1, v2, ax=ax, **kwargs)
    else:
        plot_angle_2d(v1, v2, ax=ax, **kwargs)


def plot_vector_2d(
    v: ArrayLike,
    origin=[0, 0],
    text: str | dict = None,
    ax=None,
    color=None,
    plot_components=False,
    arrow_kwargs={},
    text_kwargs={},
) -> None:
    origin = np.asarray(origin).ravel()[:2]
    v = np.asarray(v).ravel()[:2]
    ax = ax or plt.gca()

    _color = plot_dummy_line(np.stack([origin, origin + v]).T, ax=ax)
    if color is None:
        color = _color

    arrow_kwargs = {**arrow_kwargs}
    zorder = arrow_kwargs.pop("zorder", None)

    if "arrowstyle" in arrow_kwargs:
        arrowprops = {**arrow_kwargs}
    else:
        arrowprops = {"width": 1.5, "headwidth": 9, "headlength": 9, **arrow_kwargs}

    if plot_components:
        ax.annotate(
            "",
            xy=origin + [v[0], 0],
            xytext=origin,
            xycoords="data",
            arrowprops={**arrowprops, "color": "#E66B4C"},
            zorder=zorder,
        )
        ax.annotate(
            "",
            xy=origin + [0, v[1]],
            xytext=origin,
            xycoords="data",
            arrowprops={**arrowprops, "color": "#A4E64B"},
            zorder=zorder,
        )
        ax.plot(
            *np.stack([origin + [v[0], 0], origin + v]).T,
            lw=2,
            ls="--",
            color="0.7",
            zorder=zorder,
        )
        ax.plot(
            *np.stack([origin + [0, v[1]], origin + v]).T,
            lw=2,
            ls="--",
            color="0.7",
            zorder=zorder,
        )

    # Plot main vector after to ensure it's on top
    ax.annotate(
        "",
        xy=origin + v,
        xytext=origin,
        xycoords="data",
        arrowprops={"color": color, **arrowprops},
        zorder=zorder,
    )

    if text is None:
        return

    bg = ax.get_facecolor()

    if isinstance(text, str):
        text = {"v": text}
    for k, label in text.items():
        if label is None or k not in ["x", "y", "v"]:
            continue
        if k in ["x", "y"] and not plot_components:
            continue

        if k == "x":
            xy = (v[0] * 0.4, 0)
        if k == "y":
            xy = (0, v[1] * 0.4)
        if k == "v":
            xy = (v[0] * 0.4, v[1] * 0.4)

        if np.allclose(xy, origin):
            continue

        textprops = {
            "xy": origin + xy,
            "xytext": (0, 0),
            "textcoords": "offset points",
            "ha": "center",
            "va": "center",
            "fontsize": 10,
            "color": "0.2",
            "bbox": {
                "facecolor": bg,
                "edgecolor": "none",
                "pad": 0,
                "boxstyle": "circle",
            },
            **text_kwargs,
        }
        ax.annotate(f"{label}", **textprops)


def plot_vector_3d(
    v: ArrayLike,
    origin=[0, 0, 0],
    text: str | dict = None,
    ax=None,
    color=None,
    plot_components=False,
    arrow_kwargs={},
    text_kwargs={},
) -> None:
    origin = np.asarray(origin).ravel()[:3]
    v = np.asarray(v).ravel()[:3]
    ax = ax or get_3d_axes()
    ax.computed_zorder = False

    _color = plot_dummy_line(np.stack([origin, origin + v]).T, ax=ax)
    if color is None:
        color = _color

    vtext = text if text is None or isinstance(text, str) else text.get("v")

    arrowprops = dict(color=color)
    arrowprops.update(arrow_kwargs)

    if plot_components:
        lw = arrowprops.get("lw") or arrowprops.get("linewidth", 2)
        kwl = dict(lw=lw, ls="--", color="0.7")
        ax.plot(*(origin + np.stack([[v[0], 0, 0], [v[0], v[1], 0]])).T, **kwl)
        ax.plot(*(origin + np.stack([[v[0], 0, 0], [v[0], 0, v[2]]])).T, **kwl)
        ax.plot(*(origin + np.stack([[0, v[1], 0], [v[0], v[1], 0]])).T, **kwl)
        ax.plot(*(origin + np.stack([[0, v[1], 0], [0, v[1], v[2]]])).T, **kwl)
        ax.plot(*(origin + np.stack([[0, 0, v[2]], [v[0], 0, v[2]]])).T, **kwl)
        ax.plot(*(origin + np.stack([[0, 0, v[2]], [0, v[1], v[2]]])).T, **kwl)

        ax.plot(*(origin + np.stack([[v[0], v[1], 0], v])).T, **kwl)
        ax.plot(*(origin + np.stack([[v[0], 0, v[2]], v])).T, **kwl)
        ax.plot(*(origin + np.stack([[0, v[1], v[2]], v])).T, **kwl)

        kw = omit({**arrowprops}, "color")
        _x = arrow3D(
            [v[0], 0, 0], origin, ax=ax, arrow_kwargs={"color": "#E66B4C", **kw}, text="x"
        )
        _y = arrow3D(
            [0, v[1], 0], origin, ax=ax, arrow_kwargs={"color": "#A4E64B", **kw}, text="y"
        )
        _z = arrow3D(
            [0, 0, v[2]], origin, ax=ax, arrow_kwargs={"color": "#4C72E6", **kw}, text="z"
        )

    # Plot main vector after to ensure it's on top
    _arrow = arrow3D(
        v,
        origin,
        ax=ax,
        text=vtext,
        xyztext=text_kwargs.pop("xyz", origin + v * 0.4),
        arrow_kwargs=arrowprops,
        text_kwargs=text_kwargs,
    )


def plot_angle_2d(
    v1: ArrayLike,
    v2: ArrayLike,
    origin=[0, 0],
    text: str = None,
    ax=None,
    radius=None,
    unit=None,
    angle_kwargs={},
    text_kwargs={},
) -> None:
    origin = np.asarray(origin).ravel()[:2]
    v1 = np.asarray(v1).ravel()[:2]
    v2 = np.asarray(v2).ravel()[:2]
    ax = ax or plt.gca()

    if unit is None:
        unit = "data min"
    if radius is None:
        [vn1, vn2] = [np.linalg.norm(v) for v in [v1, v2]]
        vnmin = min(vn1, vn2)
        radius = 0.3 * vnmin

    _angle = AngleAnnotation(
        origin,
        origin + v1,
        origin + v2,
        ax=ax,
        size=2 * radius,
        unit=unit,
        text=text,
        textposition="inside",
        text_kwargs=text_kwargs,
        angle_kwargs=angle_kwargs,
    )


def plot_angle_3d(
    v1: ArrayLike,
    v2: ArrayLike,
    origin=[0, 0, 0],
    text: str = None,
    ax=None,
    radius=None,
    unit=None,
    angle_kwargs={},
    text_kwargs={},
) -> None:
    origin = np.asarray(origin).ravel()[:3]
    v1 = np.asarray(v1).ravel()[:3]
    v2 = np.asarray(v2).ravel()[:3]
    ax = ax or get_3d_axes()
    ax.computed_zorder = False

    if radius is None:
        [vn1, vn2] = [np.linalg.norm(v) for v in [v1, v2]]
        vnmin = min(vn1, vn2)
        radius = vnmin

    v = v1 - v2
    pts = np.apply_along_axis(
        lambda x: v2 + v * x, 1, np.linspace(0, 1, num=100)[None, :].repeat(3, 0).T
    )
    phis = np.arctan2(pts[:, 1], pts[:, 0])
    thetas = np.arccos(pts[:, 2] / np.linalg.norm(pts, axis=-1))
    arc = origin + np.stack(
        [
            radius * np.sin(thetas) * np.cos(phis),
            radius * np.sin(thetas) * np.sin(phis),
            radius * np.cos(thetas),
        ],
        axis=-1,
    )

    if "facecolor" in angle_kwargs:
        arc = np.concatenate([arc, [origin]], axis=0)
        default_edgecolor = "none"
    else:
        default_edgecolor = "k"

    edgecolor = angle_kwargs.get("edgecolor") or angle_kwargs.pop(
        "color", default_edgecolor
    )
    angleprops = dict(
        edgecolor=edgecolor,
        facecolor="none",
        linewidth=1,
    )
    angleprops.update(angle_kwargs)

    patch = Patch3D(**angleprops)
    patch.set_3d_properties(arc[:, :2], zs=arc[:, 2], zdir="z")
    ax.add_artist(patch)

    if text is not None:
        c = (
            arc[0]
            + ((arc[-2] if "facecolor" in angle_kwargs else arc[-1]) - arc[0]) * 0.5
        )
        phic = np.arctan2(c[1], c[0])
        thetac = np.arccos(c[2] / np.linalg.norm(c))
        center = (
            np.array(
                [
                    np.sin(thetac) * np.cos(phic),
                    np.sin(thetac) * np.sin(phic),
                    np.cos(thetac),
                ]
            )
            * radius
            * 0.6
        )

        textprops = dict(
            ha="center",
            va="center",
            zorder=99,
        )
        textprops.update(text_kwargs)
        ax.text(*center, text, **textprops)


class AngleAnnotation(Arc):
    """
    Draws an arc between two vectors which appears circular in display space.
    """

    def __init__(
        self,
        xy,
        p1,
        p2,
        size=75,
        unit="points",
        ax=None,
        text=None,
        textposition="inside",
        text_kwargs={},
        angle_kwargs={},
    ):
        """
        Parameters
        ----------
        xy, p1, p2 : tuple or array of two floats
            Center position and two points. Angle annotation is drawn between
            the two vectors connecting *p1* and *p2* with *xy*, respectively.
            Units are data coordinates.

        size : float
            Diameter of the angle annotation in units specified by *unit*.

        unit : str
            One of the following strings to specify the unit of *size*:

            * "pixels": pixels
            * "points": points, use points instead of pixels to not have a dependence on the DPI
            * "axes width", "axes height": relative units of Axes width, height
            * "axes min", "axes max": minimum or maximum of relative Axes width, height
            * "data width", "data height": relative units of data width, height
            * "data min", "data max": minimum or maximum of relative data width, height

        ax : `matplotlib.axes.Axes`
            The Axes to add the angle annotation to.

        text : str
            The text to mark the angle with.

        textposition : {"inside", "outside", "edge"}
            Whether to show the text in- or outside the arc. "edge" can be used
            for custom positions anchored at the arc's edge.

        text_kwargs : dict
            Dictionary of arguments passed to the Annotation.

        angle_kwargs : dict
            Dictionary of parameters are passed to `matplotlib.patches.Arc`. Use this
            to specify, color, linewidth etc. of the arc.

        """
        self.ax = ax or plt.gca()
        self._xydata = xy  # in data coordinates
        self.vec1 = p1
        self.vec2 = p2
        self.size = size
        self.unit = unit
        self.textposition = textposition

        facecolor = angle_kwargs.pop("facecolor") if "facecolor" in angle_kwargs else None

        super().__init__(
            self._xydata,
            size,
            size,
            angle=0.0,
            theta1=self.theta1,
            theta2=self.theta2,
            **angle_kwargs,
        )

        if facecolor is not None:
            x, y = self.get_fill_verts().T

            # this is a little hacky: if we set alpha < 1 on both edge and face, it will
            # blend with *itself*. to avoid this we only set alpha on the face, and manually
            # blend the edge with the background color to "pretend" it's transparent.
            alpha = self.get_alpha() or 1
            bg = self.ax.get_facecolor()
            edgecolor = self.get_edgecolor()

            self.fill_ = self.ax.fill(
                x,
                y,
                facecolor=to_rgba(facecolor, alpha),
                edgecolor=None if not edgecolor else mix_colors(edgecolor, bg, alpha),
                zorder=self.get_zorder(),
                lw=self.get_linewidth(),
            )[0]

        self.set_transform(IdentityTransform())
        self.ax.add_patch(self)

        if text is not None:
            self.text_kwargs = dict(
                ha="center",
                va="center",
                xycoords=IdentityTransform(),
                xytext=(0, 0),
                textcoords="offset points",
                annotation_clip=True,
            )
            self.text_kwargs.update(text_kwargs or {})
            self.text = self.ax.annotate(text, xy=self._center, **self.text_kwargs)

        if self.unit[:4] == "data":
            self.update_arc()

    def transform_vec(self, vec):
        return self.ax.transData.transform(vec) - self._center

    def get_fill_verts(self):
        verts = self.ax.transData.inverted().transform(self.get_verts())

        # sort by angle; the origin needs to connect to the first and last vert
        thetas = np.arctan2(verts[:, 1], verts[:, 0])
        thetas[thetas < 0] += 2 * np.pi
        verts = verts[np.argsort(thetas)]
        verts = np.concatenate([[self._xydata], verts, [self._xydata]])

        return verts

    def update_arc(self):
        _size = self.get_size()
        self._width = _size
        self._height = _size

    def update_fill(self):
        if not getattr(self, "fill_", None):
            return

        self.fill_.set_xy(self.get_fill_verts())

    def get_size(self):
        if self.unit == "pixels":
            factor = 1.0
        if self.unit == "points":
            factor = self.ax.figure.dpi / 72.0
        else:
            if self.unit[:4] == "axes":
                b = TransformedBbox(Bbox.unit(), self.ax.transAxes)
            if self.unit[:4] == "data":
                b = TransformedBbox(Bbox.unit(), self.ax.transData)

            dic = {
                "max": max(b.width, b.height),
                "min": min(b.width, b.height),
                "width": b.width,
                "height": b.height,
            }
            factor = dic[self.unit[5:]]

        return self.size * factor

    def set_size(self, size):
        self.size = size

    def get_center_in_pixels(self):
        """return center in pixels"""
        return self.ax.transData.transform(self._xydata)

    def set_center(self, xy):
        """set center in data coordinates"""
        self._xydata = xy

    def get_theta(self, vec):
        vec_in_pixels = self.transform_vec(vec)
        return np.rad2deg(np.arctan2(vec_in_pixels[1], vec_in_pixels[0]))

    def get_theta1(self):
        return self.get_theta(self.vec1)

    def get_theta2(self):
        return self.get_theta(self.vec2)

    def set_theta(self, angle):
        pass

    # Redefine attributes of the Arc to always give values in pixel space
    _center = property(get_center_in_pixels, set_center)
    theta1 = property(get_theta1, set_theta)
    theta2 = property(get_theta2, set_theta)
    width = property(get_size, set_size)
    height = property(get_size, set_size)

    # The following two methods are needed to update the text position.
    def draw(self, renderer):
        self.update_text()
        self.update_arc()
        self.update_fill()

        # if we have a fill, instead we give that an edgecolor & don't draw the original arc
        # this is to prevent issues with alpha
        if getattr(self, "fill_", None):
            with renderer._draw_disabled():
                super().draw(renderer)
        else:
            super().draw(renderer)

    def update_text(self):
        if not getattr(self, "text", None):
            return

        c = self._center
        s = self.get_size()

        angle_span = (self.theta2 - self.theta1) % 360
        angle = np.deg2rad(self.theta1 + angle_span / 2)

        if self.textposition == "inside":
            r = s / np.interp(angle_span, [60, 90, 135, 180], [3.3, 3.5, 3.8, 4])
        else:
            r = s / 2

        self.text.xy = c + r * np.array([np.cos(angle), np.sin(angle)])

        if self.textposition == "outside":

            def R90(a, r, w, h):
                if a < np.arctan(h / 2 / (r + w / 2)):
                    return np.sqrt((r + w / 2) ** 2 + (np.tan(a) * (r + w / 2)) ** 2)

                c = np.sqrt((w / 2) ** 2 + (h / 2) ** 2)
                T = np.arcsin(c * np.cos(np.pi / 2 - a + np.arcsin(h / 2 / c)) / r)

                xy = r * np.array([np.cos(a + T), np.sin(a + T)])
                xy += np.array([w / 2, h / 2])

                return np.sqrt(np.sum(xy**2))

            def R(a, r, w, h):
                aa = (a % (np.pi / 4)) * ((a % (np.pi / 2)) <= np.pi / 4) + (
                    np.pi / 4 - (a % (np.pi / 4))
                ) * ((a % (np.pi / 2)) >= np.pi / 4)

                return R90(aa, r, *[w, h][:: int(np.sign(np.cos(2 * a)))])

            bbox = self.text.get_window_extent()
            X = R(angle, r, bbox.width, bbox.height)
            trans = self.ax.figure.dpi_scale_trans.inverted()
            offs = trans.transform(((X - s / 2), 0))[0] * 72
            self.text.set_position([offs * np.cos(angle), offs * np.sin(angle)])


class Arrow3D(FancyArrowPatch):
    def __init__(self, x, y, z, dx, dy, dz, **kwargs):
        super().__init__((0, 0), (0, 0), **kwargs)
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


def arrow3D(
    v: ArrayLike,
    origin: ArrayLike,
    text: str = None,
    xyztext=None,
    arrow_kwargs={},
    text_kwargs={},
    ax=None,
) -> Arrow3D:
    ax = ax or get_3d_axes()

    arrowstyle = dict(
        tail_width=0.1,
        head_width=0.45,
        head_length=0.45,
    )
    arrowprops = dict(
        arrowstyle="Simple, "
        + ", ".join(map(lambda x: f"{x[0]}={x[1]}", arrowstyle.items())),
        mutation_scale=20,
    )
    arrowprops.update(arrow_kwargs)

    arrow = Arrow3D(*origin, *v, **arrowprops)
    ax.add_artist(arrow)

    if xyztext is None:
        xyztext = np.array(origin) + np.array(v) * 0.4
    if text is not None:
        bg = ax.zaxis.pane.get_facecolor() or "none"

        textprops = {
            "ha": "center",
            "va": "center",
            "fontsize": 10,
            "color": "0.2",
            "bbox": {
                "facecolor": bg,
                "edgecolor": "none",
                "pad": 0,
                "boxstyle": "circle",
            },
            **text_kwargs,
        }

        ax.text(*xyztext, text, **textprops)

    return arrow
