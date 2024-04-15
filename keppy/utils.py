from typing import Any
from numpy.typing import ArrayLike

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import hsv_to_rgb, rgb_to_hsv, to_hex, to_rgb, to_rgba
from matplotlib.patches import Arc
from matplotlib.transforms import IdentityTransform, Bbox, TransformedBbox


def maybe_unwrap(x: np.ndarray | Any) -> Any:
    if hasattr(x, "item") and np.size(x) == 1:
        return x.item()

    return x


def shorten_fstring_number(x: str):
    x = x.replace("E+", "e+").replace("E-", "e-")

    if "e+" in x or "e-" in x:
        [l, r] = x.split("e")
        l = l.rstrip("0").rstrip(".")
        r = r[0] + r[1:].lstrip("0")
        return l + "e" + r if len(r) > 1 else l

    return x.rstrip("0").rstrip(".")


def deep_diff(a: dict, b: dict) -> dict:
    diff = {}

    for key in b:
        if key not in a:
            diff[key] = b[key]
        elif isinstance(a[key], dict) and isinstance(b[key], dict):
            nested_diff = deep_diff(a[key], b[key])
            if nested_diff:
                diff[key] = nested_diff
        elif isinstance(a[key], list) and isinstance(b[key], list):
            if a[key] != b[key]:
                diff[key] = b[key]
        elif a[key] != b[key]:
            diff[key] = b[key]

    return diff


def is_iterable(x) -> bool:
    try:
        iter(x)
        return True
    except TypeError:
        return False


def flatten(x) -> list:
    out = []

    for i in x:
        if is_iterable(i):
            out.extend(flatten(i))
        else:
            out.append(i)

    return out


def omit(a: dict, keys: list) -> dict:
    return {k: v for k, v in a.items() if k not in keys}


def shape_equals(arr: ArrayLike, shape: tuple | list) -> bool:
    arr_shape = np.shape(arr)

    if arr_shape.length != shape.length:
        return False

    for i in range(arr_shape.length):
        if shape[i] == -1:
            continue

        if arr_shape[i] != shape[i]:
            return False

    return True


def scale_vector(v: ArrayLike, frac: float, ax=None, origin=[0, 0]) -> np.ndarray:
    """
    Scale given vector such that its maximum component has length `frac` of the smallest axis.
    """
    origin = np.asarray(origin)
    v = np.asarray(v)

    with plt.ioff():
        if ax is None:
            ax = plt.gca()

        disp_coords = ax.transAxes.transform([[0, 0], [1, 1]])  # axes to display
        coords = ax.transData.inverted().transform(disp_coords)  # display to data

    with np.errstate(divide="ignore", invalid="ignore"):
        scale = frac * np.min(
            np.nan_to_num(
                (coords[1] - coords[0]) / np.abs(v),
                nan=np.inf,
            )
        )

    plt.ion()

    return v * scale


def rotate_vector(v: ArrayLike, axis: ArrayLike, angle: float):
    """
    Rotates a vector `v` around an arbitrary axis by `angle` radians.
    """
    axis = np.asarray(axis, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    [ux, uy, uz] = axis / np.linalg.norm(axis)
    R = np.array(
        [
            [
                np.cos(angle) + ux**2 * (1 - np.cos(angle)),
                ux * uy * (1 - np.cos(angle)) - uz * np.sin(angle),
                ux * uz * (1 - np.cos(angle)) + uy * np.sin(angle),
            ],
            [
                uy * ux * (1 - np.cos(angle)) + uz * np.sin(angle),
                np.cos(angle) + uy**2 * (1 - np.cos(angle)),
                uy * uz * (1 - np.cos(angle)) - ux * np.sin(angle),
            ],
            [
                uz * ux * (1 - np.cos(angle)) - uy * np.sin(angle),
                uz * uy * (1 - np.cos(angle)) + ux * np.sin(angle),
                np.cos(angle) + uz**2 * (1 - np.cos(angle)),
            ],
        ]
    )

    return np.tensordot(v, R, axes=(-1, -1))


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
        text_kw=None,
        **kwargs,
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

        text_kw : dict
            Dictionary of arguments passed to the Annotation.

        **kwargs
            Further parameters are passed to `matplotlib.patches.Arc`. Use this
            to specify, color, linewidth etc. of the arc.

        """
        self.ax = ax or plt.gca()
        self._xydata = xy  # in data coordinates
        self.vec1 = p1
        self.vec2 = p2
        self.size = size
        self.unit = unit
        self.textposition = textposition

        super().__init__(
            self._xydata,
            size,
            size,
            angle=0.0,
            theta1=self.theta1,
            theta2=self.theta2,
            **kwargs,
        )

        self.set_transform(IdentityTransform())
        self.ax.add_patch(self)

        if text is not None:
            self.kw = dict(
                ha="center",
                va="center",
                xycoords=IdentityTransform(),
                xytext=(0, 0),
                textcoords="offset points",
                annotation_clip=True,
            )
            self.kw.update(text_kw or {})
            self.text = self.ax.annotate(text, xy=self._center, **self.kw)

        if self.unit[:4] == "data":
            self.ax.get_figure().draw_without_rendering()
            self.update_arc()

    def transform_vec(self, vec):
        return self.ax.transData.transform(vec) - self._center

    def update_arc(self):
        _size = self.get_size()
        self._width = _size
        self._height = _size

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
        super().draw(renderer)

    def update_text(self):
        if not hasattr(self, "text") or self.text is None:
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


def plot_vector(
    v: ArrayLike,
    ax=None,
    origin=[0, 0],
    color=None,
    plot_components=False,
    annotations={"v": None, "x": None, "y": None},
    **kwargs,
) -> None:
    origin = np.asarray(origin).ravel()[:2]
    v = np.asarray(v).ravel()[:2]

    if ax is None:
        ax = plt.gca()

    [dummy] = ax.plot(*np.stack([origin, origin + v]).T)
    if color is None:
        color = dummy.get_color()
    dummy.remove()

    arrowprops = {"width": 1.5, "headwidth": 12, "headlength": 12, **kwargs}

    ax.annotate(
        "",
        xy=origin + v,
        xytext=origin,
        xycoords="data",
        arrowprops={**arrowprops, "color": color},
    )

    if plot_components and ax.name != "3d":
        colors = [darken(desaturate(x, 0.6), 0.2) for x in split_complementary_colors(color)]
        ax.annotate(
            "",
            xy=origin + [v[0], 0],
            xytext=origin,
            xycoords="data",
            arrowprops={**arrowprops, "color": colors[0]},
        )
        ax.annotate(
            "",
            xy=origin + [0, v[1]],
            xytext=origin,
            xycoords="data",
            arrowprops={**arrowprops, "color": colors[1]},
        )
        ax.plot(*np.stack([origin + [v[0], 0], origin + v]).T, lw=2, ls="--", color="0.7")
        ax.plot(*np.stack([origin + [0, v[1]], origin + v]).T, lw=2, ls="--", color="0.7")

    bg = ax.get_facecolor()
    if isinstance(annotations, str):
        annotations = {"v": annotations}
    for k, text in annotations.items():
        if text is None or k not in ["x", "y", "v"]:
            continue
        if k in ["x", "y"] and not plot_components:
            continue

        if k == "x":
            xy = (v[0] * 0.5, 0)
        if k == "y":
            xy = (0, v[1] * 0.5)
        if k == "v":
            xy = (v[0] * 0.5, v[1] * 0.5)

        if np.allclose(xy, 0):
            continue

        ax.annotate(
            f"{text}",
            xy=origin + xy,
            xytext=(0, 0),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=13,
            color="0.3",
            bbox=dict(facecolor=bg, edgecolor=bg, pad=0.3),
        )


def plot_angle(v1, v2, angle: str, ax=None, origin=[0, 0], size=None, unit=None, **kwargs) -> None:
    origin = np.asarray(origin).ravel()[:2]
    v1 = np.asarray(v1).ravel()[:2]
    v2 = np.asarray(v2).ravel()[:2]

    if unit is None:
        unit = "data min"
    if size is None:
        [vn1, vn2] = [np.linalg.norm(v) for v in [v1, v2]]
        vnmin = min(vn1, vn2)
        size = 2 * 0.3 * vnmin

    angle = AngleAnnotation(
        origin, v1, v2, ax=ax, size=size, unit=unit, text=angle, textposition="inside", **kwargs
    )


def mix_colors(
    color1: str | tuple[float],
    color2: str | tuple[float],
    fraction: float = 0.5,
) -> str:
    c1 = to_rgba(color1)
    c2 = to_rgba(color2)

    p = min(max(fraction, 0), 1)
    w = p * 2 - 1
    a = c1[-1] - c2[-1]

    w1 = (((w if w * a == -1 else (w + a)) / (1 + w * a)) + 1) / 2.0
    w2 = 1 - w1

    [r, g, b] = [*map(lambda i: c1[i] * w1 + c2[i] * w2, range(3))]
    alpha = c1[-1] * p + c2[-1] * (1 - p)

    return to_hex([r, g, b, alpha])


def lighten(color: str | tuple[float], amount: float) -> str:
    return mix_colors(color, "white", 1 - amount)


def darken(color: str | tuple[float], amount: float) -> str:
    return mix_colors(color, "black", 1 - amount)


def desaturate(color: str | tuple[float], amount: float) -> str:
    [h, s, v] = rgb_to_hsv(to_rgb(color))
    return to_hex(hsv_to_rgb([h, max(0, min(1, s * (1 - amount))), v]))


def hilo(a: float, b: float, c: float) -> float:
    "Sum of the min & max of (a, b, c)"

    if c < b:
        b, c = c, b
    if b < a:
        a, b = b, a
    if c < b:
        b, c = c, b

    return a + c


def complementary_color(color: str | tuple[float]) -> str:
    if isinstance(color, tuple) or isinstance(color, list):
        c = color[:3]
        a = color[4] if len(color) > 3 else 1
    else:
        c = color
        a = 1

    [r, g, b, a] = to_rgba(c, alpha=a)
    k = hilo(r, g, b)
    return to_hex(to_rgba(tuple([k - u for u in (r, g, b)]), alpha=a))


def split_complementary_colors(color: str | tuple[float]) -> str:
    [h, s, v] = rgb_to_hsv(to_rgb(color))
    h1 = (h + 0.5 + 30 / 360) % 1
    h2 = (h + 0.5 - 30 / 360) % 1
    return to_hex(hsv_to_rgb([h1, s, v])), to_hex(hsv_to_rgb([h2, s, v]))


def n_adic_colors(color: str | tuple[float], n: int) -> str:
    [h, s, v] = rgb_to_hsv(to_rgb(color))
    return [to_hex(hsv_to_rgb([(h + i / n) % 1, s, v])) for i in range(1, n)]
