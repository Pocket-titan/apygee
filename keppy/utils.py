from typing import Any
from numpy.typing import ArrayLike

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv, to_hex, to_rgb, to_rgba


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

    return R @ v


def plot_vector(
    v: ArrayLike,
    origin=[0, 0],
    ax=None,
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
        colors = [
            darken(desaturate(x, 0.6), 0.2) for x in split_complementary_colors(color)
        ]
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
            f"${text}$",
            xy=origin + xy,
            xytext=(0, 0),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=13,
            color="0.3",
            bbox=dict(facecolor=bg, edgecolor=bg, pad=0.3),
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
