from typing import Any
from numpy.typing import ArrayLike

import matplotlib.pyplot as plt
import numpy as np
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


def deep_update(a: dict, b: dict) -> None:
    for key in b:
        if key not in a:
            a[key] = b[key]
        elif isinstance(a[key], dict) and isinstance(b[key], dict):
            deep_update(a[key], b[key])
        else:
            a[key] = b[key]


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


def scale_vector(v: ArrayLike, frac: float, ax: plt.Axes, origin=None) -> float:
    """
    Get the scale such that a vector's maximum component has length `frac` of the smallest axis.
    """
    is_3d = ax.name == "3d"

    if origin is None:
        origin = np.zeros((3,) if is_3d else (2,))

    origin = np.asarray(origin)[: (3 if is_3d else 2)]
    v = np.asarray(v)[: (3 if is_3d else 2)]

    lims = np.concatenate(
        [ax.get_xlim(), ax.get_ylim(), *[ax.get_zlim() if is_3d else []]]
    ).reshape((-1, 2))
    widths = np.diff(lims).ravel()

    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.nan_to_num(widths / np.abs(v), nan=np.inf)
        scale = frac * np.min(scale)

    return scale


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


def dot(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return np.sum(a * b, axis=-1)


def angle_between(a: ArrayLike, b: ArrayLike) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return np.round(
        np.arccos(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1)),
        15,
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
