from matplotlib.colors import hsv_to_rgb, rgb_to_hsv, to_hex, to_rgb, to_rgba


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


def omit(a: dict, keys: list) -> dict:
    return {k: v for k, v in a.items() if k not in keys}


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
