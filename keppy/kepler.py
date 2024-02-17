import numpy as np
from numpy.typing import ArrayLike


def cart_to_kep(cart: ArrayLike, mu: float | ArrayLike) -> np.ndarray:
    """
    Parameters
    ----------
    cart : array_like
        cartesian state vector: [x, y, z, vx, vy, vz]
    mu : float
        gravitational parameter. either a single float or an array_like with the same length as kep

    Returns
    -------
    kep : np.ndarray
        keplerian elements: [a, e, i, ω, Ω, θ]
    """
    cart = np.asarray(cart, dtype=np.float64).reshape((-1, 6))

    e_lim = 1e-9
    i_lim = np.pi * 1e-9

    ez = np.array([0, 0, 1], dtype=cart.dtype)
    o = np.empty(cart.shape, dtype=cart.dtype)

    r = cart[:3, :]
    v = cart[3:, :]
    rn = np.linalg.norm(r, axis=0)
    vn = np.linalg.norm(v, axis=0)

    vr = np.sum((r / rn) * v, axis=0)
    epsilon = specific_orbital_energy(v, r, mu)

    # Eccentricity
    e = eccentricity_vector(v, r, mu)
    o[1, :] = np.linalg.norm(e, axis=0)

    # Semi-major axis
    # Possible cases: e < 1 and e >= 1
    e_hyp = o[1, :] >= 1
    o[0, :] = -mu / (2.0 * epsilon)
    o[0, e_hyp] = -o[0, e_hyp]

    # Angular momentum
    h = angular_momentum_vector(r, v)
    hn = np.linalg.norm(h, axis=0)
    o[2, :] = np.arccos(h[2, :] / hn)

    # Possible cases
    eg = o[1, :] > e_lim  # e greater
    ig = np.logical_and(o[2, :] > i_lim, o[2, :] < np.pi - i_lim)  # i greater
    el = np.logical_not(eg)  # e less equal
    il = np.logical_not(ig)  # i less equal

    eg_ig = np.logical_and(eg, ig)  # e > elim & i > ilim
    eg_il = np.logical_and(eg, il)  # e > elim & i <= ilim
    el_ig = np.logical_and(el, ig)  # e <= elim & i > ilim
    el_il = np.logical_and(el, il)  # e <= elim & i <= ilim

    # Ascending node
    n = np.empty_like(h)
    nn = np.empty_like(hn)
    n[:, ig] = np.cross(ez, h[:, ig], axisa=0, axisb=0, axisc=0)
    nn[ig] = np.linalg.norm(n[:, ig], axis=0)

    # Ensure ∈ [0,2pi]
    ny_neg = np.logical_and(n[1, :] < 0.0, ig)
    o[4, ig] = np.arccos(n[0, ig] / nn[ig])
    o[4, ny_neg] = 2.0 * np.pi - o[4, ny_neg]


def kep_to_cart(kep: ArrayLike, mu: float | ArrayLike) -> np.ndarray:
    """
    Parameters
    ----------
    kep : array_like
        keplerian elements: [a, e, i, ω, Ω, θ]
    mu : float, array_like
        gravitational parameter. either a single float or an array_like with the same length as kep

    Returns
    -------
    cart : np.ndarray
        cartesian state vector: [x, y, z, vx, vy, vz]
    """
    kep = np.asarray(kep, dtype=np.float64).reshape((-1, 6))

    if isinstance(mu, (int, float)):
        mu = np.full((kep.shape[0],), mu, dtype=np.float64)
    else:
        mu = np.asarray(mu, dtype=np.float64).ravel()

    cart = np.empty(kep.shape, dtype=kep.dtype)

    a = kep[:, 0]
    e = kep[:, 1]
    i = kep[:, 2]
    omega = kep[:, 3]
    Omega = kep[:, 4]
    theta = kep[:, 5]

    per = e == 1
    nper = np.logical_not(per)

    # In the consistent equations hyperbolas have negative semi-major axis
    a[(e > 1) & (a > 0)] *= -1

    h = np.empty_like(a)
    h[nper] = specific_angular_momentum(a[nper], e[nper], mu[nper])
    h[per] = parabolic_angular_momentum(a[per], mu[per])

    rn = orbital_distance(h, mu, e, theta)
    R = euler_rotation_matrix(i, omega, Omega)

    rx = rn * np.cos(theta)
    ry = rn * np.sin(theta)
    r = np.zeros((kep.shape[0], 3), dtype=kep.dtype)
    r[:, 0] = R[0, 0, ...] * rx + R[0, 1, ...] * ry
    r[:, 1] = R[1, 0, ...] * rx + R[1, 1, ...] * ry
    r[:, 2] = R[2, 0, ...] * rx + R[2, 1, ...] * ry

    vn = mu / h
    vx = -vn * np.sin(theta)
    vy = vn * (e + np.cos(theta))

    v = np.zeros((kep.shape[0], 3), dtype=kep.dtype)
    v[:, 0] = R[0, 0, ...] * vx + R[0, 1, ...] * vy
    v[:, 1] = R[1, 0, ...] * vx + R[1, 1, ...] * vy
    v[:, 2] = R[2, 0, ...] * vx + R[2, 1, ...] * vy

    cart[:, :3] = r
    cart[:, 3:] = v

    return cart.squeeze()


def euler_rotation_matrix(i: float | ArrayLike, omega: float | ArrayLike, Omega: float | ArrayLike):
    i = np.asarray(i, dtype=np.float64).ravel()
    omega = np.asarray(omega, dtype=np.float64).ravel()
    Omega = np.asarray(Omega, dtype=np.float64).ravel()

    # Assert that a maximum of one array has a length greater than 1
    assert (i.size == 1) + (omega.size == 1) + (Omega.size == 1) > 1

    R = np.empty((3, 3, np.max([i.size, omega.size, Omega.size])), dtype=np.float64)

    c1 = np.cos(Omega)
    s1 = np.sin(Omega)

    c2 = np.cos(i)
    s2 = np.sin(i)

    c3 = np.cos(omega)
    s3 = np.sin(omega)

    # First column
    R[0, 0, ...] = c1 * c3 - c2 * s1 * s3
    R[1, 0, ...] = c3 * s1 + c1 * c2 * s3
    R[2, 0, ...] = s2 * s3

    # Second column
    R[0, 1, ...] = -c1 * s3 - c2 * c3 * s1
    R[1, 1, ...] = c1 * c2 * c3 - s1 * s3
    R[2, 1, ...] = c3 * s2

    # Third column
    R[0, 2, ...] = s1 * s2
    R[1, 2, ...] = -c1 * s2
    R[2, 2, ...] = c2

    return R


def eccentricity(v: ArrayLike, r: ArrayLike, mu: float) -> float:
    return np.linalg.norm(eccentricity_vector(v, r, mu), axis=0)


def eccentricity_vector(v: ArrayLike, r: ArrayLike, mu: float) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).ravel()
    assert 0 <= v.size <= 3
    r = np.asarray(r, dtype=np.float64).ravel()
    assert 0 <= r.size <= 3

    vn = np.linalg.norm(v, axis=0)
    rn = np.linalg.norm(r, axis=0)

    return 1 / mu * ((vn**2 - mu / rn) * r - np.dot(r, v) * v)


def angular_momentum(r: ArrayLike, v: ArrayLike) -> float:
    return np.linalg.norm(angular_momentum_vector(r, v), axis=0)


def angular_momentum_vector(r: ArrayLike, v: ArrayLike) -> np.ndarray:
    r = np.asarray(r, dtype=np.float64).ravel()
    assert 0 <= r.size <= 3
    v = np.asarray(v, dtype=np.float64).ravel()
    assert 0 <= v.size <= 3

    return np.cross(r, v, axisa=0, axisb=0, axisc=0)


def specific_angular_momentum(a: float, e: float, mu: float) -> float:
    return np.sqrt(mu * a * (1 - e**2))


def parabolic_angular_momentum(a: float, mu: float) -> float:
    return np.sqrt(mu * 2 * abs(a))


def specific_orbital_energy(v: float | ArrayLike, r: float | ArrayLike, mu: float) -> float:
    if not isinstance(v, (int, float)):
        v = np.asarray(v, dtype=np.float64).ravel()
        assert 0 <= v.size <= 3
        v = np.linalg.norm(v, axis=0)
    if not isinstance(r, (int, float)):
        r = np.asarray(r, dtype=np.float64).ravel()
        assert 0 <= r.size <= 3
        r = np.linalg.norm(r, axis=0)

    return v**2 / 2 - mu / r


def orbital_distance(h: float, mu: float, e: float, theta: float) -> float:
    return (h**2 / mu) / (1 + e * np.cos(theta))
