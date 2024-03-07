from numpy.typing import ArrayLike

import numpy as np


def cart_to_kep(
    cart: ArrayLike,
    mu: float | ArrayLike,
    e_lim=1e-9,
    i_lim=np.pi * 1e-9,
) -> np.ndarray:
    """
    Parameters
    ----------
    cart : array_like
        cartesian state vector: `[x, y, z, vx, vy, vz]`
    mu : float
        gravitational parameter. either a single float or an array_like with the same length as `kep`
    e_lim : float, optional
        eccentricity limit. Defaults to `1e-9`
    i_lim : float, optional
        inclination limit. Defaults to `np.pi * 1e-9`

    Returns
    -------
    kep : np.ndarray
        keplerian elements: `[a, e, i, ω, Ω, θ]`
    """
    cart = np.asarray(cart, dtype=np.float64).reshape((-1, 6))

    ez = np.array([0, 0, 1], dtype=cart.dtype)
    kep = np.empty(cart.shape, dtype=cart.dtype)

    r = cart[..., :3]
    v = cart[..., 3:]
    rn = np.linalg.norm(r, axis=-1)
    vr = np.sum((r / rn[..., None]) * v, axis=-1)
    epsilon = specific_orbital_energy(r, v, mu)

    # Eccentricity
    e = eccentricity_vector(r, v, mu)
    kep[:, 1] = np.linalg.norm(e, axis=-1)

    # Semi-major axis
    # Possible cases: e < 1 and e >= 1
    e_hyp = kep[:, 1] >= 1
    kep[:, 0] = -mu / (2.0 * epsilon)
    kep[e_hyp, 0] = -kep[e_hyp, 0]

    # Angular momentum
    h = angular_momentum_vector(r, v)
    hn = np.linalg.norm(h, axis=-1)
    # If our angular momentum is zero, we return the nan: no valid orbit.
    with np.errstate(invalid="ignore"):
        kep[:, 2] = np.arccos(h[..., 2] / hn)

    # Possible cases
    eg = kep[:, 1] > e_lim  # e greater
    ig = np.logical_and(kep[:, 2] > i_lim, kep[:, 2] < np.pi - i_lim)  # i greater
    el = np.logical_not(eg)  # e less equal
    il = np.logical_not(ig)  # i less equal

    eg_ig = np.logical_and(eg, ig)  # e > elim & i > ilim
    eg_il = np.logical_and(eg, il)  # e > elim & i <= ilim
    el_ig = np.logical_and(el, ig)  # e <= elim & i > ilim
    el_il = np.logical_and(el, il)  # e <= elim & i <= ilim

    # Ascending node
    n = np.zeros_like(h)
    nn = np.zeros_like(hn)

    n[ig, ...] = np.cross(ez, h[ig, :], axisa=-1, axisb=-1, axisc=-1)
    nn[ig, ...] = np.linalg.norm(n[ig, ...], axis=-1)

    ny_neg = np.logical_and(n[:, 1] < 0.0, ig)
    kep[ig, 4] = np.arccos(n[ig, 0] / nn[..., ig])
    kep[ny_neg, 4] = 2.0 * np.pi - kep[ny_neg, 4]
    kep[il, 4] = 0  # i <= ilim (non-inclined); no ascending node!

    # Argument of periapsis
    kep[el, 3] = 0  # e <= elim (circular orbit); no argument of periapsis!

    # Elliptical and hyperbolic orbits
    cos_om = np.zeros_like(hn)
    # n-vector; use angle between the two. e > elim & i > ilim
    cos_om[eg_ig] = np.sum(
        n[eg_ig, ...] * e[eg_ig, ...],
        axis=-1,
    ) / (nn[..., eg_ig] * kep[eg_ig, 1])
    # no n-vector; use angle of eccentricity vector. e > elim & i <= ilim
    cos_om[eg_il] = e[eg_il, 0] / kep[eg_il, 1]
    cos_om = cos_om[eg]  # Remove unused array positions
    cos_om[cos_om > 1.0] = 1.0
    cos_om[cos_om < -1.0] = -1.0

    kep[eg, 3] = np.arccos(cos_om)

    # first case: e and n vector angle
    ez_neg = np.logical_and(e[:, 2] < 0.0, eg_ig)
    kep[ez_neg, 3] = 2.0 * np.pi - kep[ez_neg, 3]

    # second case: ex component
    # prograde
    ey_neg = np.logical_and(kep[:, 2] < np.pi * 0.5, eg_il)
    ey_neg2 = np.logical_and(ey_neg, e[:, 1] < 0.0)
    kep[ey_neg2, 3] = 2.0 * np.pi - kep[ey_neg2, 3]

    # retrograde
    ey_neg = np.logical_and(kep[:, 2] > np.pi * 0.5, eg_il)
    ey_neg2 = np.logical_and(ey_neg, e[:, 1] >= 0.0)
    kep[ey_neg2, 3] = 2.0 * np.pi - kep[ey_neg2, 3]

    # True anomaly
    cos_nu = np.empty_like(hn)
    # three cases
    # elliptical and hyperbolic: (angle from periapsis using e and r)
    cos_nu[eg] = np.sum(e[eg, :] * r[eg, :], axis=-1) / (kep[eg, 1] * rn[eg])

    # circular and inclined: (angle from periapsis using n and r)
    # if e=0 and omega := 0, with inclination +y -> +z perihelion is ascending node
    cos_nu[el_ig] = np.sum(
        ((n[el_ig].T / nn[..., el_ig]) * (r[el_ig, ...].T / rn[..., el_ig])).T,
        axis=-1,
    )

    # circular and planar: (use angle of position vector)
    cos_nu[el_il] = r[el_il, 0] / rn[..., el_il]

    # do not fail due to number precision fluctuation
    cos_nu[cos_nu > 1.0] = 1.0
    cos_nu[cos_nu < -1.0] = -1.0

    kep[:, 5] = np.arccos(cos_nu)

    # elliptical and hyperbolic
    tmp_ind_ = np.logical_and(vr < 0.0, eg)
    kep[tmp_ind_, 5] = 2.0 * np.pi - kep[tmp_ind_, 5]

    # circular and inclined
    tmp_ind_ = np.logical_and(r[:, 2] < 0.0, el_ig)
    kep[tmp_ind_, 5] = 2.0 * np.pi - kep[tmp_ind_, 5]

    # circular and planar
    # prograde
    tmp_ind_ = np.logical_and(kep[:, 2] < np.pi * 0.5, el_il)
    tmp_ind2_ = np.logical_and(tmp_ind_, r[:, 1] < 0.0)
    kep[tmp_ind2_, 5] = 2.0 * np.pi - kep[tmp_ind2_, 5]

    # if retrograde, its reversed
    tmp_ind_ = np.logical_and(kep[:, 2] > np.pi * 0.5, el_il)
    tmp_ind2_ = np.logical_and(tmp_ind_, r[:, 1] >= 0.0)
    kep[tmp_ind2_, 5] = 2.0 * np.pi - kep[tmp_ind2_, 5]

    # Ensure ∈ [0,2pi]
    kep[:, [3, 4, 5]] = kep[:, [3, 4, 5]] % (2.0 * np.pi)

    return kep


@np.errstate(divide="ignore", invalid="ignore")
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

    par = e == 1
    npar = np.logical_not(par)

    # In the consistent equations hyperbolas have negative semi-major axis
    a[(e > 1) & (a > 0)] *= -1

    h = np.empty_like(a)
    h[npar] = specific_angular_momentum(a[npar], e[npar], mu[npar])
    h[par] = parabolic_angular_momentum(a[par], mu[par])

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
    i = np.asarray(i, dtype=np.float64)
    omega = np.asarray(omega, dtype=np.float64)
    Omega = np.asarray(Omega, dtype=np.float64)

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


def eccentricity(r: ArrayLike, v: ArrayLike, mu: float) -> float:
    return np.linalg.norm(eccentricity_vector(r, v, mu), axis=-1)


def eccentricity_vector(r: ArrayLike, v: ArrayLike, mu: float) -> np.ndarray:
    r = np.asarray(r, dtype=np.float64).reshape((-1, 3))
    v = np.asarray(v, dtype=np.float64).reshape((-1, 3))

    rn = np.linalg.norm(r, axis=-1)
    vn = np.linalg.norm(v, axis=-1)

    return 1 / mu * ((vn**2 - mu / rn)[..., None] * r - dot(r, v)[..., None] * v)


def angular_momentum(r: ArrayLike, v: ArrayLike) -> float:
    return np.linalg.norm(angular_momentum_vector(r, v), axis=-1)


def angular_momentum_vector(r: ArrayLike, v: ArrayLike) -> np.ndarray:
    r = np.asarray(r, dtype=np.float64).reshape((-1, 3))
    v = np.asarray(v, dtype=np.float64).reshape((-1, 3))

    return np.cross(r, v, axisa=-1, axisb=-1, axisc=-1)


def specific_angular_momentum(a: float, e: float, mu: float) -> float:
    return np.sqrt(mu * a * (1 - e**2))


def parabolic_angular_momentum(a: float, mu: float) -> float:
    return np.sqrt(mu * np.abs(a))  # p = a, not p = 2 * a


def specific_orbital_energy(r: float | ArrayLike, v: float | ArrayLike, mu: float) -> float:
    if not isinstance(r, (int, float)):
        r = np.asarray(r, dtype=np.float64).reshape((-1, 3))
        r = np.linalg.norm(r, axis=-1)
    if not isinstance(v, (int, float)):
        v = np.asarray(v, dtype=np.float64).reshape((-1, 3))
        v = np.linalg.norm(v, axis=-1)

    return v**2 / 2 - mu / r


def orbital_distance(h: float, mu: float, e: float, theta: float) -> float:
    return (h**2 / mu) / (1 + e * np.cos(theta))


def orbital_velocity(h: float, mu: float, e: float, theta: float) -> float:
    r = orbital_distance(h, mu, e, theta)
    a = h**2 / mu / (1 - e**2)
    return np.sqrt(mu * (2 / r - 1 / a))


def orbital_plane(r: ArrayLike, v: ArrayLike) -> np.ndarray:
    h_vec = angular_momentum_vector(r, v)
    h = np.linalg.norm(h_vec, axis=-1, keepdims=True)
    return (h_vec / h).squeeze()


def mean_motion(a: float, mu: float) -> float:
    return np.sqrt(mu / a**3)


def hyperbolic_mean_motion(a: float, mu: float) -> float:
    return np.sqrt(-mu / a**3)


def mean_anomaly(n: float, t: float, tau: float = 0.0) -> float:
    return n * (t - tau)


def eccentric_anomaly(M: float, e: float, atol: float = 1e-9, max_iter: int = 1000) -> float:
    converged = False
    E = M
    i = 0

    while i < max_iter:
        E_new = M + e * np.sin(E)

        if np.abs(E - E_new) < atol:
            converged = True
            break

        E = E_new
        i += 1

    if not converged:
        raise ValueError(f"Eccentric anomaly did not converge in {max_iter} iterations.")

    return E


def E_from_theta(theta: float, e: float) -> float:
    E = 2 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(theta / 2))
    return E


def theta_from_E(E: float, e: float) -> float:
    theta = 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2))
    return theta


def M_from_E(E: float, e: float) -> float:
    return E - e * np.sin(E)


def hyperbolic_anomaly(M: float, e: float, atol: float = 1e-9, max_iter: int = 1000) -> float:
    """
    Newton iteration
    """
    converged = False
    F = M
    i = 0

    while i < max_iter:
        F_new = F + (M - e * np.sinh(F) + F) / (e * np.cosh(F) - 1)

        if np.abs(F - F_new) < atol:
            converged = True
            break

        F = F_new
        i += 1

    if not converged:
        raise ValueError(f"Hyperbolic anomaly did not converge in {max_iter} iterations.")

    return F


def barkers_equation(M: float) -> float:
    # https://ui.adsabs.harvard.edu/abs/1985JBAA...95..113M
    W = 3 * M
    y = (W + np.sqrt(W**2 + 1)) ** (1 / 3)
    x = y - 1 / y
    theta = 2 * np.arctan(x)
    return theta


def inverse_barkers_equation(theta: float) -> float:
    M = 1 / 2 * (np.tan(theta / 2) + 1 / 3 * np.tan(theta / 2) ** 3)
    return M


def F_from_theta(theta: float, e: float) -> float:
    F = 2 * np.arctanh(np.sqrt((e - 1) / (e + 1)) * np.tan(theta / 2))
    return F


def theta_from_F(F: float, e: float) -> float:
    theta = 2 * np.arctan(np.sqrt((e + 1) / (e - 1)) * np.tanh(F / 2))
    return theta


def M_from_F(F: float, e: float) -> float:
    return e * np.sinh(F) - F


def t_from_M(M: float, n: float, tau: float = 0.0) -> float:
    t = M / n + tau
    return t


def dot(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64).reshape((-1, 3))
    b = np.asarray(b, dtype=np.float64).reshape((-1, 3))

    return np.einsum("ij,ij->i", a, b)
