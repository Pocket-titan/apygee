from numpy.typing import ArrayLike

import numpy as np


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
    shape = np.shape(kep)
    kep = np.asarray(kep, dtype=np.float64).reshape((-1, 6))
    [a, e, i, Omega, omega, theta] = kep.T

    if isinstance(mu, (int, float)):
        mu = np.full((kep.shape[0],), mu, dtype=np.float64)
    else:
        mu = np.asarray(mu, dtype=np.float64).ravel()

    # Step 1: orbital angular momentum
    par = np.isclose(e, 1)
    npar = ~par
    h = np.zeros_like(e)
    h[par] = np.sqrt(mu[par] * np.abs(a[par]))
    h[npar] = np.sqrt(mu[npar] * a[npar] * (1 - e[npar] ** 2))

    # Step 2: transform to perifocal frame (p, q, w)
    r_w = (h**2 / mu / (1 + e * np.cos(theta)))[:, None] * np.stack(
        [np.cos(theta), np.sin(theta), np.zeros(len(theta))], axis=-1
    )
    v_w = (mu / h)[:, None] * np.stack(
        [-np.sin(theta), e + np.cos(theta), np.zeros(len(theta))], axis=-1
    )

    # Step 3: rotate the perifocal frame
    R = euler_rotation_matrix(i, omega, Omega)
    r_rot = np.einsum("ij,ijk->ik", r_w, R)
    v_rot = np.einsum("ij,ijk->ik", v_w, R)

    cart = np.concatenate([r_rot, v_rot], axis=-1)
    return cart.reshape(shape)


def cart_to_kep(cart: ArrayLike, mu: float | ArrayLike) -> np.ndarray:
    """
    Parameters
    ----------
    cart : array_like
        cartesian state vector: `[x, y, z, vx, vy, vz]`
    mu : float
        gravitational parameter. either a single float or an array_like with the same length as `kep`

    Returns
    -------
    kep : np.ndarray
        keplerian elements: `[a, e, i, ω, Ω, θ]`
    """
    shape = np.shape(cart)
    cart = np.asarray(cart, dtype=np.float64).reshape((-1, 6))

    if isinstance(mu, (int, float)):
        mu = np.full((cart.shape[0],), mu, dtype=np.float64)
    else:
        mu = np.asarray(mu, dtype=np.float64).ravel()

    r_vec = cart[:, :3]
    v_vec = cart[:, 3:]

    # Step 1: position and velocity magnitudes
    r = np.linalg.norm(r_vec, axis=-1, keepdims=True)
    v = np.linalg.norm(v_vec, axis=-1, keepdims=True)
    v_r = dot(r_vec / r, v_vec)
    v_p = np.sqrt(v.squeeze() ** 2 - v_r**2)  # noqa: F841

    # Step 2: orbital angular momentum
    h_vec = np.cross(r_vec, v_vec, axis=-1)
    h = np.linalg.norm(h_vec, axis=-1)

    # Step 3: eccentricity
    e_vec = np.cross(v_vec, h_vec, axis=-1) / mu[:, None] - r_vec / r
    e = np.linalg.norm(e_vec, axis=-1)
    circ = np.isclose(e, 0)

    # Step 4: inclination
    i = np.arccos(h_vec[:, -1] / h)
    equi = np.isclose(i, 0)

    # Step 5: right ascension of the ascending node
    K = np.array([0, 0, 1])
    N_vec = np.cross(K, h_vec)
    N = np.linalg.norm(N_vec, axis=-1)

    # If i == 0 or i == π, Omega is undefined: set to 0
    with np.errstate(invalid="ignore"):
        Omega = np.nan_to_num(np.arccos(N_vec[:, 0] / N), nan=0.0)
        Omega[equi] = 0.0

    Omega = np.mod(Omega, 2 * np.pi)
    Omega[N_vec[:, 1] < 0] = 2 * np.pi - Omega[N_vec[:, 1] < 0]

    # Step 6: argument of periapsis
    pro = e_vec[:, -1] >= 0
    retro = ~pro

    # If i == 0 or i == π, omega is undefined: set to 2d case
    with np.errstate(invalid="ignore"):
        omega = np.nan_to_num(np.arccos(dot(N_vec, e_vec) / (N * e)), nan=0.0)
    omega[equi & circ] = 0.0  # could also set to π
    omega[equi & ~circ] = np.arctan2(e_vec[equi & ~circ, 1], e_vec[equi & ~circ, 0])

    omega = np.mod(omega, 2 * np.pi)
    omega[retro] = 2 * np.pi - omega[retro]

    # Step 7: true anomaly
    away = v_r >= 0
    towards = ~away

    with np.errstate(invalid="ignore"):
        theta = np.nan_to_num(np.arccos(dot(r_vec / r, e_vec / e[:, None])), nan=0.0)

    theta = np.mod(theta, 2 * np.pi)
    theta[towards] = 2 * np.pi - theta[towards]

    # Step 8: semi-major axis
    par = np.isclose(e, 1)
    npar = ~par
    a = np.zeros_like(e)
    a[par] = h[par] ** 2 / mu[par]  # p = a if parabolic
    a[npar] = (h[npar] ** 2 / mu[npar]) / (1 - e[npar] ** 2)

    kep = np.stack([a, e, i, Omega, omega, theta], axis=-1)
    return kep.reshape(shape)


def euler_rotation_matrix(
    i: float | ArrayLike,
    omega: float | ArrayLike,
    Omega: float | ArrayLike,
) -> np.ndarray:
    i = np.asarray(i, dtype=np.float64)
    omega = np.asarray(omega, dtype=np.float64)
    Omega = np.asarray(Omega, dtype=np.float64)

    x = np.stack([-omega, -i, -Omega], axis=-1, dtype=np.float64)
    [s, c] = [np.sin(x).T, np.cos(x).T]
    [s1, s2, s3] = s
    [c1, c2, c3] = c

    R = np.array(
        [
            [c1 * c3 - c2 * s1 * s3, -c1 * s3 - c2 * c3 * s1, s1 * s2],
            [c3 * s1 + c1 * c2 * s3, c1 * c2 * c3 - s1 * s3, -c1 * s2],
            [s2 * s3, c3 * s2, c2],
        ]
    )

    if len(R.shape) > 2:
        R = R.transpose((2, 0, 1))

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


def sphere_of_influence(a: float, m: float, M: float) -> float:
    """
    Parameters
    ----------
    a : float
        semi-major axis of orbiting body around central body
    m : float
        mass of the orbiting body
    M : float
        mass of the central body

    Returns
    -------
    r : float
        radius of the sphere of influence
    """
    return a * (m / M) ** (2 / 5)


def dot(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return np.sum(a * b, axis=-1)


def angle_between(a: ArrayLike, b: ArrayLike) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return np.arccos(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))
