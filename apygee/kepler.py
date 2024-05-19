from numpy.typing import ArrayLike

import numpy as np

from apygee.utils import dot


def kep_to_cart(kep: ArrayLike, mu: float | ArrayLike) -> np.ndarray:
    """
    Parameters
    ----------
    kep : array_like
        keplerian elements: [a, e, i, Ω, ω, θ]
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
    eps = np.finfo(np.float64).eps

    if isinstance(mu, (int, float)):
        mu = np.full((kep.shape[0],), mu, dtype=np.float64)
    else:
        mu = np.asarray(mu, dtype=np.float64).ravel()

    # Step 1: orbital angular momentum
    h = np.zeros_like(e)
    par = np.isclose(e, 1, atol=eps)
    h[par] = np.sqrt(mu[par] * np.abs(a[par]))
    h[~par] = np.sqrt(mu[~par] * np.abs(a[~par] * (1 - e[~par] ** 2)))

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
        keplerian elements: `[a, e, i, Ω, ω, θ]`
    """
    shape = np.shape(cart)
    cart = np.asarray(cart, dtype=np.float64).reshape((-1, 6))
    eps = 20 * np.finfo(np.float64).eps

    if isinstance(mu, (int, float)):
        mu = np.full((cart.shape[0],), mu, dtype=np.float64)
    else:
        mu = np.asarray(mu, dtype=np.float64).ravel()

    # Step 1: position and velocity vectors
    r_vec = cart[:, :3]
    v_vec = cart[:, 3:]
    r = np.linalg.norm(r_vec, axis=-1)

    # Step 2: calculate angular momentum and semi-latus rectum (orbital parameter)
    h_vec = np.cross(r_vec, v_vec, axis=-1)
    h = np.linalg.norm(h_vec, axis=-1)
    p = h**2 / mu

    # Step 3: calculate ascending node
    K = np.array([0, 0, 1])
    N_vec = np.cross(K, h_vec, axis=-1)
    N = np.linalg.norm(N_vec, axis=-1)

    # Step 4: eccentricity
    e_vec = np.cross(v_vec, h_vec, axis=-1) / mu[:, None] - r_vec / r[:, None]
    e = np.linalg.norm(e_vec, axis=-1)
    par = np.abs(e - 1.0) < eps
    circ = np.abs(e) < eps

    # Step 5: semi-major axis
    a = np.zeros_like(p)
    a[par] = p[par]
    a[~par] = p[~par] / (1 - e[~par] ** 2)

    # Step 6: inclination
    i = np.arccos(h_vec[:, -1] / h)
    retro = i > (np.pi / 2)
    equa = (np.abs(i) < eps) | (np.abs(i - np.pi) < eps)
    N_vec[equa] = np.array([1, 0, 0])
    N[equa] = 1.0

    # Step 7: right ascension of the ascending node
    Omega = np.arccos(N_vec[:, 0] / N)
    Omega[(N_vec[:, 1] / N) < 0] = 2 * np.pi - Omega[(N_vec[:, 1] / N) < 0]

    # Step 8: argument of periapsis
    omega = np.zeros_like(e)
    omega[circ] = 0.0
    omega[~circ] = np.arccos(
        np.clip(dot(e_vec[~circ] / e[~circ, None], N_vec[~circ] / N[~circ, None]), -1, 1)
    )
    iy = ~circ & equa & ((~retro & (e_vec[:, 1] < 0)) | (retro & (e_vec[:, 1] >= 0)))
    iz = ~circ & ~equa & (e_vec[:, 2] < 0)
    omega[iy | iz] = 2 * np.pi - omega[iy | iz]

    # Step 9: true anomaly
    e_vec[circ] = N_vec[circ] / N[circ, None]
    theta = np.arccos(
        np.clip(dot(r_vec / r[:, None], e_vec / np.linalg.norm(e_vec, axis=-1)), -1, 1)
    )  # important to recalculate the norm of e_vec here, because we have just changed it (for circular orbits)

    iy = circ & equa & ((~retro & (r_vec[:, 1] < 0)) | (retro & (r_vec[:, 1] >= 0)))
    iz = circ & ~equa & ((~retro & (r_vec[:, 2] < 0)) | (retro & (r_vec[:, 2] >= 0)))
    ih = ~circ & (dot(r_vec, v_vec) < 0)
    theta[iy | iz | ih] = 2 * np.pi - theta[iy | iz | ih]

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


def specific_orbital_energy(
    r: float | ArrayLike, v: float | ArrayLike, mu: float
) -> float:
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


def eccentric_anomaly(
    M: float, e: float, atol: float = 1e-9, max_iter: int = 1000
) -> float:
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


def hyperbolic_anomaly(
    M: float, e: float, atol: float = 1e-9, max_iter: int = 1000
) -> float:
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


def t_from_M_parabolic(M: float, h: float, mu: float, tau: float = 0.0) -> float:
    return M * (h**3) / (mu**2) + tau


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
