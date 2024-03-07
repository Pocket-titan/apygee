import numpy as np

from keppy.kepler import cart_to_kep, kep_to_cart


def test_cart_one():
    cart = np.array([10, 1, 1, 2, 2, 2])
    kep = cart_to_kep(cart, mu=1)
    cart2 = kep_to_cart(kep, mu=1)
    assert np.allclose(cart, cart2)


def test_cart_two():
    cart = np.array([[1, 10, 1, 2, 2, 2], [1, 1, 5, 2, 2, 2]])
    kep = cart_to_kep(cart, mu=1)
    cart2 = kep_to_cart(kep, mu=1)
    assert np.allclose(cart, cart2)


def test_kep_one():
    kep = np.array([10, 0.2, np.pi / 7, 0, 0, np.pi / 4])
    cart = kep_to_cart(kep, mu=1)
    kep2 = cart_to_kep(cart, mu=1)
    assert np.allclose(kep, kep2)


def test_kep_two():
    kep = np.array(
        [
            [10, 0.2, np.pi / 7, 0, 0, np.pi / 4],
            [15, 0.95, np.pi / 2, 0, 0, np.pi / 9],
        ]
    )
    cart = kep_to_cart(kep, mu=1)
    kep2 = cart_to_kep(cart, mu=1)
    assert np.allclose(kep, kep2)
