from kepy.kepler import cart_to_kep, kep_to_cart
from .conftest import cart_isclose, kep_isclose


def test_cart_to_kep(cart, mu):
    print(f"{cart = }")
    kep = cart_to_kep(cart, mu=mu)
    print(f"{kep = }")
    cart2 = kep_to_cart(kep, mu=mu)
    print(f"{cart2 = }")
    assert cart_isclose(cart, cart2)


def test_kep_to_cart(kep, mu):
    print(f"{kep = }")
    cart = kep_to_cart(kep, mu=mu)
    print(f"{cart = }")
    kep2 = cart_to_kep(cart, mu=mu)
    print(f"{kep2 = }")
    assert kep_isclose(kep, kep2)
