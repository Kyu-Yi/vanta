import pytest

from toy_calc import apply_discount, arithmetic_mean, clamp, shipping_band


def test_clamp_bounds_values():
    assert clamp(-3, 0, 10) == 0
    assert clamp(7, 0, 10) == 7
    assert clamp(14, 0, 10) == 10


def test_shipping_band_examples():
    assert shipping_band(120) == "light"
    assert shipping_band(750) == "standard"
    assert shipping_band(1200) == "heavy"


def test_apply_discount_rounds_to_cents():
    assert apply_discount(1299, 25) == 974


def test_arithmetic_mean_of_even_values():
    assert arithmetic_mean([2, 4, 6, 8]) == pytest.approx(5.0)
