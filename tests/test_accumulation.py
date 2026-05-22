import pytest

from autodiff import Value


def test_repeated_variable_addition_accumulates_gradient():
    x = Value(3)

    z = x + x
    z.backward()

    assert x.grad == 2.0


def test_repeated_variable_multiplication_accumulates_gradient():
    x = Value(4)

    z = x * x
    z.backward()

    assert x.grad == 8.0


def test_repeated_variable_subtraction_cancels_gradient():
    x = Value(5)

    z = x - x
    z.backward()

    assert x.grad == 0.0


def test_shared_intermediate_addition_accumulates_gradient():
    x = Value(2)
    y = Value(3)

    a = x * y
    z = a + a
    z.backward()

    assert x.grad == 6.0
    assert y.grad == 4.0


def test_shared_intermediate_multiplication_accumulates_gradient():
    x = Value(2)
    y = Value(3)

    a = x * y
    z = a * a
    z.backward()

    assert x.grad == 36.0
    assert y.grad == 24.0


def test_gradient_accumulates_from_multiple_paths():
    x = Value(2)
    y = Value(3)

    z = x * y + x * x + y
    z.backward()

    assert x.grad == 7.0
    assert y.grad == 3.0


def test_backward_resets_reachable_gradients_before_recomputing():
    x = Value(2)
    y = Value(3)

    z = x * y
    z.backward()
    z.backward()

    assert x.grad == 3.0
    assert y.grad == 2.0


def test_division_gradient_uses_approximation():
    x = Value(10)
    y = Value(3)

    z = x / y
    z.backward()

    assert x.grad == pytest.approx(1 / 3)
    assert y.grad == pytest.approx(-10 / 9)
