from autodiff import Value


def test_backward_on_leaf_seeds_gradient():
    x = Value(2)

    x.backward()

    assert x.grad == 1.0


def test_backward_through_addition():
    x = Value(3)
    y = Value(7)
    z = x + y

    z.backward()

    assert x.grad == 1.0
    assert y.grad == 1.0


def test_backward_through_multiplication():
    x = Value(5)
    y = Value(6)
    z = x * y

    z.backward()

    assert x.grad == 6.0
    assert y.grad == 5.0


def test_backward_through_subtraction():
    x = Value(10)
    y = Value(5)
    z = x - y

    z.backward()

    assert x.grad == 1.0
    assert y.grad == -1.0


def test_backward_through_division():
    x = Value(100)
    y = Value(25)
    z = x / y

    z.backward()

    assert x.grad == 0.04
    assert y.grad == -0.16


def test_backward_through_power():
    x = Value(3)
    z = x**2

    z.backward()

    assert x.grad == 6.0


def test_backward_through_negation():
    x = Value(5)
    z = -x

    z.backward()

    assert x.grad == -1.0


def test_backward_through_composed_expression():
    x = Value(5)
    y = Value(10)

    z = x * y + x
    z.backward()

    assert x.grad == 11.0
    assert y.grad == 5.0


def test_backward_through_nested_expression():
    x = Value(3)
    y = Value(4)

    z = (x + y) * (x - y)
    z.backward()

    assert x.grad == 6.0
    assert y.grad == -8.0
