import pytest

from autodiff import Value


def test_value_initializes_with_int():
    x = Value(3)

    assert x.data == 3.0


def test_value_initializes_with_float():
    x = Value(2.5)

    assert x.data == 2.5


def test_value_rejects_non_numeric_input():
    with pytest.raises(TypeError):
        Value("A")


def test_value_repr_shows_data():
    x = Value(1)

    assert repr(x) == "Value(data=1.0)"


def test_adds_two_values():
    x = Value(1)
    y = Value(2)

    result = x + y

    assert isinstance(result, Value)
    assert result.data == 3.0


def test_adds_value_to_python_scalar_in_both_orders():
    x = Value(1)

    result_scalar_left_int = x + 2
    result_scalar_left_float = x + 2.5
    result_reflected_int = 2 + x
    result_reflected_float = 2.5 + x

    assert isinstance(result_scalar_left_int, Value)
    assert result_scalar_left_int.data == 3.0

    assert isinstance(result_scalar_left_float, Value)
    assert result_scalar_left_float.data == 3.5

    assert isinstance(result_reflected_int, Value)
    assert result_reflected_int.data == 3.0

    assert isinstance(result_reflected_float, Value)
    assert result_reflected_float.data == 3.5


def test_multiplies_two_values():
    x = Value(2)
    y = Value(3)

    result = x * y

    assert isinstance(result, Value)
    assert result.data == 6.0


def test_multiplies_value_by_python_scalar_in_both_orders():
    x = Value(2)

    result_scalar_left_int = x * 2
    result_scalar_left_float = x * 2.5
    result_reflected_int = 2 * x
    result_reflected_float = 2.5 * x

    assert isinstance(result_scalar_left_int, Value)
    assert result_scalar_left_int.data == 4.0

    assert isinstance(result_scalar_left_float, Value)
    assert result_scalar_left_float.data == 5.0

    assert isinstance(result_reflected_int, Value)
    assert result_reflected_int.data == 4.0

    assert isinstance(result_reflected_float, Value)
    assert result_reflected_float.data == 5.0


def test_subtracts_two_values():
    x = Value(5)
    y = Value(3)

    result = x - y

    assert isinstance(result, Value)
    assert result.data == 2.0


def test_subtracts_python_scalar_from_value_in_both_orders():
    x = Value(5)

    result_scalar_left_int = x - 2
    result_scalar_left_float = x - 2.5
    result_reflected_int = 2 - x
    result_reflected_float = 10.0 - x

    assert isinstance(result_scalar_left_int, Value)
    assert result_scalar_left_int.data == 3.0

    assert isinstance(result_scalar_left_float, Value)
    assert result_scalar_left_float.data == 2.5

    assert isinstance(result_reflected_int, Value)
    assert result_reflected_int.data == -3.0

    assert isinstance(result_reflected_float, Value)
    assert result_reflected_float.data == 5.0


def test_divides_two_values():
    x = Value(4)
    y = Value(2)

    result = x / y

    assert isinstance(result, Value)
    assert result.data == 2.0


def test_divides_value_by_python_scalar_in_both_orders():
    x = Value(12)

    result_scalar_left_int = x / 2
    result_scalar_left_float = x / 3.0
    result_reflected_int = 36 / x
    result_reflected_float = 72.0 / x

    assert isinstance(result_scalar_left_int, Value)
    assert result_scalar_left_int.data == 6.0

    assert isinstance(result_scalar_left_float, Value)
    assert result_scalar_left_float.data == 4.0

    assert isinstance(result_reflected_int, Value)
    assert result_reflected_int.data == 3.0

    assert isinstance(result_reflected_float, Value)
    assert result_reflected_float.data == 6.0


def test_division_by_zero_raises_error():
    x = Value(10)

    with pytest.raises(ZeroDivisionError):
        x / 0


def test_division_by_zero_value_raises_error():
    x = Value(10)

    with pytest.raises(ZeroDivisionError):
        x / Value(0)


def test_raises_value_to_python_scalar_power():
    x = Value(2)

    result_scalar_left_int = x**2
    result_scalar_left_float = x**3.0

    assert isinstance(result_scalar_left_int, Value)
    assert result_scalar_left_int.data == 4.0

    assert isinstance(result_scalar_left_float, Value)
    assert result_scalar_left_float.data == 8.0


def test_power_rejects_non_numeric_exponent():
    x = Value(10)

    with pytest.raises(TypeError):
        x ** "A"


def test_composed_forward_expression_returns_expected_value():
    x = Value(10)

    result1 = x + 3
    result2 = result1 * 3
    result3 = result2**2
    result4 = result3 / 3.0
    result5 = result4 - 1.0

    assert isinstance(result5, Value)
    assert result5.data == 506.0


def test_negation_returns_negative_value():
    x = Value(8)
    y = -x

    assert y.data == -8.0
