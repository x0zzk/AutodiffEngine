from autodiff import Value


def test_leaf_node_has_no_provenance():
    x = Value(2)

    assert x._parents == []
    assert x._op == ""
    assert x.grad == 0.0


def test_node_parent_op_provenance():
    x = Value(2)
    y = Value(4)

    z = x + y

    assert x in z._parents
    assert y in z._parents
    assert z._op == "+"


def test_addition_local_backward_updates_parent_gradients():
    x = Value(8)
    y = Value(2)
    z = x + y

    z.grad = 1.0
    z._backward()

    assert x.grad == 1.0
    assert y.grad == 1.0


def test_multiplication_local_backward_updates_parent_gradients():
    x = Value(3)
    y = Value(4)
    z = x * y

    z.grad = 1.0
    z._backward()

    assert x.grad == 4.0
    assert y.grad == 3.0


def test_subtraction_local_backward_updates_parent_gradients():
    x = Value(10)
    y = Value(5)
    z = x - y

    z.grad = 1.0
    z._backward()

    assert x.grad == 1.0
    assert y.grad == -1.0


def test_division_local_backward_updates_parent_gradients():
    x = Value(100)
    y = Value(10)
    z = x / y

    z.grad = 1.0
    z._backward()

    assert x.grad == 0.1
    assert y.grad == -1.0


def test_power_local_backward_updates_parent_gradients():
    x = Value(10)
    z = x**2.0

    z.grad = 4.0
    z._backward()

    assert x.grad == 80.0


def test_negation_local_backward_updates_parent_gradients():
    x = Value(10)
    z = -x

    z.grad = 5.0
    z._backward()

    assert x.grad == -5.0


def test_addition_result_stores_graph_provenance():
    x = Value(1)
    y = Value(2)
    z = x + y

    assert z._parents == [x, y]
    assert z._op == "+"


def test_negation_result_stores_graph_provenance():
    x = Value(3)
    z = -x

    assert z._parents == [x]
    assert z._op == "neg"


def test_power_result_tracks_base_but_not_constant_exponent():
    x = Value(3)
    z = x**2

    assert z._parents == [x]
    assert z._op == "**"


def test_reflected_subtraction_local_backward_updates_parent_gradient():
    x = Value(10)
    z = 100 - x

    z.grad = 1.0
    z._backward()

    assert x.grad == -1.0


def test_reflected_subtraction_stores_graph_provenance():
    x = Value(3)
    z = 10 - x

    assert z._op == "-"
    assert x in z._parents


def test_reflected_division_local_backward_updates_parent_gradient():
    x = Value(10)
    z = 1000 / x

    z.grad = 1.0
    z._backward()

    assert x.grad == -10.0


def test_reflected_division_stores_graph_provenance():
    x = Value(10)
    z = 100 / x

    assert z._op == "/"
    assert x in z._parents
