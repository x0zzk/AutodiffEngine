class Value:
    def __init__(self, data: int | float):
        if not isinstance(data, (int, float)):
            raise TypeError("Value data must be an int or float")
        self.data = float(data)
        self.grad = 0.0

        # Dependencies used to produce this node
        self._parents = []

        # Operation label
        self._op = ""

        # Local rule for pushing this node's grad to its parents
        self._backward = lambda: None

    def __add__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        result = Value(self.data + other.data)
        result._parents = [self, other]
        result._op = "+"

        def _backward():
            self.grad += result.grad
            other.grad += result.grad

        result._backward = _backward

        return result

    def __radd__(self, other):
        if not isinstance(other, Value):
            other = Value(other)

        return self + other

    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)

        result = Value(self.data * other.data)
        result._parents = [self, other]
        result._op = "*"

        def _backward():
            self.grad += other.data * result.grad
            other.grad += self.data * result.grad

        result._backward = _backward

        return result

    def __rmul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)

        return self * other

    def __sub__(self, other):
        if not isinstance(other, Value):
            other = Value(other)

        result = Value(self.data - other.data)
        result._parents = [self, other]
        result._op = "-"

        def _backward():
            self.grad += result.grad
            other.grad -= result.grad

        result._backward = _backward

        return result

    def __rsub__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return other - self

    def __truediv__(self, other):
        if not isinstance(other, Value):
            other = Value(other)

        if other.data == 0:
            raise ZeroDivisionError("Cannot divide by zero")

        result = Value(self.data / other.data)
        result._parents = [self, other]
        result._op = "/"

        def _backward():
            self.grad += (1 / other.data) * result.grad
            other.grad -= (self.data / other.data**2) * result.grad

        result._backward = _backward

        return result

    def __rtruediv__(self, other):
        if not isinstance(other, Value):
            other = Value(other)

        if self.data == 0:
            raise ZeroDivisionError("Cannot divide by zero")

        return other / self

    def __pow__(self, other):
        if not isinstance(other, (int, float)):
            raise TypeError("Exponent must be an int or float")
        result = Value(self.data**other)
        result._parents = [self]
        result._op = "**"

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * result.grad

        result._backward = _backward

        return result

    def __neg__(self):
        result = Value(-self.data)
        result._parents = [self]
        result._op = "neg"

        def _backward():
            self.grad -= result.grad

        result._backward = _backward

        return result

    def __repr__(self):
        return f"Value(data={self.data})"

    def backward(self):
        topo = []
        visited = set()

        def build_topo(node):
            if node not in visited:
                visited.add(node)

                for parent in node._parents:
                    build_topo(parent)

                topo.append(node)

        build_topo(self)

        for node in topo:
            node.grad = 0.0

        self.grad = 1.0

        for node in reversed(topo):
            node._backward()
