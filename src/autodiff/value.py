class Value:
    def __init__(self, data: int | float):
        if not isinstance(data, (int, float)):
            raise TypeError("Value data must be an int or float")
        self.data = float(data)

    def __add__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return Value(self.data + other.data)

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return Value(self.data * other.data)

    def __rmul__(self, other):
        return self * other

    def __sub__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return Value(self.data - other.data)

    def __rsub__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return Value(other.data - self.data)

    def __truediv__(self, other):
        if not isinstance(other, Value):
            other = Value(other)

        if other.data == 0:
            raise ZeroDivisionError("Cannot divide by zero")

        return Value(self.data / other.data)

    def __rtruediv__(self, other):
        if not isinstance(other, Value):
            other = Value(other)

        if self.data == 0:
            raise ZeroDivisionError("Cannot divide by zero")

        return Value(other.data / self.data)

    def __pow__(self, other):
        if not isinstance(other, (int, float)):
            raise TypeError("Exponent must be an int or float")
        return Value(self.data**other)

    def __neg__(self):
        return Value(-self.data)

    def __repr__(self):
        return f"Value(data={self.data})"
