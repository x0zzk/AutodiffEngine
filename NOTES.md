# Project Notes

## Project scope

- Build a small reverse-mode automatic differentiation engine for scalar computation graphs.
- Prioritize conceptual transparency over feature breadth.
- Start with scalar forward operations before adding graph provenance and reverse-mode traversal.
- Exclude tensors, broadcasting, batching, and neural-network layers from the initial scope.

## Invariants

- Each `Value` object stores a single scalar numeric value.
- Scalar inputs are normalized to `float`.
- Primitive arithmetic operations return new `Value` objects.
- Invalid non-numeric inputs should fail explicitly rather than produce implicit behavior.
- Division by zero should raise `ZeroDivisionError`.

## Commit plan

1. Initialize repository structure and scope.
2. Add scalar `Value` abstraction and forward operations.
3. Store graph provenance and local gradient rules.
4. Implement reverse-mode backward traversal.
5. Add gradient accumulation tests and edge-case handling.
6. Expand README with design, limitations, and testing strategy.

## Implementation notes

- `Value` is the central abstraction.
- Operator overloading is used to make scalar expressions resemble ordinary Python arithmetic.
- Reflected operators are included for mixed expressions where a Python scalar appears on the left-hand side.
- Power is currently restricted to constant Python scalar exponents.

## Testing checklist

Validation will focus first on the forward semantics of the `Value` abstraction. In particular, tests will verify that
the basic arithmetic operations produce scalar results under both direct and mixed operand usage, including cases where
`Value` appears on either the left-hand or right-hand side of an expression.

## Testing notes

A forward division test exposed an exception-class bug: division by zero should raise `ZeroDivisionError`, not
`TypeError`, because zero is a valid numeric scalar but an invalid denominator.

## Edge cases

- `Value / 0`
- `Value / Value(0)`
- non-numeric constructor inputs
- non-numeric power exponents
- reflected subtraction and reflected division, where operand order changes the result
- boolean inputs, since `bool` is a subclass of `int` in Python

## Terminology / conceptual distinctions

- This project is building toward reverse-mode autodiff, not forward-mode dual-number propagation.

## References / reading notes

- PyTorch autograd mechanics
- Automatic Differentiation in Machine Learning: a Survey
