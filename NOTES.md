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
- Operation results store references to the `Value` objects that produced them.
- Operation results store metadata identifying the primitive operation.
- Local derivative rules accumulate into parent gradients using `+=`.
- Local derivative rules should not overwrite existing gradients.
- Forward data should not change during local gradient propagation.
- Graph construction is acyclic because new nodes are only created from existing nodes.

## Commit plan

1. Initialize repository structure and scope.
2. Add scalar `Value` abstraction and forward operations.
3. Store graph provenance and local gradient rules.
4. Implement reverse-mode backward traversal.
5. Add gradient accumulation tests and edge-case handling.
6. Expand README with design, limitations, and testing strategy.

## Commit 3: Graph provenance and local gradient rules

### Goal

Extend the `Value` abstraction so that each result produced by a primitive operation records enough information for
graph computation.

At this stage, the engine should store graph provenance and local derivative rules, but should not yet implement full
reverse-mode traversal. The purpose of this commit is to make each operation locally differentiable and graph-aware.

### Scope

This commit should add:

- parent/dependency references for each operation result
- a gradient field on each `Value`
- local derivative rules for primitive operations
- tests for graph structure and local gradient propagation

This commit should not add:

- full topological traversal
- a public `backward()` method
- global chain-rule propagation through an entire graph
- gradient reset utilities
- tensor operations
- broadcasting

## Implementation notes

- `Value` is the central abstraction.
- Operator overloading is used to make scalar expressions resemble ordinary Python arithmetic.
- Reflected operators are included for mixed expressions where a Python scalar appears on the left-hand side.
- Power is currently restricted to constant Python scalar exponents.

### Graph provenance

Each operation result should remember the operands that produced it. These references form the local dependency
structure of the computation graph.

For example, if `z = x + y`, then `z` should store references to `x` and `y` and record that it was produced by
addition.

### Operation metadata

Each operation result should store a small operation label. The label is mainly for debugging, testing, and graph
inspection.

Candidate labels:

- `+`
- `*`
- `-`
- `/`
- `**`
- `neg`

The exact labels are less important than using them consistently.

### Gradient field

Each `Value` should have a `grad` field Initialized to `0.0`.

At this stage, gradients are not propagated through the full graph. Instead, local derivative rules should update only
the immediate parents of a node when the local backward rule is invoked.

### Local derivative rules

Each primitive operation should define how an upstream gradient from the output contributes to the gradients of its
immediate inputs.

For `z = x + y`:

- `x.grad += z.grad`
- `y.grad += z.grad`

For `z = x * y`:

- `x.grad += y.data * z.grad`
- `y.grad += x.data * z.grad`

For `z = x - y`:

- `x.grad += z.grad`
- `y.grad += -z.grad`

For `z = x / y`:

- `x.grad += ( 1 / y.data) * z.grad`
- `y.grad += -(x.data / y.data ** 2) * z.grad`

For `z = x ** n`, where `n` is a constant:

- `x.grad += n * x.data**(n - 1) * z.grad`

For `z = -x`:

- `x.grad += -z.grad`

The exponent in `x ** n` is treated as a constant, not as a graph-tracked `Value`.

## Testing checklist

Validation will focus first on the forward semantics of the `Value` abstraction. In particular, tests will verify that
the basic arithmetic operations produce scalar results under both direct and mixed operand usage, including cases where
`Value` appears on either the left-hand or right-hand side of an expression.

## Commit 3 testing checklist

The goal of Commit 3 is to verify graph provenance and local derivative rules without implementing full reverse-mode
traversal.

Graph provenance tests should verify:

- a leaf `Value` has no parents
- an operation result stores its parent dependencies
- an operation result stores the correct operation label
- forward numerical behavior from Commit 2 still works
- scalar operands are handled correctly when coerced into `Value` objects

Local gradient tests should verify that:

## Commit 3 acceptance criteria

Commit 3 is complete when:

- all Commit 2 forward tests still pass
- each `Value` has a gradient field initialized to `0.0`
- operation results store parent references
- operation results store operation metadata
- each primitive operation has a local derivative rule
- local derivative rules can be tested without full graph traversal
- no public `backward()` traversal has been implemented yet.

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
