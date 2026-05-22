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

## Commit 4: Reverse-mode backward traversal

### Goal

Extend the `Value` abstraction so that a final output node can propagate gradients backward through the entire
computation graph.

At this stage, the engine should implement full reverse-mode traversal using the graph provenance and local derivative
rules added in Commit 3. The purpose of this commit is to connect the local `_backward` rules into a complete chain-rule
computation over a scalar computation graph.

### Scope

This commit should add:

- a public `backward()` method on `Value`
- topological traversal from an output node through its parent dependencies
- output gradient seeding
- reverse-order execution of local `_backward` rules
- tests for gradients through composed expressions

This commit should not add:

- tensor operations
- broadcasting
- neural-network layers
- finite-difference gradient checking
- extensive shared-subgraph edge-case tests
- gradient reset utilities
- repeated-call `backward()` behavior guarantees

### Backward method

The public `backward()` method should be called on the final output node of a scalar expression.

For example, if `z` is the final scalar output:

```python
z.backward()
```

then every reachable input node should receive its accumulated sensitivity with respect to `z`.

For example:

```python
x = Value(2)
y = Value(3)

z = x * y + x
z.backward()
```

Mathematically:

- `z = xy + x`
- `dz/dx = y + 1`
- `dz/dy = x`

Therefore:

- `x.grad == 4.0`
- `y.grad == 2.0`

### Topological ordering

Before gradients are propagated, the engine should build a topological ordering of the computation graph.

A topological ordering stores graph nodes in dependency order, meaning parent nodes appear before the nodes that depend
on them. The backward pass then processes this list in reverse order.

For example, if:

```python
z = x * y + x
```

then the graph contains:

- `x`
- `y`
- `x * y`
- `z`

A valid topological order is:

```text
x, y, x * y, z
```

The reverse-mode pass should process those nodes in reverse:

```text
z, x * y, y, x
```

The exact order of independent leaf nodes is not important. What matters is that each node is processed after all of the
nodes that depend on it have already sent gradient contributions into it.

### Depth-first traversal

A simple way to build the topological ordering is to perform depth-first search from the output node.

Conceptually:

```python
topo = []
visited = set()

def build_topo(node):
    if node not in visited:
        visited.add(node)

        for parent in node._parents:
            build_topo(parent)

        topo.append(node)
```

Calling `build_topo(self)` inside `backward()` records all nodes that contribute to the selected output.

The DFS traversal is performed once from the final output node. When a node has multiple parents, the traversal
recursively visits each parent before appending the current node to the topological list.

### Relationship between `topo` and `grad`

The topological list stores computational dependencies, not derivative values.

Sensitivities are accumulated in each node's `grad` field. The purpose of the topological ordering is to ensure that
each node has received its complete upstream sensitivity before its local derivative rule propagates contributions to
its parents.

In other words:

- `topo` stores `Value` objects in a valid traversal order
- `node.grad` stores the accumulated sensitivity of the output with respect to that node
- `node._backward()` uses `node.grad` to update the gradients of `node._parents`

The DFS traversal combines graph structure into one topological ordering. The `_backward` rules combine gradient
contributions into each node's `grad` field.

### Output gradient seeding

The output node should begin with gradient `1.0`.

This represents the derivative of the output with respect to itself:

```text
d(output) / d(output) = 1
```

In code:

```python
self.grad = 1.0
```

This seed gradient is the initial upstream sensitivity that begins the reverse-mode computation.

### Reverse traversal

After building the topological ordering, the backward pass should iterate through the nodes in reverse topological
order.

Conceptually:

```python
for node in reversed(topo):
    node._backward()
```

Each node's `_backward()` function distributes that node's current gradient to its immediate parents.

Commit 3 defined the local derivative rules. Commit 4 determines the correct global order in which those rules should be
applied.

### Chain-rule propagation

The key mathematical idea in this commit is the chain rule.

Each local `_backward()` rule computes how one node contributes gradient to its direct parents. The reverse traversal
combines these local contributions across the full graph.

For a composed expression:

```python
z = (x + y) * x
```

the backward pass should propagate gradients through both the multiplication node and the addition node.

The engine should not need a separate derivative formula for the whole expression. Instead, the derivative of the full
expression should emerge from composing the local derivative rules in the correct order.

## Commit 4 testing checklist

The goal of Commit 4 is to verify full reverse-mode traversal through composed scalar expressions.

Backward traversal tests should verify that:

- calling `backward()` on a leaf node seeds its gradient to `1.0`
- calling `backward()` on an addition result updates both parent gradients
- calling `backward()` on a multiplication result updates both parent gradients
- calling `backward()` on a subtraction result handles signs correctly
- calling `backward()` on a division result handles numerator and denominator gradients correctly
- calling `backward()` on a power result applies the constant-exponent derivative
- calling `backward()` on a negation result propagates a negative gradient
- composed expressions propagate gradients through more than one operation
- forward numerical behavior from earlier commits still works

Composed-expression tests should include examples like:

```python
x = Value(2)
y = Value(3)

z = x * y + x
z.backward()

assert x.grad == 4.0
assert y.grad == 2.0
```

and:

```python
x = Value(3)
y = Value(4)

z = (x + y) * (x - y)
z.backward()

assert x.grad == 6.0
assert y.grad == -8.0
```

## Commit 4 acceptance criteria

Commit 4 is complete when:

- all Commit 2 forward tests still pass
- all Commit 3 graph provenance and local-gradient tests still pass
- `Value` has a public `backward()` method
- `backward()` builds a topological ordering of the reachable graph
- `backward()` seeds the output node gradient to `1.0`
- `backward()` invokes local `_backward()` rules in reverse topological order
- gradients propagate correctly through composed scalar expressions
- the implementation still only supports scalar computation graphs
- shared-subgraph and repeated-variable edge cases are left for Commit 5 unless they arise naturally

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
