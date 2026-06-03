# Autodiff Engine

## Overview

This project implements a small scalar reverse-mode automatic differentiation engine in Python. It builds dynamic
computation graphs during ordinary arithmetic execution and applies reverse-mode differentiation to compute gradients
with respect to scalar inputs.

The engine is intentionally scalar-first: each `Value` object represents a single numeric quantity, stores references to
its predecessors, records the operation that produced it, and accumulates gradients during backpropagation. This
constrained design makes the mechanics of automatic differentiation inspectable without introducing additional
complexity.

## Motivation

Automatic differentiation converts symbolic derivative rules into an executable computational procedure. Rather than
manually deriving gradients for every composed expression, the engine records how a value was computed and then applies
the chain rule backward through the resulting computation graph.

I built this project to understand the mechanism underlying gradient-based optimization systems. Although modern machine
learning libraries operate over tensors and highly optimized kernels, their conceptual foundation still relies on local
derivative rules, graph structure, and systematic gradient propagation.

The goal of this project is therefore not to reproduce a full machine learning framework, but to implement the essential
reverse-mode autodiff mechanism from first principles.

## Core Ideas

The engine is based on four central ideas:

1. **Scalar value objects** Each scalar is represented by a `Value` object containing its numeric data, gradient,
   predecessor nodes, and backward function.

2. **Operator overloading** Python arithmetic operators are overloaded so expressions such as `a * b + c` construct
   computation graph nodes automatically.

3. **Local derivative rules** Each primitive operation defines how gradients flow to its immediate inputs.

4. **Reverse topological traversal** Backpropagation visits nodes in reverse dependency order so that each node receives
   gradient contributions from all downstream computations.

## Implementation

The implementation centers on a `Value` abstraction. Primitive operations such as addition, multiplication,
exponentiation, negation, subtraction, and division construct new `Value` objects while preserving dependency
information.

Each operation attaches a local `_backward` function to the output node. During backpropagation, the engine first
constructs a topological ordering of the computation graph, then traverses that ordering in reverse to accumulate
gradients.

For example, an expression such as:

```python
z = (x * y) + (x ** 2)
z.backward()
```

constructs a computation graph during the forward pass and computes the partial derivatives of `z` with respect to `x`
and `y` during the backward pass.

## Validation

The project includes tests for both forward computation and gradient behavior.

The test suite validates:

- scalar arithmetic operations
- mixed operations between `Value` objects and Python numeric types
- right-hand operator overloads such as `2 + value` and `2 * value`
- invalid operation handling
- division and exponentiation behavior
- gradient accumulation through composed expressions
- reverse-mode traversal over computation graphs

These tests are intended to verify both the numerical semantics of the `Value` abstraction and the correctness of
gradient propagation.

## Design Trade-offs

This project deliberately uses scalar values rather than tensors. That decision reduces implementation complexity and
makes the computation graph easier to inspect. The trade-off is that the engine does not support vectorized operations,
broadcasting, matrix multiplication, or tensor-valued gradients.

The implementation also prioritizes conceptual clarity over performance. It is designed as a learning and demonstration
project rather than a production numerical computing library.

## Future Work

Possible extensions include:

- finite-difference gradient checking
- graph visualization
- additional nonlinear primitives such as `sin`, `cos`, `tanh`, `exp`, and `log`
- tensor-valued operations
- broadcasting semantics
- simple neural-network components built on top of the scalar engine

## Purpose

This project demonstrates how reverse-mode automatic differentiation can be implemented from first principles using a
small, inspectable scalar computation graph. It serves as a bridge between mathematical differentiation, graph-based
computation, and the mechanisms used by larger machine learning systems.
