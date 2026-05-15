# Autodiff Engine

## Overview

This project builds toward a small reverse-mode autodiff engine for scalar computation graphs.

## Motivation

Autodiff provides an algorithmic approach to differentiation, turning a central mathematical operation into a practical
computational tool. It is one of the core mechanisms in many modern machine learning libraries, where gradients are
needed for optimization, parameter updates, and model training.

I chose this project because a small scalar autodiff engine is compact enough to inspect directly while still exposing
ideas behind larger systems: primitive operations, computation graphs, local derivative rules, and gradient
accumulation. The goal of this project is to understand the mechanism by which derivatives are computed through program
execution.

This project therefore prioritizes conceptual transparency over feature breadth. By the end of the project, I wanted to
become more comfortable building a mathematically motivated Python project with deliberate scope and appropriate unit
tests.

## Mathematical Concepts

This section will describe the mathematical ideas used by the engine, including computation graphs, local derivatives,
and reverse-mode application of the chain rule.

## Implementation

The implementation is being developed toward a directed acyclic graph representation of scalar expressions. Each
primitive operation constructs a new `Value` object, allowing later stages of the engine to recover dependency structure
during reverse-mode differentiation.

### Operator overloading

Operator overloading allows ordinary Python arithmetic syntax to construct new `Value` objects.

## Experiments and validation

Validation will focus first on the forward semantics of the `Value` abstraction. In particular, tests will verify that
the basic arithmetic operations produce scalar results under both direct and mixed operand usage, including cases where
`Value` appears on either the left-hand or right-hand side of an expression.

Additional tests will examine error handling for unsupported inputs and edge cases, such as invalid exponent types and
division by zero. Later validation stages will focus on graph structure, reverse traversal, and gradient accumulation in
composed expressions.

## Trade-offs

Scalar-first design simplifies the engine. Each node only represents a single numeric value, which makes the computation
graph easier to inspect, but does not support tensor semantics.

## Future Work

Future work may include tensor-valued operations, broadcasting, additional primitive functions, graph visualization, and
finite-difference gradient checks.
