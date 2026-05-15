# Autodiff Engine

## Motivation

This project prioritizes conceptual transparency over feature breadth.

## Mathematical Concepts

## Implementation

The implementation will utilize a computational graph structure to utilize DAG
properties through dynamic programming principles.

The core abstraction used is the Value object.

## Experiments / Validation

## Trade-offs

Scalar-first design simplifies the engine. Each node only represents a single numeric value and its local gradient relations. The implementation is easier to inspect, but does not support tensor semantics.

## Future Work

Since the features of this automatic differentiation implentation are realtively
small in scope, the future work can focus on added the ability to use tensors or broadcasting.
This will
