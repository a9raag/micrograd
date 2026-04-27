# micrograd

A minimal scalar-valued **automatic differentiation (autograd) engine** and neural network library, written in **C++**. Inspired by [Andrej Karpathy's Python micrograd](https://github.com/karpathy/micrograd), this project implements backpropagation over dynamically built computational graphs and demonstrates training a small multi-layer perceptron (MLP) from scratch.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Key Concepts](#key-concepts)
  - [Value – The Autograd Node](#value--the-autograd-node)
  - [Backpropagation](#backpropagation)
  - [Neural Network Layers](#neural-network-layers)
- [Building](#building)
- [Running](#running)
- [Example Usage](#example-usage)
- [Architecture Diagram](#architecture-diagram)

---

## Overview

micrograd is an educational implementation of the core mechanics behind modern deep learning frameworks (like PyTorch) — but reduced to its absolute essentials:

- A `Value` class that wraps a scalar and records the operations applied to it.
- Operator overloading so that arithmetic on `Value` objects transparently builds a computation graph.
- A `backward()` method that walks the graph in reverse topological order and accumulates gradients via the chain rule.
- `Neuron`, `Layer`, and `MLP` classes built on top of `Value` to demonstrate a working neural network.

---

## Project Structure

```
micrograd/
├── include/
│   └── engine.h      # Header: Value class declaration + operator friend declarations
├── engine.cpp        # Implementation of Value and its operations/backprop
├── nn.cpp            # Neural network classes (Neuron, Layer, MLP) + training demo
├── CMakeLists.txt    # CMake build configuration
└── README.md
```

| File | Purpose |
|------|---------|
| `include/engine.h` | Public interface for the `Value` autograd node |
| `engine.cpp` | Core autograd engine: scalar operations, gradient accumulation, `backward()` |
| `nn.cpp` | Neural network building blocks and a full MLP training loop |
| `CMakeLists.txt` | Builds `engine` as a shared library and `nn` as the main executable |

---

## Key Concepts

### Value – The Autograd Node

The central data structure is the `Value` class (`include/engine.h`, `engine.cpp`). Every scalar in the computation is wrapped in a `Value`. Each `Value` stores:

| Field | Type | Description |
|-------|------|-------------|
| `data` | `double` | The forward-pass scalar value |
| `grad` | `double` | Accumulated gradient (∂loss/∂this) |
| `prev` | `set<shared_ptr<Value>>` | Pointers to the operands that produced this node |
| `_op` | `string` | Name of the operation (e.g. `"+"`, `"*"`, `"tanh"`) |
| `node_backward` | `function<void()>` | Lambda that propagates gradient to `prev` nodes |

Supported operations:

| Operation | Method / Operator |
|-----------|------------------|
| Addition | `operator+` |
| Multiplication | `operator*` |
| Subtraction | `operator-` |
| Division | `operator/` |
| Power | `pow(float n)` |
| Hyperbolic tangent | `tanh()` |
| Negation | `neg()` |

All operations return a new `shared_ptr<Value>` and capture the backward closure needed to propagate gradients.

### Backpropagation

`Value::backward()` triggers a full reverse-mode automatic differentiation pass:

1. Sets the gradient of the root node to `1.0`.
2. Builds a **topological ordering** of all reachable nodes using DFS.
3. Iterates the nodes in topological order, calling each node's `node_backward` lambda to accumulate `grad` into its children.

This mirrors how PyTorch's `.backward()` works, but at a single-scalar granularity.

### Neural Network Layers

`nn.cpp` builds a complete neural network on top of the `Value` engine:

#### `Neuron`
- Holds `n_inputs` weight `Value`s (randomly initialized in `[-1, 1]`) and one bias `Value` (initialised to `0`).
- `operator()(inputs)` computes the linear combination: `bias + Σ(w_i * x_i)`.
- `get_params()` returns all weights and the bias as a flat vector — used for gradient descent.

#### `Layer`
- A collection of `n_outs` `Neuron`s, each receiving the same `n_inputs`-dimensional input.
- `operator()(inputs)` runs all neurons and returns a vector of output `Value`s.

#### `MLP` (Multi-Layer Perceptron)
- Accepts `nin` (number of inputs) and `nouts` (list of output sizes per layer).
- Chains layers: input → hidden layers → output.
- `operator()(inputs)` performs the full forward pass.
- `get_params()` collects every weight and bias across all layers.

---

## Building

### Prerequisites

- CMake ≥ 3.0
- A C++14-compatible compiler (GCC, Clang, MSVC)

### Steps

```bash
mkdir build && cd build
cmake ..
cmake --build .
```

This produces:
- `libengine.so` (or `.dylib` / `.dll`) — the shared autograd engine library
- `nn` — the executable that runs the MLP training demo

---

## Running

```bash
./nn
```

The default `main()` in `nn.cpp` calls `test_mlp_large()`, which:

1. Creates a 3 → 4 → 4 → 1 MLP.
2. Runs 100 epochs of gradient descent (MSE loss, learning rate `0.001`) on a 4-sample dataset.
3. Prints the loss after each epoch.

---

## Example Usage

Below is a condensed example of what the engine can do (drawn from `engine.cpp`):

```cpp
#include "include/engine.h"

// Build inputs and weights
auto x1 = make_shared<Value>(2.0);   x1->label = "x1";
auto x2 = make_shared<Value>(0.0);   x2->label = "x2";
auto w1 = make_shared<Value>(-3.0);  w1->label = "w1";
auto w2 = make_shared<Value>(1.0);   w2->label = "w2";
auto b  = make_shared<Value>(6.88);  b->label  = "b";

// Forward pass — builds the computation graph
auto n = (x1 * w1) + (x2 * w2) + b;
auto o = n->tanh();

// Backward pass — computes all gradients
o->backward();

std::cout << "x1 grad: " << x1->get_grad() << std::endl;
std::cout << "w1 grad: " << w1->get_grad() << std::endl;
```

And training an MLP:

```cpp
MLP mlp(3, {4, 4, 1});                      // 3 inputs, two hidden layers of 4, 1 output
auto outputs = mlp(inputs);                 // forward pass
auto loss = compute_mse(outputs, targets);  // scalar loss Value
loss->backward();                           // backprop

for (auto param : mlp.get_params()) {       // gradient descent step
    param->set_data(param->get_data() - lr * param->get_grad());
    param->set_grad(0.0);                   // zero gradients
}
```

---

## Architecture Diagram

```
           ┌──────────────────────────────────────────┐
           │              engine (libengine)           │
           │                                          │
           │   Value ──── data, grad                  │
           │     │        prev (children)             │
           │     │        node_backward (lambda)      │
           │     │                                    │
           │     └── +  *  -  /  pow  tanh  neg       │
           │               │                          │
           │           backward()                     │
           │    (topological sort → chain rule)       │
           └──────────────────┬───────────────────────┘
                              │ uses
           ┌──────────────────▼───────────────────────┐
           │              nn (executable)              │
           │                                          │
           │  Neuron   →  weights[] + bias            │
           │     ↓         linear combination         │
           │  Layer    →  [Neuron, Neuron, ...]       │
           │     ↓                                    │
           │  MLP      →  [Layer, Layer, ...]         │
           │     ↓         forward pass               │
           │  Training loop (MSE loss + SGD)          │
           └──────────────────────────────────────────┘
```
