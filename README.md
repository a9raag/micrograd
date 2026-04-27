# micrograd

A minimal **automatic differentiation (autograd) engine** and tensor playground written in **C++ with CUDA**. This branch evolves the original scalar-only micrograd implementation into a tensor-based project with CUDA-backed compute kernels, a tensor-aware `Value` graph, and experiments for bigram models and small neural networks.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Key Concepts](#key-concepts)
  - [Tensor-backed Value graph](#tensor-backed-value-graph)
  - [CUDA compute backends](#cuda-compute-backends)
  - [Example experiments](#example-experiments)
- [Building](#building)
- [Running](#running)
- [Dataset](#dataset)
- [Notes](#notes)

---

## Overview

This branch keeps the educational spirit of micrograd while changing the implementation significantly:

- `Value` now stores `Tensor<float>` objects instead of scalar `double` values.
- Tensor operations are delegated to `Compute1D` and `Compute2D` backends.
- CUDA kernels power tensor math such as elementwise ops, reductions, indexing, and matrix multiplication.
- Example entry points in `src/main.cu` exercise tensor math, autograd, and a character-level bigram training loop.

Because this is a CUDA-first branch, it is intentionally separate from the original scalar API on `main`.

---

## Project Structure

```text
micrograd/
├── CMakeLists.txt
├── README.md
├── data/
│   └── names.txt
└── src/
    ├── compute1d.cu
    ├── compute2d.cu
    ├── cuda_compute.cu
    ├── data.cpp
    ├── engine.cu
    ├── helper.cpp
    ├── main.cu
    ├── nn.cpp
    ├── tensor.cu
    └── include/
        ├── base_compute.h
        ├── compute1d.h
        ├── compute2d.h
        ├── data.h
        ├── engine.h
        ├── helper.h
        └── tensor.h
```

| File | Purpose |
|------|---------|
| `src/include/engine.h`, `src/engine.cu` | Tensor-aware autograd node implementation |
| `src/include/tensor.h`, `src/tensor.cu` | Tensor container, operators, reshaping, reductions, and indexing |
| `src/include/compute1d.h`, `src/compute1d.cu` | 1D CUDA compute backend |
| `src/include/compute2d.h`, `src/compute2d.cu` | 2D CUDA compute backend |
| `src/cuda_compute.cu` | CUDA kernels shared by the compute backends |
| `src/data.cpp` | Simple dataset reader for the names corpus |
| `src/nn.cpp` | Experimental layer and MLP helpers |
| `src/main.cu` | Manual smoke tests and training/demo entry points |

---

## Key Concepts

### Tensor-backed Value graph

The `Value` class records tensor operations and attaches a backward lambda to each output node. The graph is traversed in reverse topological order by `backward()`, just like the scalar implementation on `main`, but gradients now flow through tensor operations such as:

- elementwise addition, subtraction, multiplication, and division
- `pow`, `tanh`, `relu`, `sigmoid`, `exp`, and `log`
- reductions like `sum()` and `mean()`
- matrix multiplication via `dot()`
- tensor slicing with `subTensor()`

### CUDA compute backends

`Tensor<T>` delegates storage and math to a `BaseCompute<T>` implementation selected from the tensor rank:

- `Compute1D<T>` for vectors
- `Compute2D<T>` for matrices

These classes use CUDA-managed allocations and launch kernels from `src/cuda_compute.cu` for math and indexing operations.

### Example experiments

`src/main.cu` contains small entry points for:

- tensor math smoke tests
- autograd experiments
- data loading checks
- a bigram probability model
- a bigram neural-network-style training loop

---

## Building

### Prerequisites

- CMake >= 3.18
- A C++17 compiler
- A working CUDA toolkit with `nvcc` available to CMake

### Configure and build

```bash
cmake -S . -B build
cmake --build build
```

If CUDA is not installed, CMake still configures successfully but skips the executable target and prints a warning.

When CUDA is available, the build produces:

- `micrograd` — the CUDA demo executable

---

## Running

The executable accepts a small command selector:

```bash
./build/micrograd tensor2d
./build/micrograd data
./build/micrograd bigram-probability
./build/micrograd bigram-nn
```

If no argument is provided, the binary runs the lightweight `tensor2d` smoke test.

---

## Dataset

The bundled names corpus lives at:

```text
/data/names.txt
```

The executable resolves this file from the repository source directory at compile time, so it works from an out-of-tree build directory without requiring hard-coded absolute paths.

---

## Notes

- This branch is a large architectural change relative to `main`, not a drop-in extension of the scalar-only implementation.
- The demo code in `src/main.cu` is still intended for experimentation and manual verification.
- The CUDA target is optional at configure time so the repository remains inspectable on systems without a CUDA toolkit.
