# Background

## 1. Atomistic Simulations and Force Fields

### 1.1 Molecular Dynamics and Energy Evaluation

1. Brief history of classical force fields (e.g., Lennard‑Jones, AMBER, CHARMM)
2. Limitations in transferability and accuracy

### 1.2 Accuracy vs. Computational Cost Dilemma

1. Trade‑offs faced by high‑fidelity quantum methods (DFT, CCSD(T))
2. Emergence of surrogate models

### 1.3 Rise of Machine‑Learning Force Fields (MLFFs)

1. Kernel methods (sGDML, GAP)
2. Deep learning approaches (SchNet, DimeNet++, NequIP, Allegro, MACE)
3.  Motivation for speed and accuracy improvements

*Add a brief sentence that say that elaborating on studying the background of the passage from symmetry to equivariant Neural Network is beyond the scope of this project

## 2. Introducing the MACE architecture

### 2.1 Graph representation of atomic configurations

Nodes = atoms, edges = neighbor interactions

Neighbor lists and cutoffs

### 2.2 Higher‑order message passing in MACE

Tensor products and many‑body interactions

Use of spherical harmonics and Clebsch–Gordan coefficients

### 2.3 Architecture details

Embedding layer, interaction blocks, output heads

O(3) equivariance via tensor algebra

### 2.4 Computational complexity analysis

Scaling with neighbor count, basis size, interaction order

Identified bottlenecks (tensor product evaluation, memory bandwidth)

### 2.5 Current performance landscape

Comparison to NequIP, Allegro in accuracy and throughput

## Approaches and Methods for evaluating and accelerating machine learing architecture

## Numerical Precision theory and impact on tensor computations and results

**Layout**
1. Here the goal is to explain the computer science theory of numerical precision and what it implies
2. we should probably derive, somehow, how tensor product get's lower precision in math

## Evaluation Metrics

1. Abs error
2. Relative abs error
3. RMSE
4. MAE
5. etc...