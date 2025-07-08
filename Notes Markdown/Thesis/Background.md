## 1. Atomistic Simulations and Force Fields

### 1.1 Molecular dynamics and energy evaluation

* Brief history of classical force fields (e.g., Lennard‑Jones, AMBER, CHARMM)
* Limitations in transferability and accuracy

### 1.2 Accuracy vs. computational cost dilemma

* Trade‑offs faced by high‑fidelity quantum methods (DFT, CCSD(T))
* Emergence of surrogate models

### 1.3 Rise of machine‑learning force fields (MLFFs)

* Kernel methods (sGDML, GAP)
* Deep learning approaches (SchNet, DimeNet++, NequIP, Allegro, MACE)
* Motivation for speed and accuracy improvements

---

## 2. From Symmetry to Equivariant Neural Networks

### 2.1 Physical symmetries of atomistic systems

* Translational, rotational, and permutational invariance
* Importance for data efficiency and generalization

### 2.2 Group‑theoretic foundations

* Brief primer on SO(3) and E(3) groups
* Representations, irreducible representations (irreps)

### 2.3 Equivariance in neural architectures

* Definition and benefits
* Examples: Tensor field networks, SE(3)‑Transformers

### 2.4 Implications for MLFFs

* Conservation laws and physical plausibility

---

## 3. Message Passing Neural Networks and the MACE Architecture

### 3.1 Graph representation of atomic configurations

* Nodes = atoms, edges = neighbor interactions
* Neighbor lists and cutoffs

### 3.2 Higher‑order message passing in MACE

* Tensor products and many‑body interactions
* Use of spherical harmonics and Clebsch–Gordan coefficients

### 3.3 Architecture details

* Embedding layer, interaction blocks, output heads
* O(3) equivariance via tensor algebra

### 3.4 Computational complexity analysis

* Scaling with neighbor count, basis size, interaction order
* Identified bottlenecks (tensor product evaluation, memory bandwidth)

### 3.5 Current performance landscape

* Comparison to NequIP, Allegro in accuracy and throughput

---

## 4. Numerical Precision Theory in Scientific Machine Learning

### 4.1 Floating‑point formats and their properties

* IEEE‑754 FP64, FP32, FP16, BF16; integer and posits (brief mention)
* Dynamic range, mantissa length, exponent bias

### 4.2 Rounding error and propagation

* Machine epsilon, catastrophic cancellation
* Impact on iterative algorithms and stability

### 4.3 Mixed‑precision training vs. inference

* Loss scaling, gradient underflow/overflow
* Framework support (PyTorch Automatic Mixed Precision, NVIDIA Tensor Cores)

### 4.4 Low‑precision in geometric deep learning

* Quantization of weights, activations, and message tensors
* Specific challenges for equivariant operations (CG products)

### 4.5 Hardware considerations

* GPU/TPU tensor units, custom ASICs
* Memory bandwidth vs. arithmetic throughput

---

## 5. Strategies for Accelerating Equivariant GNNs

### 5.1 Kernel fusion and memory‑efficient implementations

* Fused tensor product kernels, efficient neighbor gathering

### 5.2 Basis compression and low‑rank approximations

* Reducing spherical harmonic order, irreps pruning

### 5.3 Quantization, pruning, and distillation

* Weight sharing, structured sparsity

### 5.4 Sparse neighbor lists and dynamic cutoffs

* Adaptive sampling of interaction graph

### 5.5 Comparative evaluation of speed‑accuracy trade‑offs in literature

---

## 6. Experimental Landscape and Benchmarks

### 6.1 Datasets

* QM9, MD17, SPICE, OC20, ISO17, ANI‑1x, etc.

### 6.2 Evaluation metrics

* MAE on energies, forces; wall‑clock throughput; memory footprint

### 6.3 Baseline models

* Short descriptions of SchNet, DimeNet++, PhysNet, NequIP, Allegro, GPUs vs. TPUs

### 6.4 Existing attempts to speed up MACE

* (e.g., JaxMACE, 8‑bit inference experiments, tensor‑core optimizations)

---

## 7. Synthesis: Research Gap and Thesis Objectives

### 7.1 Summary of limitations in current state‑of‑the‑art

* High memory footprint of high‑order tensor products
* Limited support for sub‑FP32 precision in equivariant libraries

### 7.2 Hypothesis and research questions

* Can mixed/low‑precision inference significantly accelerate MACE without degrading accuracy?
* What algorithmic or architectural modifications are required to maintain equivariance under quantization?

### 7.3 Expected contributions

* Theoretical analysis of numerical precision effects in equivariant layers
* Implementation of low‑precision MACE kernels
* Empirical benchmarks demonstrating speedups and accuracy retention

### 7.4 Chapter road‑map

* How each subsequent chapter builds upon the background

---

**Note**: Add citations inline (e.g., @Batatia2022\_MACE) once bibliography is integrated. Consider including illustrative figures such as (1) symmetry transformations, (2) MACE interaction block diagram, (3) error vs. precision curves for tensor products.
