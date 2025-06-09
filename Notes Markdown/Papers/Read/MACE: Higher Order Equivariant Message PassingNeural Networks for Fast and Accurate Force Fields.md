# MACE: Higher Order Equivariant Message PassingNeural Networks for Fast and Accurate Force Fields

Link: [MACE](https://arxiv.org/abs/2206.07697)

## Main outcome

Using four-body messages reduces therequired number of message passing iterations to just two, resulting in a fast andhighly parallelizable model, reaching or exceeding state-of-the-art accuracy on therMD17, 3BPA, and AcAc benchmark tasks.

## Introduction

- As a result of the increased body order of the messages, only two messagepassing iterations are necessary to achieve high accuracy - unlike the typical ﬁve or six iterationsof MPNNs, making it scalable and parallelizable.	
- 30 mins training of NVIDIA A100 GPU

## Background

### MPNN Interatomic Potentials

- graph embedded in 3D Euclidean space
- Node —> Atom
- Edges connect nodes (atoms) if the corresponding atoms are within a given distance of each othe

Below is a concrete walkthrough of those MPNN equations using a **simple diatomic system** (two atoms) to show exactly how messages, updates, and readouts work in an interatomic potential model.

#### 1. Node state

At each layer $t$, each atom $i$ has a state

$$
\sigma_i^{(t)} = \bigl(r_i,\; z_i,\; h_i^{(t)}\bigr)
$$

* $r_i\in\mathbb R^3$: 3D position
* $z_i$: element type (e.g. "O", "H")
* $h_i^{(t)}\in\mathbb R^d$: learnable feature vector

#### Example initialization

* Atom 1 (Oxygen):
  $\;r_1=[0,0,0],\quad z_1=\mathrm{O},\quad h_1^{(0)}=\mathrm{Embed}(\mathrm{O})\in\mathbb R^d$
* Atom 2 (Hydrogen):
  $\;r_2=[1,0,0],\quad z_2=\mathrm{H},\quad h_2^{(0)}=\mathrm{Embed}(\mathrm{H})\in\mathbb R^d$

---

#### 2. Message construction

At layer $t$, each atom $i$ gathers messages from its neighbors $\mathcal N(i)$.  For our two-atom system, each atom's only neighbor is the other atom:

$$
m_i^{(t)}
=\bigoplus_{j\in\mathcal N(i)}
M_t\!\bigl(\sigma_i^{(t)},\,\sigma_j^{(t)}\bigr)
\quad\text{(e.g.\ sum over \(j\))}
$$

#### Example at $t=0$

* Distance vector:
  $\displaystyle r_{12}=r_2 - r_1 = [1,0,0],\quad \|r_{12}\|=1$
* Message function $M_0$ could be e.g.

  $$
    M_0\bigl(\sigma_1^{(0)},\sigma_2^{(0)}\bigr)
    = \mathrm{MLP}\!\bigl([\,h_1^{(0)}\!,\,h_2^{(0)}\!,\,\|r_{12}\|\,]\bigr)
    \;\in\mathbb R^p
  $$
* So atom 1's message:

  $$
    m_1^{(0)} = M_0(\sigma_1^{(0)},\sigma_2^{(0)})
             \quad(\text{only neighbor }j=2)
  $$
* And symmetrically for atom 2:
  $m_2^{(0)} = M_0(\sigma_2^{(0)},\sigma_1^{(0)})$.

---

#### 3. State update

Use each message to update the node features:

$$
h_i^{(t+1)}
= U_t\!\Bigl(\,h_i^{(t)},\,m_i^{(t)}\Bigr)
\quad
\Bigl(\text{e.g.\ }U_t([h,m])=\mathrm{ReLU}(W[h; m]+b)\Bigr)
$$

#### Example at $t=0\to1$

* Concatenate $[\,h_i^{(t)},\,m_i^{(t)}]\in\mathbb R^{d+p}$
* Apply a small MLP or gated update
* Result is the next-layer feature $h_i^{(t+1)}\in\mathbb R^d$

---

#### 4. Readout (site energies)

After $T$ layers, each atom's state $\sigma_i^{(T)}$ is mapped to a **site energy** via

$$
E_i
= \sum_{t=1}^T R_t\bigl(\sigma_i^{(t)}\bigr)
=\sum_{t=1}^T R_t\bigl(h_i^{(t)}\bigr)
\quad\bigl(R_t:\mathbb R^d\to\mathbb R\bigr).
$$

The **total energy** is then

$$
E_{\rm total} = \sum_i E_i.
$$

#### Example readout

* Each layer contributes $R_t\bigl(h_i^{(t)}\bigr)$, e.g.\ a linear map
* For two atoms,

  $$
    E_{\rm total}
    = E_1 + E_2
    = \sum_{t=1}^T \bigl[R_t(h_1^{(t)}) + R_t(h_2^{(t)})\bigr].
  $$

---

#### ▶︎ Putting it all together (for two atoms)

1. **Initialize**
   $(r_i,z_i,h_i^{(0)})$ for $i=1,2$.
2. **For $t=0,\dots,T-1$:**

   * **Message**
     $m_1^{(t)} = M_t(\sigma_1^{(t)},\sigma_2^{(t)})$,
     $m_2^{(t)} = M_t(\sigma_2^{(t)},\sigma_1^{(t)})$.
   * **Update**
     $h_i^{(t+1)} = U_t\bigl(h_i^{(t)},m_i^{(t)}\bigr)$.
3. **Readout**
   $E_i = \sum_{t=1}^T R_t\bigl(h_i^{(t)}\bigr)$,
   $E_{\rm total}=E_1+E_2$.

This same pattern generalizes to any number of atoms: **pool messages over all neighbors**, **update each node**, then **read out** local energies and sum.

<div style="background-color: #77C618; color: black; padding: 15px; border-radius: 5px;">

**Point of observation**

| Step | Operation | Safe for FP16/BF16? | Why? |
|------|-----------|---------------------|------|
| **Node embeddings** $h_i^{(0)} = \mathrm{Embed}(z_i)$ | Table lookup or learned MLP | ✅ Yes | Typically small, static; fast and safe in FP16 |
| **Message function** $M_t(\sigma_i, \sigma_j)$ | MLP over pairwise features | ✅ Yes | Dominated by tensor ops; good FP16 acceleration |
| **Distance norms** $\|\mathbf{r}_j - \mathbf{r}_i\|$ | Vector subtraction + norm | ⚠️ Prefer FP32 | Risk of cancellation for small displacements |
| **Angle & basis computation** $Y_\ell^m(\hat{r}_{ij})$ | Trig ops (acos, atan2) | ⚠️ FP32 preferred or post-cast | Numerical errors if computed in FP16, but can be cast down afterwards |
| **Message pooling** $\sum_j M_t(\cdot)$ | Perm-invariant sum over neighbors | ⚠️ Use FP32 for sum | Prevents underflow/cancellation in large molecules |
| **Feature updates** $h_i^{(t+1)} = U_t(h_i^{(t)}, m_i^{(t)})$ | MLP or GRU block | ✅ Yes | Standard tensor ops; good fit for FP16/BF16 |
| **Readout MLPs** $R_t(h_i^{(t)})$ | Maps to site energies | ✅ Yes | Scalar outputs → well-behaved |
| **Energy accumulation** $E = \sum_i E_i$ | Global reduction | ⚠️ Must be FP32 | Errors accumulate if summed in FP16 |
| **Force computation** $-\nabla_{r_i} E$ | Backprop/autograd | ⚠️ AMP + loss scaling needed | Gradients prone to underflow in FP16 |

**Rules of thumb**: apply low precision when values are bounded and errors don't accumulate, but retain high precision where precision compounds
</div>

### Equivarient GNN

- the group of interest is 0(3) equivarient if it has internal features that transform under the rotation $\mathbf{Q} \in O(3)$

#### 🧠 1. Goal: Symmetry-aware GNNs

When modeling atomic systems, **physical properties like energy and force must respect symmetry**:

* Rotating or reflecting the molecule shouldn’t change the predicted energy.
* Force vectors must rotate in the same way as the atoms.

To capture this, we build **equivariant neural networks** — models where the **features inside the network transform predictably under symmetry operations**.

---

#### 🔁 2. What is O(3)?

The **orthogonal group O(3)** includes:

* **Rotations** in 3D space (SO(3))
* **Reflections** (i.e. parity inversion $\mathbf{x} \to -\mathbf{x}$)

These represent the **rigid motions** of 3D space that preserve distances.

---

#### 🏗️ 3. What Does Equivariance Mean Here?

A GNN is **O(3) equivariant** if the node features transform under O(3) in a well-defined way.

The condition is:

$$
h_i^{(t)}(Q \cdot (r_1, \ldots, r_N)) = D(Q) \, h_i^{(t)}(r_1, \ldots, r_N)
$$

* $h_i^{(t)}$: feature vector at atom $i$, layer $t$. $t$ refers to a step in the message passing process: each layer updates features from the previous one:
    $$ h_i^{t+1} = U_t(h_i^{(t)}, m_i^{(t)})$$
* so $t=0$ is the input embeddings (from atomic number of chemical species) and $t=1,2,\ldots,T$ are intermediate message passing layers where after $T$ layers we read out the final features. We can think of t as time steps in an information-exchange process across the graph
* $Q \in O(3)$: a rotation or reflection
* $Q \cdot (r_1, ..., r_N)$: apply rotation $Q$ to all atom positions
* $D(Q)$: representation matrix of $Q$ acting on the feature vector

So:

> If you rotate all the atoms, the internal features rotate **with them** — no change in model behavior, just a change in coordinates.

This is exactly what makes models like **MACE, NequIP, SEGNN** special — they build this **equivariance** into their layers using **group theory**.

---

#### 🔍 4. What Is $h_{i,kLM}^{(t)}$?

This is a **structured version of the feature vector** $h_i^{(t)}$, where each part is labeled by:

| Symbol | Meaning                                           |
| ------ | ------------------------------------------------- |
| $k$    | channel index (like “filter number”)              |
| $L$    | angular momentum (i.e., how it transforms)        |
| $M$    | component of the irrep, ranging from $-L$ to $+L$ |

So:

* $L = 0$: scalar — does **not change** under rotation ⇒ invariant
* $L = 1$: vector — rotates like a 3D vector
* $L = 2$: rank-2 tensor — transforms like a quadrupole, etc.

---

#### 🧮 5. How Does It Transform?

This equation:

$$
h_{i,kLM}^{(t)}(Q \cdot \{r_j\}) = \sum_{M'} D_{M'M}^{(L)}(Q) \, h_{i,kLM'}^{(t)}(\{r_j\})
$$

says that **each L-block of the feature vector transforms via a Wigner D-matrix** $D^{(L)}(Q)$.

#### This is exactly:

* The **irreducible representation** of O(3) for degree $L$
* Each $L$-block has $2L+1$ components (e.g., 3 for $L=1$, 5 for $L=2$)
* The Wigner D-matrix mixes these components under rotation
* $k$ is the feature dimension within each irrep type where k is a parallel copy of the same type of angular feature enabling more expressive learning.

---

#### 🧪 Quick Example

Suppose we have a single atom with a feature:

$$
h_i^{(t)} = 
\begin{bmatrix}
\underbrace{0.8}_{L=0\text{ scalar}} \\
\underbrace{[1.0,\ 0.0,\ 0.0]}_{L=1\text{ vector}} \\
\underbrace{[...] }_{L=2\text{ tensor}}
\end{bmatrix}
$$

If we apply a rotation $Q$ (say 90° around z), the updated features will be:

* Scalar stays the same
* Vector becomes $Q \cdot [1.0, 0.0, 0.0] = [0.0, 1.0, 0.0]$
* Tensor components get rotated via a 5×5 Wigner matrix $D^{(2)}(Q)$

### The MACE architecture


#### 🔧 **1. General Setup: MACE Extends MPNNs**

* The MACE model builds on the **Message Passing Neural Network (MPNN)** framework.
* Its **core innovation** lies in how **messages** are constructed during message passing.

---

#### 🧠 **2. Hierarchical Message Construction**

* Messages $m_i^{(t)}$ are computed using a **hierarchical expansion** over increasing interaction orders:

  $$
  m_i^{(t)} = \sum_j u_1(\sigma_i^{(t)}, \sigma_j^{(t)}) + \sum_{j_1,j_2} u_2(\sigma_i^{(t)}; \sigma_{j_1}^{(t)}, \sigma_{j_2}^{(t)}) + \cdots
  $$

  * Each $u_\nu$ is a **learnable function** capturing ν-body interactions (e.g., pairwise, triplet, etc.).
  * $\sigma_i^{(t)}$ are node features at layer $t$.
  * $\nu$ controls the **maximum correlation order**.

##### Key Point:

* This includes **self-interactions** (e.g., $j_1 = j_2$) in the sums, which simplifies later tensor operations and **avoids the exponential cost** of enumerating all combinations like in DimeNet.

---

#### 🧱 **3. Edge Embeddings and 2-Body Features (Equation 8)**

* They embed directional information using:

  * **Radial basis functions** $R$ (learned with MLPs).
  * **Spherical harmonics** $Y_l^m$ for angular parts.
  * **Clebsch-Gordan coefficients** $C$ to maintain SO(3) **equivariance**.
* The 2-body feature is:

  $$
  A^{(t)}_{i,kl_3m_3} = \sum_{j \in N(i)} \sum_{l_1m_1,l_2m_2} C^{l_3m_3}_{l_1m_1,l_2m_2} R(r_{ji}) Y_{l_1}^{m_1}(\hat{r}_{ji}) W h_j^{(t)}
  $$

  * This encodes both radial and angular structure of local atomic environments.

---

#### 🧮 **4. Higher-Order Features via Tensor Products (Equation 10)**

* Higher-order interactions are formed from **tensor products** of the 2-body features $A^{(t)}_i$, then projected to specific angular components using **generalized Clebsch-Gordan coefficients**.

  $$
  B^{(t)}_{i,\eta^\nu kLM} = \sum_{lm} C_{LM}^{\eta^\nu,lm} \prod_{\xi=1}^\nu \sum_{\tilde{k}} w^{(t)}_{k k^{\tilde{\xi}}} A^{(t)}_{i,k^{\tilde{\xi}}m_\xi}
  $$

  * $\nu$ is the **order of the interaction** (triplet, quadruplet, etc.).
  * $C$ selects combinations that transform correctly under SO(3).
  * Efficiently computed because these coefficients are sparse and precomputable.

---

#### ✉️ **5. Final Message Construction (Equation 11)**

* The actual message $m^{(t)}_i$ is a **linear combination** of all these higher-order $B^{(t)}$ features:

  $$
  m^{(t)}_{i,kLM} = \sum_\nu \sum_{\eta^\nu} W_{zikL,\eta^\nu}^{(t)} B^{(t)}_{i,\eta^\nu kLM}
  $$

  * Indexed by receiving atom type $z_i$, body order $\nu$, and equivariance order $L$.
  * This is the efficient realization of the body-ordered expansion in Equation (7).

---

#### 🔁 **6. Update Step (Equation 12)**

* Node features are updated linearly using:

  $$
  h^{(t+1)}_{i,kLM} = \sum_{\tilde{k}} W_{kL,\tilde{k}}^{(t)} m^{(t)}_{i,kLM} + \sum_{\tilde{k}} W_{z_i kL,\tilde{k}}^{(t)} h^{(t)}_{i,kLM}
  $$

  * Includes **residual connections** to stabilize training.

---

#### 📤 **7. Readout Phase (Equation 13)**

* Energy for each atom $E_i$ is obtained by summing contributions across all layers.

  * For $t < T$: use a simple linear mapping on **invariant** features.
  * For final layer $t = T$: use a small **MLP** on invariant features.

---

<div style="background-color: #77C618; color: black; padding: 15px; border-radius: 5px;">

**Point of observation**

---

### ⚙️ **Low-Precision Strategies in MACE**

| **Component**                     | **Role in MACE**                        | **Precision Strategy**                                                                           | **Benefits**                               | **Risks & Mitigations**                                                          |
| --------------------------------- | --------------------------------------- | ------------------------------------------------------------------------------------------------ | ------------------------------------------ | -------------------------------------------------------------------------------- |
| **Tensor Product (Eq. 10)**       | Build higher-order equivariant features | Use `float16`/`bfloat16` for inputs and multiplications; precompute CG coefficients in `float32` | Major speedup, large memory savings        | Loss of equivariance if CG coefficients are low precision → keep CG in `float32` |
| **Radial Basis + MLP**            | Embed interatomic distances             | Safe to use `float16`/`bfloat16`                                                                 | Fast computation, stable gradients         | Minimal risk if smooth basis; validate numerical range                           |
| **Spherical Harmonics $Y_l^m$**   | Encodes directionality                  | Use `float32` or analytically stable low-precision basis                                         | Ensures SO(3) equivariance                 | Use `float32` to avoid angular artifacts                                         |
| **Message Aggregation (Eq. 11)**  | Combine body-ordered tensors            | Mixed precision (low-precision inputs, float32 accumulators)                                     | High performance, still numerically stable | Use PyTorch AMP for gradient scaling                                             |
| **Node Feature Updates (Eq. 12)** | Residual update step                    | Mixed precision recommended                                                                      | Efficient backpropagation                  | Risk of exploding/vanishing gradients → apply loss scaling                       |
| **Readout MLP (Eq. 13)**          | Predict site energies                   | Fully use `float16` or quantized INT8                                                            | Safe, fast inference                       | None if only invariant inputs used                                               |
| **Clebsch–Gordan Coefficients**   | Enforce equivariance structure          | Keep in `float32`, use as constants                                                              | Guarantees SO(3) structure                 | Must avoid quantization or rounding error                                        |
| **Training Pipeline**             | Whole training loop                     | Mixed-precision training (`autocast`, `GradScaler`)                                              | 1.5–2x speedup, less memory                | Underflow → use loss scaling                                                     |

</div>

### Scaling and Computatinoal Cost

#### 1. Equivariant Tensor Product Cost	

Tensor product operations (especially Eq. 8) are expensive due to high angular resolution and edge-level computation	MACE computes expensive edge-based product once, then does node-based contractions (Eq. 10) using loop tensor contractions (faster)	This is still a major compute load — ideal place to apply low-precision optimization, kernel fusion, or custom CUDA ops

#### 2. Parallelism and Training Time Gains
Observation: MACE with L=0 is ~10× faster than NequIP/BOTNet while maintaining accuracy.

Training speedup: MACE reaches BOTNet-level accuracy in 30 mins vs 24+ hrs.

Reason: Its tensor-based message formulation + small receptive field reduces GPU memory contention and communication overhead → better scaling across multiple GPUs.

<div style="background-color: #77C618; color: black; padding: 15px; border-radius: 5px;">

**Point of observation**

## 🔬 **Low-Precision Training: What Can Be Tested**

| **Test/Strategy**                             | **Where to Apply**                         | **Expected Benefit**                            | **Tools to Use**                                                             |
| --------------------------------------------- | ------------------------------------------ | ----------------------------------------------- | ---------------------------------------------------------------------------- |
| **Mixed-Precision Training**                  | Everywhere except CG and SH ops            | 2–3× speedup, lower memory                      | `torch.cuda.amp` (PyTorch) or `jax.lax.precision`                            |
| **Low-Precision Tensor Contractions**         | Eq. (10) — node-level loop contractions    | Significant speedup at L ≥ 1 or ν ≥ 2           | Implement custom fused kernels or use `Triton`, `XLA`, `NVIDIA Tensor Cores` |
| **Keep Clebsch–Gordan / SH Terms in Float32** | Tensor ops involving rotation-equivariance | Avoids equivariance-breaking artifacts          | Store CG/Ylm tables as `float32` constants                                   |
| **Loss Scaling**                              | During backpropagation                     | Prevents underflow in float16 gradients         | Use `GradScaler` in PyTorch                                                  |
| **Quantize MLP Readout Layers**               | Final site energy prediction               | 8-bit quantization viable without accuracy loss | Use `torch.quantization` or ONNX INT8

</div>

<div style="background-color: #77C618; color: black; padding: 15px; border-radius: 5px;">

## 🧪 **Future/Experimental Ideas**

You could also explore:

1. **Layer-wise Precision Scaling**:

   * Use float16 for readout and intermediate MLPs.
   * Use float32 for initial feature computation and CG terms.

2. **Progressive Correlation Order**:

   * Start training with ν = 1, then progressively raise it.
   * Benefit: lower early-phase compute + stable convergence.

3. **Neural Compression of Species Embeddings**:

   * Replace fixed species embeddings with **low-rank or quantized embeddings**.
   * Especially helpful for large-S datasets (e.g., periodic table-sized chemical space).

4. **Operator Fusion + Kernel Optimization**:

   * Fuse radial, angular, and contraction steps to reduce memory reads/writes.
   * Implement with Triton or TVM for custom GPUs.               

</div>