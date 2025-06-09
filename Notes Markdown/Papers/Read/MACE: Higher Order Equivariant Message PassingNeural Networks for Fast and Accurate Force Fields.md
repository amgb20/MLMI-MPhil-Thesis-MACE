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