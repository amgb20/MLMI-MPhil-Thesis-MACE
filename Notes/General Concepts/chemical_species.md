# Chemical Species

**Chemical species** refers to the **type of atom or element** present in the molecular system being modeled.

### 🔬 Definition in MACE Context:

A **chemical species** typically means a distinct atomic **element**, such as:

* Hydrogen (H),
* Carbon (C),
* Oxygen (O),
* Nitrogen (N),
* etc.

Each chemical species is treated differently in the model because:

* Different species have different nuclear charges,
* They interact differently with neighboring atoms,
* They may have different learnable embeddings (in neural models like MACE),
* Their interactions contribute differently to the energy and force predictions.

> **Example** with a water molecule ($H_2O$) where the molecule is represented with 3 nodes: H (node 1), H (nodes 2) and O (node 3).
> Mace assigns an embedding vector to each chemical species where:


### 🧠 In the MACE Architecture:

MACE uses **equivariant message passing** to model how atoms interact. To do this effectively:

* Each chemical species is given a **learned embedding vector**

    - embedding_H = f(H) $\in$ $\mathbb{R}^d$

    - embedding_O = f(O) $\in$ $\mathbb{R}^d$
* These embeddings are used to **initialize node features** in the graph,
* The model learns **how different species interact** under physical constraints (e.g., rotational equivariance using SO(3)).

### 🚀 Why It Matters:

Including information about the **chemical species** allows MACE to:

* Generalize across molecules with different atoms,
* Capture species-specific behavior (e.g., how oxygen bonds vs. hydrogen),
* Enable transfer learning to unseen molecules containing known species.

Let me know if you’d like to see how chemical species embeddings are encoded in code or equations in MACE, King Julian.
