MACE’s convolution is **fundamentally SO(3)-equivariant**, meaning it respects all 3D rotations—but it does **not currently** implement the **SO(2) axis-alignment trick** described in Passaro et al. (2023) and used in eSCN.

---

### ✅ What MACE does

* Uses **full SO(3) tensor products** (via Clebsch–Gordan algebra) for equivariant message passing—no explicit axis alignment is performed ([proceedings.mlr.press][1], [papers.neurips.cc][2]).
* Computation remains faithful to 3D rotations throughout, with complexity typical for SO(3) operations.

---

### 🔧 What SO(2) axis-alignment would add

* **Axis alignment** rotates each edge's coordinate frame so its radial vector maps to a fixed axis.
* This simplifies equivariance to only rotations around that axis—an **SO(2)** problem.
* It **sparsifies** the Clebsch–Gordan tensors, reducing tensor-product complexity from **O(L⁶)** to **O(L³)** ([proceedings.mlr.press][1]).
* Models like **eSCN** already use this trick, but it's **not yet implemented in MACE** out-of-the-box.

---

### 🆚 Comparison

| Feature                           | MACE (current)  | MACE w/ SO(2) axis-alignment |
| --------------------------------- | --------------- | ---------------------------- |
| Equivariance Group                | SO(3)           | SO(3) → executed as SO(2)    |
| Tensor-product computational cost | Standard, dense | Sparse, more efficient       |
| Implementation in code            | ✔️ Yes          | ❌ Not yet                    |
| Used in other models              | —               | ✔️ eSCN                      |

---

### 🧪 So… Is SO(2) implemented yet in MACE?

**Not currently** in the official JESU/Torch MACE repository. While the **conceptual exponent reduction** to SO(2) is well-known and used in eSCN, it **hasn't been integrated** into the standard MACE codebase.

However, the architecture is well-structured, so you *could* prototype this optimization—particularly within the `conv_tp` operation, by:

1. Rotating input edge data so that radial vectors align to a canonical axis,
2. Applying the tensor product in this reduced frame,
3. Optionally rotating outputs back (or keeping consistent with subsequent operations).

Would you like help sketching or implementing that optimization?

[1]: https://proceedings.mlr.press/v202/passaro23a/passaro23a.pdf?utm_source=chatgpt.com "[PDF] Reducing SO(3) Convolutions to SO(2) for Efficient Equivariant GNNs"
[2]: https://papers.neurips.cc/paper_files/paper/2022/file/4a36c3c51af11ed9f34615b81edb5bbc-Paper-Conference.pdf?utm_source=chatgpt.com "[PDF] MACE: Higher Order Equivariant Message Passing Neural Networks ..."
