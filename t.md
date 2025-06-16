Great! Let’s map your **table components** to the **blocks in the schematic** you posted.
I’ll annotate the table accordingly so you can see where each tensor or shape belongs.

---

### 📊 **Interaction Blocks Table with Schematic References**

| Component             | Block Shape / Dim              | Schematic Block                                                                                                                        |
| --------------------- | ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------- |
| **Node Attributes**   | (N\_nodes, 3)                  | `species one-hot` at top left                                                                                                          |
| **Node Features**     | (N\_nodes, 32)                 | `Node embedding h^{(0)}` → input to linear + interaction                                                                               |
| **Edge Attributes**   | (N\_edges, 16)                 | `Y_lm angular embedding` (Ylm) combined with distance → blue edge attributes                                                           |
| **Edge Features**     | (N\_edges, 8)                  | `radial embedding`  (top blue radial embedding box)                                                                                    |
| **Hidden Irreps**     | 32x0e → 32                     | intermediate latent rep within interaction layers (no direct schematic box, but the internal node feature representation after update) |
| **Edge Irreps**       | 32x0e → 32                     | not explicit in diagram, but part of processed edge latent features                                                                    |
| **Target Irreps**     | 32x0e+32x1o+32x2e+32x3o → 512  | target space of each conv / linear in interaction block (output of `Neighbour sum + linear`, input to `Product`)                       |
| **Linear Up Out**     | (N\_nodes, 32)                 | `Linear` box right after node features                                                                                                 |
| **Conv TP Out (msg)** | (N\_edges, 512)                | `One-particle basis conv_tp` box (before aggregation)                                                                                  |
| **Message sum (agg)** | (N\_nodes, 512)                | `Neighbour sum + linear` box (aggregated message at node level)                                                                        |
| **Final Linear Out**  | (N\_nodes, 512)                | `Final linear` that happens after neighbour sum before product                                                                         |
| **After reshape**     | (N\_nodes, 32, 16) (32×16=512) | output node features `h^{(l+1)}` (green update box at bottom of interaction + product section)                                         |

---

### 🌟 How it flows through the schematic

✅ **Top embedding**

* species one-hot → Node Attributes
* radial embedding → Edge Features
* Ylm → Edge Attributes

✅ **Interaction**

* `Linear` → Linear Up Out
* `conv_tp` → Conv TP Out
* `Neighbour sum + linear` → Message sum + Final Linear Out

✅ **Product**

* `Product` box → hidden powers B, feeding into update
* `Update` → After reshape → updated Node Features

✅ **Readout**

* Final node features feed into node energy output

---

### Example shape walk-through

If `N = 200`, `E = 1972`:

| Stage                   | Example Shape |
| ----------------------- | ------------- |
| Node Attributes         | (200, 3)      |
| Node Features (in)      | (200, 32)     |
| Edge Attributes         | (1972, 16)    |
| Edge Features           | (1972, 8)     |
| Linear Up Out           | (200, 32)     |
| Conv TP Out             | (1972, 512)   |
| Message sum / agg       | (200, 512)    |
| Final Linear Out        | (200, 512)    |
| After reshape (h^{l+1}) | (200, 32, 16) |

---

💡 **Would you like:**
✅ This as a diagram overlay on your schematic?
✅ A LaTeX version of the table for a report?
✅ Annotated schematic with arrows and shapes?

Tell me what format helps most! 🚀
