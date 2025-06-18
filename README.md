# MLMI-MPhil-Thesis-MACE
This is a University of Cambridge MLMI MPhil Thesis on **Speeding up MACE**


## Virtual Environement Instructions

Activate the conda venv:
```bash
conda activate mace_gpu_env
```

To install libraries:
```bash
conda ...
```

## HPC Instructions

Interactive nodes on CPU:
```bash
sintr -A MLMI-ab3149-SL2-CPU -p icelake -N1 -n1 -t 1:0:0 --qos=INTR
```

Interactive nodes on GPU:
```bash
sintr -A MLMI-ab3149-SL2-GPU -p ampere --gres=gpu:1 -N1 -n1 -t 1:0:0 --qos=INTR
```

Full Sbatch job:
```bash
sbatch <slurm file>
```

Check the queue position:
```bash
squeue -u ab3149
```

---
## 🚀 How to Run Jupyter Notebook on HPC with Interactive GPU Node

This guide sets up a Jupyter Notebook on a **GPU node** using a **Conda environment with CUDA support**, and allows you to connect from your **local IDE or browser**.

---

### 🖥️ Terminal 1 (Local Terminal): Start Interactive GPU Session

Start an interactive job using SLURM:

```bash
sintr -A MLMI-ab3149-SL2-GPU -p ampere --gres=gpu:1 -N1 -n1 -t 1:00:00 --qos=INTR
```

✅ Wait for the prompt to place you on a GPU node, e.g. `gpu-q-8`.

---

### 🧠 Terminal 2 (Now on the GPU node): Setup Jupyter

1. **Activate your Conda environment**

```bash
conda activate mace_gpu_env
```

2. **Check the hostname** (you’ll need this in the next step)

```bash
hostname
```

Example output: `gpu-q-8`

3. **Launch Jupyter Notebook server**

```bash
jupyter notebook --no-browser --ip=0.0.0.0 --port=8081
```

📌 This will output a URL with a token, like:
`http://127.0.0.1:8081/?token=xxxxxxxxxxxxxxxx`

---

### 🧼 Terminal 3 (Local Machine): Clean up and Setup SSH Tunnel

1. **(Optional)** Free port 8081 if it's blocked

```bash
lsof -ti:8081 | xargs kill -9
```

2. **Create an SSH Tunnel to the GPU Node**

```bash
ssh -L 8081:gpu-q-8:8081 <your-crsid>@login-e-4.hpc.cam.ac.uk
```

🔁 Replace:

* `gpu-q-8` with your actual **GPU node name**
* `login-e-4` with the login node you used

---

### 🌐 Access Jupyter Notebook

* **Option 1**: In browser, go to:
  `http://127.0.0.1:8081/?token=...` (from Terminal 2)

* **Option 2**: In your IDE (e.g. VSCode, PyCharm):

  * Open a `.ipynb` file
  * Select: `Kernel > Select Another Kernel > Enter Jupyter URL`
  * Paste: `http://127.0.0.1:8081/?token=...`

---

### 🧪 Extras

* To view all active notebooks and their tokens:

```bash
jupyter notebook list
```

* To shut down Jupyter cleanly:

```bash
CTRL + C
```

Then confirm `Shutdown this notebook server (y/[n])?` with `y`.

---

## TODO List

### Open questions

- [ ] It seems that they are hundreds of different Tensor config? which one are we using [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html)

### Speeding up

Some GPT suggestions for starting experiments on speeding up inference: [Speeding UP MACE](Notes/General%20Concepts/Speeding_up.md)

### Coding

Coding notes can be accessed in [Tutorial Notes](Notes/Tutorials/T03-MACE-Theory.md)
- [ ] [MACE GitHub Repository](https://github.com/ACEsuit/mace) — try to look through the code and train a MACE-small on a subset of a dataset.
- [x] Running the MACE tutorials #3

### Reading
- [ ] [Computing hydration free energies of small molecules with first principles accuracy](https://arxiv.org/abs/2405.18171)
- [ ] [Stochastic Interpolants: A Unifying Framework for Flows and Diffusions](https://arxiv.org/abs/2303.08797)
- [ ] [Forces are not enough]

### Videos
- [x] [Machine learning potentials always extrapolate, it does not matter](https://www.youtube.com/watch?v=WgFAZygGV8w)
- [x] [Atomic Cluster Expansion: A framework for fast and accurate ML force fields](https://www.youtube.com/watch?v=ja-3UrdSRi4)
- [x] [Orb-v3: atomistic simulation at scale | Tim Duignan & Sander Vandenhaute](https://www.youtube.com/watch?v=pRbvRl0_FyE)
- [x] [Day 3 - Harnessing Geometric ML for Molecular Design | Michael Bronstein](https://www.youtube.com/watch?v=zsIyzLtwAHY)
- [ ] [The Computational Chemist YouTube channel](https://www.youtube.com/@thecomputationalchemist)
- [x] [Open Catalyst Project videos](https://www.youtube.com/@opencatalystproject3509/videos?app=desktop)
- [x] [MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields](https://www.youtube.com/watch?v=I9Y2le9e74A&ab_channel=ValenceLabs)

## Completed Task

9/06/25
- Run the mace tutorials and did a subproject by evaluating the computation efficiency of each step of the mace schematic
- Read MACE paper and notes
- Main thing to look at is the equivarient MPNNs tensor product (equation 8 in paper)

---

10/06/25
- watched: the Machine learning potentials always extrapolate, it does not matter --> can we obtain good prediction accuracy using a fraction of the feature dimensions? yes but wea re still working in extrapolation: the psace is not small enough to gall into the interpolation regime
- watched: Atomic Cluster Expansion: A framework for fast and accurate ML force fields
- watched: Orb-v3: atomistic simulation at scale | Tim Duignan & Sander Vandenhaute
- watched: Day 3 - Harnessing Geometric ML for Molecular Design | Michael Bronstein
- watched all videos from Open Catalyst Project videos
- Questions: why are the videos useful for what i do, they seem quite theoretical but not quite related to understand how to speed up the architecture of MACE? no?
- question: check whether mace has been impleemented to do so(3) to so(2): [video link from open catalyst min 17](https://www.youtube.com/watch?v=Y6JwgqAQpKI&list=PLU7acyFOb6DXgCTAi2TwKXaFD_i3C6hSL&index=6&ab_channel=OpenCatalystProject) and paper link [Reducing SO(3) Convolutions to SO(2) for Efficient Equivariant GNNs](https://proceedings.mlr.press/v202/passaro23a/passaro23a.pdf?utm_source=chatgpt.com)

### 🔧 What SO(2) axis-alignment would add

* **Axis alignment** rotates each edge's coordinate frame so its radial vector maps to a fixed axis.
* This simplifies equivariance to only rotations around that axis—an **SO(2)** problem.
* It **sparsifies** the Clebsch–Gordan tensors, reducing tensor-product complexity from **O(L⁶)** to **O(L³)** ([proceedings.mlr.press][1]).
* Models like **eSCN** already use this trick, but it's **not yet implemented in MACE** out-of-the-box.

---

11/06/25

- coding: did the mace tutorial 1 and 2

---

12/06/25

- read the high precision low accuracy paper
- implement the HALP paper on mace? pleasae refer to [link for HALP](./Notes%20Markdown/General%20Concepts/HALP_to_MACE.md)
- one of the main issues that we need to deal with his HPC. it is probably the worst thing ever and it is impossible to work on it, espcacially when you want to use gpu all the time for testing. when you ask for an interactive node (1h max), and get disconnedted, you have to do the process again and again, it is a major downer

- train a small dataset with H, C and O on default precision fp64 and fp32

🧮 Precision Comparison: FP64 vs FP32 (MACE)

| Metric                           | **FP64 (float64)** | **FP32 (float32)** | 📝 Notes                             |
| -------------------------------- | ------------------ | ------------------ | ------------------------------------ |
| **Train RMSE Energy (meV/atom)** | **5.7**            | 15.2               | FP64 achieves 2.7× lower energy RMSE |
| **Train RMSE Forces (meV/Å)**    | 102.6              | 104.2              | Nearly identical force accuracy      |
| **Train Relative F RMSE (%)**    | 4.71%              | 4.78%              | Negligible difference                |
| **Valid RMSE Energy (meV/atom)** | **6.3**            | 15.7               | FP64 clearly better (2.5× lower)     |
| **Valid RMSE Forces (meV/Å)**    | 177.3              | **162.2**          | FP32 slightly better                 |
| **Valid Relative F RMSE (%)**    | 6.85%              | **6.27%**          | FP32 slightly better                 |
| **Test RMSE Energy (meV/atom)**  | **6.2**            | 14.9               | FP64 better                          |
| **Test RMSE Forces (meV/Å)**     | 176.7              | **169.5**          | FP32 slightly better                 |
| **Test Relative F RMSE (%)**     | 7.71%              | **7.40%**          | FP32 slightly better                 |

---

Resource Usage Comparison

| Metric              | **FP64**  | **FP32**      | 📝 Notes                           |
| ------------------- | --------- | ------------- | ---------------------------------- |
| **Total Time**      | 135.7 s   | **125.2 s**   | FP32 is \~8% faster                |
| **Peak GPU Memory** | 0.41 GB   | **0.42 GB**   | Virtually identical                |
| **CPU RSS Memory**  | 1545.3 MB | **1664.5 MB** | FP32 uses slightly more CPU memory |

---

Summary

| Aspect                         | Winner      | Notes                                           |
| ------------------------------ | ----------- | ----------------------------------------------- |
| **Energy Prediction Accuracy** | 🥇 **FP64** | Consistently lower RMSE across train/valid/test |
| **Force Prediction Accuracy**  | ⚖️ **Tie**  | FP32 slightly better on validation/test forces  |
| **Speed**                      | 🥇 **FP32** | \~10 seconds faster                             |
| **GPU/Memory**                 | ⚖️ **Tie**  | Marginal differences                            |

---

## Homework

- run MACE and print out the shape of all the steps
- kahan summation --> implement anf understand ---> understand
- take the interaction and make a comparispon between fp64/32
- get rid of linear in conv

---

14-15/06/25

- i have set up the dev environment to debug inside the core architecture of mace

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
