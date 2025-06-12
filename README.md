# MLMI-MPhil-Thesis-MACE
This is a University of Cambridge MLMI MPhil Thesis on **Speeding up MACE**


## Virtual Environement Instructions

Activate the conda venv:
```
conda activate mace_gpu_env
```

To install libraries:
```
conda ...
```

## HPC Instructions

Interactive nodes on CPU:
```
sintr -A MLMI-ab3149-SL2-CPU -p icelake -N1 -n1 -t 1:0:0 --qos=INTR
```

Interactive nodes on GPU:
```
sintr -A MLMI-ab3149-SL2-GPU -p ampere --gres=gpu:1 -N1 -n1 -t 1:0:0 --qos=INTR
```

Full Sbatch job:
```
sbatch <slurm file>
```

Check the queue position:
```
squeue -u ab3149
```

## How to run Jupyter Notebook with HPC interactive CPU/GPU node

This guide assumes you're using a GPU partition and want to launch Jupyter inside a Conda environment with CUDA support.

### ⚙️ 1. Start an Interactive GPU Session

```
sintr -A MLMI-ab3149-SL2-GPU -p ampere --gres=gpu:1 -N1 -n1 -t 1:00:00 --qos=INTR
```

### 🐍 2. Activate Your Conda Environment
```
conda activate <your_env>
```

### 📓 3. Launch Jupyter Notebook on the GPU Node
```
jupyter notebook --no-browser --ip=0.0.0.0 --port=8081
```

### 🔁 4. Open Tunnel from Your Local Machine
In a terminal on your laptop, forward the port:
```
ssh -L 8081:gpu-q-8:8081 <your-crsid>@login-e-4.hpc.cam.ac.uk
```
- Replace gpu-q-8 with your node name (from squeue)

- Replace login-e-4 with the login node you used

### 🌐 5. Access Jupyter Notebook in Browser
```
jupyter notebook list
```

And open the link or copy paste the token in the link in step 3


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

