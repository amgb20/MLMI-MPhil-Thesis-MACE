# MLMI-MPhil-Thesis-MACE
This is a University of Cambridge MLMI MPhil Thesis on **Speeding up MACE**

## TODO List

### Open questions

- [ ] It seems that they are hundreds of different Tensor config? which one are we using [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html)

### Speeding up

Some GPT suggestions for starting experiments on speeding up inference: [Speeding UP MACE](Notes/General%20Concepts/Speeding_up.md)

### Coding

Coding notes can be accessed in [Tutorial Notes](Notes/Tutorials/T03-MACE-Theory.md)
- [ ] [MACE GitHub Repository](https://github.com/ACEsuit/mace) — try to look through the code and train a MACE-small on a subset of a dataset.
- [ ] Running the MACE tutorials #3

### Reading
- [ ] [Computing hydration free energies of small molecules with first principles accuracy](https://arxiv.org/abs/2405.18171)
- [ ] [Stochastic Interpolants: A Unifying Framework for Flows and Diffusions](https://arxiv.org/abs/2303.08797)

### Videos
- [ ] [Machine learning potentials always extrapolate, it does not matter](https://www.youtube.com/watch?v=WgFAZygGV8w)
- [ ] [Atomic Cluster Expansion: A framework for fast and accurate ML force fields](https://www.youtube.com/watch?v=ja-3UrdSRi4)
- [ ] [Orb-v3: atomistic simulation at scale | Tim Duignan & Sander Vandenhaute](https://www.youtube.com/watch?v=pRbvRl0_FyE)
- [ ] [Day 3 - Harnessing Geometric ML for Molecular Design | Michael Bronstein](https://www.youtube.com/watch?v=zsIyzLtwAHY)
- [ ] [Every video on The Computational Chemist YouTube channel](https://www.youtube.com/@thecomputationalchemist)
- [ ] [Open Catalyst Project videos](https://www.youtube.com/@opencatalystproject3509/videos?app=desktop)
- [x] [MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields](https://www.youtube.com/watch?v=I9Y2le9e74A&ab_channel=ValenceLabs)

### Completed Task

- Run the mace tutorials and did a subproject by evaluating the computation efficiency of each step of the mace schematic
- Read MACE paper and notes
- Main thing to look at is the equivarient MPNNs tensor product (equation 8 in paper)