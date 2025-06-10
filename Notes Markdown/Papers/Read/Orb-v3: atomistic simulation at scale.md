# Orb-v3: atomistic simulation at scale

## How to use Orbv3 for speeding up MACE?

1. Orb v3 intentionally caps neighbor counts per atom (e.g., max 20 neighbors), leading to much sparser message-passing graphs. This drastically reduces computation without sacrificing accuracy on larger system
> Techniques for adaptive neighbor selection and message pruning during runtime.

2.  smoothed graph attention, avoiding costly tensor products or spherical harmonics
> Replacing or hybridizing parts of MACE’s equivariant layers with efficient attention mechanisms.

> Investigating if selective non-equivariant layers can suffice in less sensitive parts of the model.

3.  We introduce equigrad: 
> A simple and differentiable regularization strategy which strengthens the symmetry-induced inductive biases without sacrificing performance. Orb v3 uses the Equigrad regularizer, softly enforcing rotation symmetry while enabling simpler and faster architectures. Adapting a weaker regularization (like equigrad) into MACE to relax strict equivariance requirements.

Experimenting with trade-offs between equivariance and speed through losses that penalize rotation-gradient mismatches.

4. Various architectural improvements.
‍
> These improve precision towards fine-grained computational workflows; notable examples include optional conservatism (i.e. computation of the interatomic forces via gradients of the energy) and smoother radial embeddings


5. Strongly reduced computational cost using sparse atom graphs: 
> Our sparse direct model, orb-v3-direct-20, is 10x faster and 5x more memory efficient than MACE-MPA-0, and 40x faster and 40x more memory efficient than Sevennet. We believe this scalability is essential to perform accurate atomic simulations at scale.

6. Sparse Training and Weight Pruning
While MACE’s architecture is dense, recent research shows that GNN weights and message-passing flows can be sparsified, shaving off FLOPs during both training and inference.

>Sparse training methods (e.g., RigL, SET) applicable to GNN layers in MACE.
>Incremental pruning of attention heads or tensor components during training.

7. Model distillation:
> distill complex, equivariant teacher models into smaller, faster student networks, preserving accuracy but accelerating inference 10–20×