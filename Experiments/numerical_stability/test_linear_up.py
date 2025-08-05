import torch
from mace import data, modules, tools
import numpy as np
import torch.nn.functional
from e3nn import o3
import ase.io
import matplotlib.pyplot as plt
import warnings
import copy
import pandas as pd
warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
import types
import copy
from e3nn.o3 import Irreps
import cuequivariance   as cue
import cuequivariance_torch as cuet

try:
    import cuequivariance as cue
    cueq_available = True
    print("✓ cuEquivariance library is available")
except ImportError:
    cueq_available = False
    print("✗ cuEquivariance library is not available - cuEq will be disabled")

import torch
from e3nn import o3

# Flag: is cuEq available?
try:
    import cuequivariance        as cue
    import cuequivariance_torch  as cuet
    CUET_AVAILABLE = True
except ImportError:
    CUET_AVAILABLE = False

class LinearLayer(torch.nn.Module):
    """
    Thin wrapper over cuet.Linear or e3nn.o3.Linear with a unified constructor
    and a single-forward signature: y = W x.
    """

    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        *,
        cueq_config=None,          # your CuEquivarianceConfig
        shared_weights: bool = True,
        internal_weights: bool = True,
        use_cueq: bool = True,
        math_dtype: torch.dtype,
        device: torch.device
    ):
        super().__init__()
        if use_cueq:
            self.impl = cuet.Linear(
                cue.Irreps("O3", irreps_in),
                cue.Irreps("O3", irreps_out),
                layout="mul_ir",
                shared_weights=shared_weights,
                math_dtype=math_dtype,
                use_fallback=True,
            ).to(device)
        else:
            self.impl = o3.Linear(
                irreps_in,
                irreps_out,
                shared_weights=shared_weights,
                internal_weights=internal_weights,
            ).to(device)
            # Ensure parameters are in the correct dtype
            for param in self.impl.parameters():
                param.data = param.data.to(math_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : Tensor of shape [..., irreps_in.dim]
        returns y : Tensor of shape [..., irreps_out.dim]
        """
        output = self.impl(x)
        # Ensure output is in the same dtype as input
        if output.dtype != x.dtype:
            output = output.to(x.dtype)
        return output

def get_default_model_config(z_table):
    # setup some default parameters based on the actual dataset
    num_elements = len(z_table.zs)
    # Create atomic energies array with default values for each element
    # You can adjust these values based on your needs
    atomic_energies = np.array([-1.0] * num_elements, dtype=np.float64)  # Default energy per element
    cutoff = 6

    default_model_config = dict(
            num_elements=num_elements,  # number of chemical elements (dynamic)
            atomic_energies=atomic_energies,  # atomic energies used for normalisation
            avg_num_neighbors=180,  # avg number of neighbours of the atoms, used for internal normalisation of messages
            atomic_numbers=z_table.zs,  # atomic numbers, used to specify chemical element embeddings of the model
            r_max=cutoff,  # cutoff
            num_bessel=8,  # number of radial features
            num_polynomial_cutoff=5,  # smoothness of the radial cutoff
            max_ell=3,  # expansion order of spherical harmonic adge attributes
            num_interactions=2,  # number of layers, typically 2
            interaction_cls_first=modules.interaction_classes["RealAgnosticInteractionBlock"],
            interaction_cls=modules.interaction_classes["RealAgnosticInteractionBlock"],
            hidden_irreps=o3.Irreps("128x0e + 128x1o"),  # 8: number of embedding channels, 0e, 1o is specifying which equivariant messages to use. Here up to L_max=1
            correlation=3,  # correlation order of the messages (body order - 1)
            MLP_irreps=o3.Irreps("16x0e"),  # number of hidden dimensions of last layer readout MLP
            gate=torch.nn.functional.silu,  # nonlinearity used in last layer readout MLP
        )

    return default_model_config


def data_prep():
    single_molecule = ase.io.read('Experiments/numerical_stability/md22_double-walled_nanotube.xyz', index='0')

    # Detect elements present in the dataset
    atomic_numbers = single_molecule.numbers
    unique_atomic_numbers = sorted(set(atomic_numbers))
    print(f"Elements found in dataset: {unique_atomic_numbers}")
    print(f"Element symbols: {single_molecule.get_chemical_symbols()[:10]}...")  # Show first 10 symbols
    
    Rcut = 3.0 # cutoff radius
    # z_table = tools.AtomicNumberTable([1, 6, 8])
    z_table = tools.AtomicNumberTable(unique_atomic_numbers)
    print(f"Created z_table with {len(z_table.zs)} elements: {z_table.zs}")

    config = data.Configuration(
        atomic_numbers=single_molecule.numbers,
        positions=single_molecule.positions,
        properties={},
        property_weights={},
    )

    # we handle configurations using the AtomicData class
    batch = data.AtomicData.from_config(config, z_table=z_table, cutoff=Rcut)

    vectors, lengths = modules.utils.get_edge_vectors_and_lengths(
    positions=batch["positions"],
    edge_index=batch["edge_index"],
    shifts=batch["shifts"],
    )
    print(f'there are {batch.positions.shape[0]} nodes and {len(lengths)} edges')
    print(f'lengths is shape {lengths.shape}')
    print(f'vectors is shape {vectors.shape}')

    return batch, lengths, vectors, z_table


def benchmark_linear_up(linear_up_fn, inputs, warmup=10, runs=50):

    traced_fn = torch.jit.trace(linear_up_fn, (inputs,))
    
    with torch.inference_mode():
        # if the inputs is a tuple, unpack it
        if isinstance(inputs, tuple):
            output = traced_fn(*inputs)
        else:
            output = traced_fn(inputs)
    torch.cuda.synchronize()

    for _ in range(warmup):
        if isinstance(inputs, tuple):
            traced_fn(*inputs)
        else:
            traced_fn(inputs)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()

    time_ms = []
    for _ in range(runs):
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        traced_fn(inputs)
        end.record()
        torch.cuda.synchronize()
        time_ms.append(start.elapsed_time(end))

    mean_time_ms = (sum(time_ms)/len(time_ms))
    peak_mem_bytes = torch.cuda.max_memory_allocated()
    
    # Test output precision
    test_output = traced_fn(inputs)
    output_dtype = test_output.dtype
    
    return mean_time_ms, peak_mem_bytes, output_dtype

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    torch.manual_seed(0)
    
    batch, lengths, vectors, z_table = data_prep()
    cfg   = get_default_model_config(z_table)
    model = modules.MACE(**cfg).to(device=device, dtype=torch.float64)

    # grab the two TensorProduct modules because we want to get the dimensions of the weights
    tp0 = model.interactions[0].conv_tp # first layer
    tp1 = model.interactions[1].conv_tp # second layer

    # turn their irreps into e3nn.Irreps to get dimensions
    in_ir0, attr_ir0, out_ir0 = (
        Irreps(tp0.irreps_in1),
        Irreps(tp0.irreps_in2),
        Irreps(tp0.irreps_out),
    )
    in_ir1, attr_ir1, out_ir1 = (
        Irreps(tp1.irreps_in1),
        Irreps(tp1.irreps_in2),
        Irreps(tp1.irreps_out),
    )

    # get the l0 of linear_up - transforms from node_feats_irreps to edge_irreps
    # For simplicity, we'll use the same irreps as the tensor product layers
    linear_up0_cueq = LinearLayer(in_ir0, out_ir0, use_cueq=True, math_dtype=torch.float64, device=device)
    linear_up1_cueq = LinearLayer(in_ir1, out_ir1, use_cueq=True, math_dtype=torch.float64, device=device)
    linear_up0_e3nn = LinearLayer(in_ir0, out_ir0, use_cueq=False, math_dtype=torch.float64, device=device)
    linear_up1_e3nn = LinearLayer(in_ir1, out_ir1, use_cueq=False, math_dtype=torch.float64, device=device)

    N, E = 100, 10000
    nf0 = torch.randn(N, in_ir0.dim,    dtype=torch.float64).to(device)
    nf1 = torch.randn(N, in_ir1.dim,    dtype=torch.float64).to(device)
    tw0 = torch.randn(E, tp0.weight_numel, dtype=torch.float64).to(device)
    tw1 = torch.randn(E, tp1.weight_numel, dtype=torch.float64).to(device)
    ei  = torch.randint(0, N, (2, E), dtype=torch.int64).to(device)

    # Benchmark cuEquivariance vs e3nn linear layers
    print("\n" + "="*60)
    print("BENCHMARK: cuEquivariance vs e3nn Linear Layers")
    print("="*60)
    
    # Verify FP64 precision before benchmarking
    print(f"\nFP64 Precision Verification:")
    print(f"nf0 dtype: {nf0.dtype}")
    print(f"nf1 dtype: {nf1.dtype}")
    assert nf0.dtype == torch.float64, f"nf0 should be float64, got {nf0.dtype}"
    assert nf1.dtype == torch.float64, f"nf1 should be float64, got {nf1.dtype}"
    print("✓ Input tensors are using FP64 precision!")
    
    time_ms_cueq_linear_up0, peak_mem_cueq_linear_up0, dtype_cueq_l0 = benchmark_linear_up(linear_up0_cueq, nf0)
    time_ms_cueq_linear_up1, peak_mem_cueq_linear_up1, dtype_cueq_l1 = benchmark_linear_up(linear_up1_cueq, nf1)
    time_ms_e3nn_linear_up0, peak_mem_e3nn_linear_up0, dtype_e3nn_l0 = benchmark_linear_up(linear_up0_e3nn, nf0)
    time_ms_e3nn_linear_up1, peak_mem_e3nn_linear_up1, dtype_e3nn_l1 = benchmark_linear_up(linear_up1_e3nn, nf1)

    print(f"\nBenchmark Results (FP64):")
    print(f"cuEquivariance linear up0: {time_ms_cueq_linear_up0:.2f} ms, {peak_mem_cueq_linear_up0/1024**2:.2f} MB, output dtype: {dtype_cueq_l0}")
    print(f"cuEquivariance linear up1: {time_ms_cueq_linear_up1:.2f} ms, {peak_mem_cueq_linear_up1/1024**2:.2f} MB, output dtype: {dtype_cueq_l1}")
    print(f"e3nn linear up0: {time_ms_e3nn_linear_up0:.2f} ms, {peak_mem_e3nn_linear_up0/1024**2:.2f} MB, output dtype: {dtype_e3nn_l0}")
    print(f"e3nn linear up1: {time_ms_e3nn_linear_up1:.2f} ms, {peak_mem_e3nn_linear_up1/1024**2:.2f} MB, output dtype: {dtype_e3nn_l1}")
    
    # Test precision
    print(f"\nPrecision Verification:")
    print(f"Layer 0 cuEquivariance output dtype: {dtype_cueq_l0} {'✓' if dtype_cueq_l0 == torch.float64 else '✗'}")
    print(f"Layer 0 e3nn output dtype: {dtype_e3nn_l0} {'✓' if dtype_e3nn_l0 == torch.float64 else '✗'}")
    print(f"Layer 1 cuEquivariance output dtype: {dtype_cueq_l1} {'✓' if dtype_cueq_l1 == torch.float64 else '✗'}")
    print(f"Layer 1 e3nn output dtype: {dtype_e3nn_l1} {'✓' if dtype_e3nn_l1 == torch.float64 else '✗'}")
    
    # Assert all outputs are FP64
    assert dtype_cueq_l0 == torch.float64, f"cuEquivariance Layer 0 output should be float64, got {dtype_cueq_l0}"
    assert dtype_e3nn_l0 == torch.float64, f"e3nn Layer 0 output should be float64, got {dtype_e3nn_l0}"
    assert dtype_cueq_l1 == torch.float64, f"cuEquivariance Layer 1 output should be float64, got {dtype_cueq_l1}"
    assert dtype_e3nn_l1 == torch.float64, f"e3nn Layer 1 output should be float64, got {dtype_e3nn_l1}"
    print("✓ All output tensors are using FP64 precision!")
    
    # Calculate speedups
    speedup_l0 = time_ms_e3nn_linear_up0 / time_ms_cueq_linear_up0
    speedup_l1 = time_ms_e3nn_linear_up1 / time_ms_cueq_linear_up1
    print(f"\nSpeedup (e3nn/cuEquivariance):")
    print(f"Layer 0: {speedup_l0:.2f}x")
    print(f"Layer 1: {speedup_l1:.2f}x")

if __name__ == "__main__":
    main()