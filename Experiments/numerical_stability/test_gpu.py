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

def get_default_model_config(z_table):
    # setup some default parameters based on the actual dataset
    num_elements = len(z_table.zs)
    # Create atomic energies array with default values for each element
    # You can adjust these values based on your needs
    atomic_energies = np.array([-1.0] * num_elements, dtype=float)  # Default energy per element
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

def with_cueq_conv_fusion(conv_tp: torch.nn.Module) -> torch.nn.Module:
    """Wraps a cuet.ConvTensorProduct to use conv fusion"""
    conv_tp.original_forward = conv_tp.forward

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        tp_weights: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        sender = edge_index[0]
        receiver = edge_index[1]
        return self.original_forward(
            [tp_weights, node_feats, edge_attrs],
            {1: sender},
            {0: node_feats},
            {0: receiver},
        )

    conv_tp.forward = types.MethodType(forward, conv_tp)
    return conv_tp

def benchmark_cuda(fn, inputs, warmup=10, runs=50):
    
    traced_fn = torch.jit.trace(fn, inputs)
    with torch.inference_mode():
        traced_fn(*inputs)
    torch.cuda.synchronize()

    for _ in range(warmup):
        traced_fn(*inputs)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()

    time_ms = []
    for _ in range(runs):
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        traced_fn(*inputs)
        end.record()
        torch.cuda.synchronize()
        time_ms.append(start.elapsed_time(end))

    mean_time_ms = (sum(time_ms)/len(time_ms))
    peak_mem_bytes = torch.cuda.max_memory_allocated()
    return mean_time_ms, peak_mem_bytes

def benchmark_cpu(fn, warmup=3, runs=10):
    import time
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(runs):
        fn()
    return (time.perf_counter()-t0)*1000/runs, 0

def get_memory_and_time_usage(device: str, fn, inputs):
    return (benchmark_cuda if device.startswith("cuda") else benchmark_cpu)(fn, inputs)

# build cue descriptors + polynomial modules which allows us to use the cuequivariance library and directly conv_tp without
# having to write our own custom conv_tp config file which is annoying atm
def make_poly(in_ir, attr_ir, out_ir, math_dtype, device):
    desc = cue.descriptors.channelwise_tensor_product(
        cue.Irreps("O3", in_ir),
        cue.Irreps("O3", attr_ir),
        cue.Irreps("O3", out_ir),
    )
    return cuet.SegmentedPolynomial(
        desc.flatten_coefficient_modes()
            .squeeze_modes()
            .polynomial,
        math_dtype=math_dtype,
        output_dtype_map=[-1]
    ).to(device)

def get_dimensions(tp0, tp1, in_ir0, attr_ir0, out_ir0, in_ir1, attr_ir1, out_ir1):
    print("L0 output dim:", out_ir0.dim)
    print("L1 output dim:", out_ir1.dim)

    print("Layer 0 conv_tp dims:")
    print(" - in feat dim        ", in_ir0.dim)
    print(" - edge-attr (Y_l) dim ", attr_ir0.dim)
    print(" - radial-MLP weight dim", tp0.weight_numel)
    print(" - out feat dim       ", out_ir0.dim, "\n")

    print("Layer 1 conv_tp dims:")
    print(" - in feat dim        ", in_ir1.dim)
    print(" - edge-attr (Y_l) dim ", attr_ir1.dim)
    print(" - radial-MLP weight dim", tp1.weight_numel)
    print(" - out feat dim       ", out_ir1.dim, "\n")


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

    # sanity check printing
    get_dimensions(tp0, tp1, in_ir0, attr_ir0, out_ir0, in_ir1, attr_ir1, out_ir1)


    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    torch.manual_seed(0)

    # we are building for the first layer and second layer
    poly0 = make_poly(in_ir0, attr_ir0, out_ir0, math_dtype=torch.float64, device=device)
    poly1 = make_poly(in_ir1, attr_ir1, out_ir1, math_dtype=torch.float64, device=device)

    # fuse them with the helper function which is given by MACE in wrapper_ops.py
    fused0_ref = with_cueq_conv_fusion(poly0)
    fused1_ref = with_cueq_conv_fusion(poly1)

    # cast layers once and store dtype
    layers0, layers1 = {}, {}
    for name, dtype in [
        ("FP64", torch.float64),
        ("FP32", torch.float32),
        ("TF32", torch.float32),
        ("FP16", torch.float16),
        ("BF16", torch.bfloat16),
    ]:
        if name in ("FP64","FP32","FP16","BF16") and device.startswith("cuda"):
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32   = False
        if name == "TF32" and device.startswith("cuda"):
            dtype = torch.float32
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32   = True

        m0 = copy.deepcopy(fused0_ref).to(dtype)
        m1 = copy.deepcopy(fused1_ref).to(dtype)
        layers0[name] = (m0, dtype)
        layers1[name] = (m1, dtype)
    

    # simulate a mini-batch with the same number of nodes and edges as the real run
    N, E = 100, 10000
    nf0 = torch.randn(N, in_ir0.dim,    dtype=torch.float64)
    nf1 = torch.randn(N, in_ir1.dim,    dtype=torch.float64)
    ea  = torch.randn(E, attr_ir0.dim,  dtype=torch.float64)
    tw0 = torch.randn(E, tp0.weight_numel, dtype=torch.float64)
    tw1 = torch.randn(E, tp1.weight_numel, dtype=torch.float64)
    ei  = torch.randint(0, N, (2, E), dtype=torch.int64)

    inputs0, inputs1 = {}, {}
    for name, (layer0, dt0) in layers0.items():
        inputs0[name] = (
            nf0.to(device=device, dtype=dt0),
            ea .to(device=device, dtype=dt0),
            tw0.to(device=device, dtype=dt0),
            ei .to(device)
        )
        layer1, dt1 = layers1[name]
        inputs1[name] = (
            nf1.to(device=device, dtype=dt1),
            ea .to(device=device, dtype=dt1),
            tw1.to(device=device, dtype=dt1),
            ei .to(device)
        )

    # benchmark forward only
    results = []
    for name in layers0:
        layer0, _ = layers0[name]
        layer1, _ = layers1[name]

        t0, m0 = get_memory_and_time_usage(device, layer0, inputs0[name])
        t1, m1 = get_memory_and_time_usage(device, layer1, inputs1[name])
        results.append((name, t0, m0, t1, m1))

    # Convert results to DataFrame
    df_results = pd.DataFrame(results, columns=['dtype', 'L0_time', 'L0_mem', 'L1_time', 'L1_mem'])
    df_results['L0_time'] = df_results['L0_time']
    df_results['L1_time'] = df_results['L1_time']
    df_results['L0_mem'] = (df_results['L0_mem'] / 1e6)
    df_results['L1_mem'] = (df_results['L1_mem'] / 1e6)
    df_results['L1/L0_time'] = (df_results['L1_time'] / df_results['L0_time'])
    df_results['L1/L0_mem'] = (df_results['L1_mem'] / df_results['L0_mem'])

    # Display the DataFrame
    print("Benchmark Results when cuet.SegmentedPolynomial is used in math_dtype: fp64. All results are in ms and MB")
    
    # Format with scientific notation
    # pd.set_option('display.float_format', '{:.5e}'.format)
    print(df_results.to_string(index=False))




if __name__ == "__main__":
    main()