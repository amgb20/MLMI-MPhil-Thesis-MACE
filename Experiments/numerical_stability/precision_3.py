import torch
from mace import data, modules, tools
import numpy as np
import torch.nn.functional
from e3nn import o3
import ase.io
from mace.modules.wrapper_ops import CuEquivarianceConfig, OEQConfig
import matplotlib.pyplot as plt
import warnings
import copy
import pandas as pd
warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")

import ns_plots


try:
    import cuequivariance as cue
    cueq_available = True
    print("✓ cuEquivariance library is available")
except ImportError:
    cueq_available = False
    print("✗ cuEquivariance library is not available - cuEq will be disabled")

# TODO: warm up run because the first few CUDA kernels and cuDNN calls often include JIT compilation or algorithm selection overhead --> good practice to do dry runs per precision before starting timed measurements
# TODO: warm-up and repetition where we need to run each block multiple times to get a more accurate time measurements adn report mean and std to smooth out GPU jitter
# TODO: Error disributions: plot histograms (or boxplots) of per-element fwd/bwd errors across all atoms to catch any outliers
# TODO: Pareto frontier: A scatter “error vs. time” plot where each precision is a dot can more cleanly show the tradeoff curve.
# TODO: Graph Size scaling: test on several molecule/graph sizes (e.g. 50, 500, 5000 atoms) and visualise how speedup and error scale with system size.


def get_default_model_config(z_table):
    # setup some default parameters based on the actual dataset
    num_elements = len(z_table.zs)
    # Create atomic energies array with default values for each element
    # You can adjust these values based on your needs
    atomic_energies = np.array([-1.0] * num_elements, dtype=float)  # Default energy per element
    cutoff = 3

    # Add cuEq configuration as a CuEquivarianceConfig instance
    # Option 1: Disable cuEq to enable conv_fusion
    cueq_config = CuEquivarianceConfig(
        enabled=False,  # Changed from True to False to allow conv_fusion
        layout="ir_mul",
        group="O3_e3nn",
        optimize_all=True,
    )
    
    # Option 2: Remove cuEq config entirely (uncomment below and comment out above)
    # cueq_config = None

    oeq_config = OEQConfig(
        enabled=True,
        optimize_all=True,
        conv_fusion="atomic",  # or another supported value
    )

    default_model_config = dict(
            num_elements=num_elements,  # number of chemical elements (dynamic)
            atomic_energies=atomic_energies,  # atomic energies used for normalisation
            avg_num_neighbors=8,  # avg number of neighbours of the atoms, used for internal normalisation of messages
            atomic_numbers=z_table.zs,  # atomic numbers, used to specify chemical element embeddings of the model
            r_max=cutoff,  # cutoff
            num_bessel=8,  # number of radial features
            num_polynomial_cutoff=6,  # smoothness of the radial cutoff
            max_ell=2,  # expansion order of spherical harmonic adge attributes
            num_interactions=2,  # number of layers, typically 2
            interaction_cls_first=modules.interaction_classes["RealAgnosticInteractionBlock"],
            interaction_cls=modules.interaction_classes["RealAgnosticInteractionBlock"],
            hidden_irreps=o3.Irreps("8x0e + 8x1o"),  # 8: number of embedding channels, 0e, 1o is specifying which equivariant messages to use. Here up to L_max=1
            correlation=3,  # correlation order of the messages (body order - 1)
            MLP_irreps=o3.Irreps("16x0e"),  # number of hidden dimensions of last layer readout MLP
            gate=torch.nn.functional.silu,  # nonlinearity used in last layer readout MLP
            cueq_config=cueq_config,  # Enable cuEq acceleration
            oeq_config=oeq_config,  # Enable OEQ acceleration
        )

    return default_model_config


def data_prep():
    single_molecule = ase.io.read('Experiments/Official MACE notebook/data/md22_double-walled_nanotube.xyz', index='0')

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


def get_embedding_blocks(default_model_config, batch, lengths, vectors, z_table, device='cpu'):
    # set up a mace model to get all of the blocks in one place:
    model = modules.MACE(**default_model_config)
    model = model.to(device)  # Move model to the specified device
    
    # Get the model's dtype to ensure consistency
    model_dtype = next(model.parameters()).dtype
    print(f"Model dtype: {model_dtype}")

    # Move all input data to the same device and dtype as the model
    batch = batch.to(device).to(model_dtype)
    lengths = lengths.to(device).to(model_dtype)
    vectors = vectors.to(device).to(model_dtype)
    
    # Ensure all tensors in the batch have the correct dtype
    # Handle batch as a PyTorch Geometric Data object
    if hasattr(batch, '__dict__'):
        for key, value in batch.__dict__.items():
            if hasattr(value, 'dtype') and hasattr(value, 'to'):
                if value.dtype != model_dtype:
                    print(f"Converting {key} from {value.dtype} to {model_dtype}")
                    setattr(batch, key, value.to(model_dtype))
    
    # Debug: Print all tensor dtypes
    print("Tensor dtypes after conversion:")
    if hasattr(batch, '__dict__'):
        for key, value in batch.__dict__.items():
            if hasattr(value, 'dtype'):
                print(f"  {key}: {value.dtype}")
    print(f"  lengths: {lengths.dtype}")
    print(f"  vectors: {vectors.dtype}")

    initial_node_features = model.node_embedding(batch.node_attrs)

    edge_features,_ = model.radial_embedding(lengths, batch["node_attrs"], batch["edge_index"], z_table)
    edge_attributes = model.spherical_harmonics(vectors)

    # print('initial_node_features is (num_atoms, num_channels):', initial_node_features.shape)
    # print('edge_features is (num_edge, num_bessel_func):', edge_features.shape)
    # print('edge_attributes is (num_edge, dimension of spherical harmonics):', edge_attributes.shape)
    # print(
    #     '\nInitial node features. Note that they are the same for each chemical element\n',
    #     initial_node_features
    # )

    return model, initial_node_features, edge_features, edge_attributes

def get_interaction_block(model, batch, lengths, vectors, initial_node_features, edge_features, edge_attributes):
    Interaction = model.interactions[0]

    intermediate_node_features, sc = Interaction(
        node_feats=initial_node_features,
        node_attrs=batch.node_attrs,
        edge_feats=edge_features,
        edge_attrs=edge_attributes,
        edge_index=batch.edge_index,
    )

    print("the output of the interaction is (num_atoms, channels, dim. spherical harmonics):", intermediate_node_features.shape)
    
    return intermediate_node_features, sc

def run_precision_comparison(device='cpu'):
    # Prepare data first to get z_table, then model config
    batch, lengths, vectors, z_table = data_prep()
    # default_model_config = get_default_model_config()
    default_model_config = get_default_model_config(z_table)
    
    # 1) Build one "master" FP64 model and capture its state
    torch.set_default_dtype(torch.float64)
    master = modules.MACE(**default_model_config).to(device=device, dtype=torch.float64)
    master_state = master.state_dict()
 
    precisions = [torch.float64, torch.float32]
    
    # Add TF32 and BF16 support for CUDA devices
    if torch.cuda.is_available():
        # Check for BF16 support
        try:
            torch.zeros(1, device='cuda', dtype=torch.bfloat16)
            precisions.append(torch.bfloat16)
            print("✓ BF16 supported on this device")
        except Exception:
            print("✗ BF16 not supported on this device, skipping BF16.")
        
        # Check for FP16 support
        try:
            torch.zeros(1, device='cuda', dtype=torch.float16)
            precisions.append(torch.float16)
            print("✓ FP16 supported on this device")
        except Exception:
            print("✗ FP16 not supported on this device, skipping FP16.")
    else:
        print("Mixed precision formats not supported on CPU, skipping BF16/FP16.")

    results = {0: {}, 1: {}}  # block_idx -> dtype -> {'output': ..., 'grad': ...}

    metrics = []

    for dtype in precisions:
        print(f"\n=== Testing {dtype} ===")
        
        # Special handling for TF32: enable/disable TF32 backends
        if dtype == torch.float32 and device.startswith('cuda'):
            # Test FP32 with TF32 enabled
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("  TF32 enabled for matmul and cudnn")
        elif dtype == torch.float32 and device.startswith('cuda'):
            # Test FP32 with TF32 disabled (for comparison)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            print("  TF32 disabled for matmul and cudnn")
        
        # a) re-instantiate & load the very same FP64 weights
        m = modules.MACE(**default_model_config)
        m.load_state_dict(master_state)
        # b) now cast every registered param & buffer
        m = m.to(device=device, dtype=dtype)

        # 2) Recompute all embeddings at this precision
        batch_d   = batch.to(device=device, dtype=dtype)
        lengths_d = lengths.to(device=device, dtype=dtype)
        vectors_d = vectors.to(device=device, dtype=dtype)

        x0 = m.node_embedding(batch_d.node_attrs)
        x0 = x0.requires_grad_()
        x0.retain_grad()
        e_feats, _ = m.radial_embedding(lengths_d,
                                        batch_d.node_attrs,
                                        batch_d.edge_index,
                                        z_table)
        e_attrs = m.spherical_harmonics(vectors_d)

        # 3) **Cast node_attrs** to the same float dtype as everything else!
        node_attrs_d = batch_d.node_attrs.to(dtype)

        inputs0 = {
            'node_feats': x0,
            'node_attrs': node_attrs_d,
            'edge_feats': e_feats,
            'edge_attrs': e_attrs,
            'edge_index': batch_d.edge_index.to(torch.long),
        }

        # Pull out the exact same interaction blocks & products
        block0   = m.interactions[0]
        product0 = m.products[0]
        block1   = m.interactions[1]

        print("=== InteractionBlock[0] ===")
        print("Has conv_fusion? ", hasattr(block0, "conv_fusion"))
        print("Has conv_tp? ", hasattr(block0, "conv_tp"))
        print("  conv_fusion =", getattr(block0, "conv_fusion", None))
        print("  conv_tp =", getattr(block0, "conv_tp", None))

        # Forward block 0
        print("skip_tp.weight.dtype:", block0.skip_tp.weight.dtype, "node_feats.dtype:", inputs0["node_feats"].dtype)
        output0, _ = block0(**inputs0)
        # block memeroy for block 0
        if device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats(device)
            start0 = torch.cuda.Event(enable_timing=True)
            end0 = torch.cuda.Event(enable_timing=True)
            start0.record()
        output0, _ = block0(**inputs0)
        if device.startswith('cuda'):
            end0.record()
            torch.cuda.synchronize()
            t0 = start0.elapsed_time(end0)
            m0 = torch.cuda.max_memory_allocated(device)
        else:
            import time
            t0 = time.perf_counter()
            output0, _ = block0(**inputs0)
            t0 = (time.perf_counter() - t0) * 1000
            m0 = 0
        output0_prod = product0(node_feats=output0, sc=None, node_attrs=inputs0['node_attrs'])
        loss0 = (output0 ** 2).sum()
        loss0.backward()
        grad0 = inputs0['node_feats'].grad
        if grad0 is None:
            raise RuntimeError("Gradient for inputs0['node_feats'] is None. Ensure .retain_grad() is called on the tensor.")
        
        # Create precision label for metrics
        precision_label = str(dtype)
        if dtype == torch.float32 and device.startswith('cuda'):
            if torch.backends.cuda.matmul.allow_tf32:
                precision_label = "tf32"
            else:
                precision_label = "fp32_no_tf32"
        
        results[0][dtype] = {
            'output': output0.detach().cpu().double().numpy(),
            'grad': grad0.detach().cpu().double().numpy(),
        }
        metrics.append({
            'block': 0,
            'precision': precision_label,
            'time_ms': t0,
            'memory_bytes': m0,
        })
        
        # Prepare inputs for block 1
        inputs1 = dict(inputs0)
        node_feats1 = output0_prod.clone().detach().to(device=device, dtype=dtype).requires_grad_(True)
        node_feats1.retain_grad()  # Ensure gradient is retained for non-leaf tensor
        inputs1['node_feats'] = node_feats1
        # Forward block 1
        output1, _ = block1(**inputs1)
        # block memeroy for block 1
        if device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats(device)
            start1 = torch.cuda.Event(enable_timing=True)
            end1 = torch.cuda.Event(enable_timing=True)
            start1.record()
        output1, _ = block1(**inputs1)
        if device.startswith('cuda'):
            end1.record()
            torch.cuda.synchronize()
            t1 = start1.elapsed_time(end1)
            m1 = torch.cuda.max_memory_allocated(device)
        else:
            import time
            t1 = time.perf_counter()
            output1, _ = block1(**inputs1)
            t1 = (time.perf_counter() - t1) * 1000
            m1 = 0
        loss1 = (output1 ** 2).sum()
        loss1.backward()
        grad1 = inputs1['node_feats'].grad
        if grad1 is None:
            raise RuntimeError("Gradient for inputs1['node_feats'] is None. Ensure .retain_grad() is called on the tensor.")
        results[1][dtype] = {
            'output': output1.detach().cpu().double().numpy(),
            'grad': grad1.detach().cpu().double().numpy(),
        }
        metrics.append({
            'block': 1,
            'precision': precision_label,
            'time_ms': t1,
            'memory_bytes': m1,
        })

    # Now compare for both blocks and output as DataFrame
    all_dfs = {}
    for block_idx in [0, 1]:
        print(f"\n=== Interaction Block {block_idx} ===")
        ref = results[block_idx][torch.float64]
        rows = []
        
        # Get all precisions except FP64 for comparison
        comparison_precisions = [p for p in precisions if p != torch.float64]
        
        for dtype in comparison_precisions:
            # Forward error
            err_fwd = np.abs(ref['output'] - results[block_idx][dtype]['output'])
            rel_fwd = np.abs(err_fwd / (np.abs(ref['output']) + 1e-12))
            # Backward error
            err_bwd = np.abs(ref['grad'] - results[block_idx][dtype]['grad'])
            rel_bwd = np.abs(err_bwd / (np.abs(ref['grad']) + 1e-12))
            
            # Create precision label for comparison
            precision_label = str(dtype)
            if dtype == torch.float32 and device.startswith('cuda'):
                if torch.backends.cuda.matmul.allow_tf32:
                    precision_label = "tf32"
                else:
                    precision_label = "fp32_no_tf32"
            
            rows.append({
                'block': block_idx,
                'precision': precision_label,
                'fwd_max_abs': err_fwd.max(),
                'fwd_max_rel': rel_fwd.max(),
                'bwd_max_abs': err_bwd.max(),
                'bwd_max_rel': rel_bwd.max(),
            })

            df_sub = pd.DataFrame(rows)
            print(f"\nComparison to FP64 for block {block_idx} (FP64 vs. {precision_label}):")
            print(df_sub)

        df = pd.DataFrame(rows, columns=['block', 'precision', 'fwd_max_abs', 'fwd_max_rel', 'bwd_max_abs', 'bwd_max_rel'])
        print(f"\nComparison to FP64 for block {block_idx}:")
        print(df)
        all_dfs[f'block_{block_idx}'] = df

    df_perf = pd.DataFrame(metrics)
    df_perf.to_excel(f'Experiments/precision_3/xlsx/precision_metrics.xlsx', index=False)

    ns_plots.plot_precision_comparison(df_perf, 'Experiments/precision_3/figs/precision_comparison.png')

    with pd.ExcelWriter('Experiments/precision_3/xlsx/precision_3.xlsx') as writer:
        for sheet, df in all_dfs.items():
            df.to_excel(writer, sheet_name=sheet, index=False)
    print(f"Excel file saved as Experiments/precision_3/xlsx/precision_3.xlsx")

def test_tf32_vs_fp32(device='cuda'):
    """Explicitly test TF32 vs FP32 performance and accuracy"""
    if not device.startswith('cuda'):
        print("TF32 testing requires CUDA device")
        return
    
    print("\n=== TF32 vs FP32 Comparison ===")
    
    # Prepare data first to get z_table, then model config
    batch, lengths, vectors, z_table = data_prep()
    default_model_config = get_default_model_config(z_table)
    
    # Build master FP64 model
    torch.set_default_dtype(torch.float64)
    master = modules.MACE(**default_model_config).to(device=device, dtype=torch.float64)
    master_state = master.state_dict()
    
    tf32_results = {}
    fp32_results = {}
    
    # Test configurations
    configs = [
        ("tf32", True, True),
        ("fp32_no_tf32", False, False)
    ]
    
    metrics = []
    
    for config_name, allow_matmul_tf32, allow_cudnn_tf32 in configs:
        print(f"\n--- Testing {config_name} ---")
        
        # Set TF32 flags
        torch.backends.cuda.matmul.allow_tf32 = allow_matmul_tf32
        torch.backends.cudnn.allow_tf32 = allow_cudnn_tf32
        
        # Create model
        m = modules.MACE(**default_model_config)
        m.load_state_dict(master_state)
        m = m.to(device=device, dtype=torch.float32)
        
        # Prepare data
        batch_d = batch.to(device=device, dtype=torch.float32)
        lengths_d = lengths.to(device=device, dtype=torch.float32)
        vectors_d = vectors.to(device=device, dtype=torch.float32)
        
        x0 = m.node_embedding(batch_d.node_attrs)
        x0 = x0.requires_grad_()
        x0.retain_grad()
        e_feats, _ = m.radial_embedding(lengths_d, batch_d.node_attrs, batch_d.edge_index, z_table)
        e_attrs = m.spherical_harmonics(vectors_d)
        node_attrs_d = batch_d.node_attrs.to(torch.float32)
        
        inputs = {
            'node_feats': x0,
            'node_attrs': node_attrs_d,
            'edge_feats': e_feats,
            'edge_attrs': e_attrs,
            'edge_index': batch_d.edge_index.to(torch.long),
        }
        
        # Test both blocks
        for block_idx in [0, 1]:
            block = m.interactions[block_idx]
            
            # Time the forward pass
            torch.cuda.reset_peak_memory_stats(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            
            output, _ = block(**inputs)
            
            end.record()
            torch.cuda.synchronize()
            time_ms = start.elapsed_time(end)
            memory_bytes = torch.cuda.max_memory_allocated(device)
            
            # Compute gradients
            loss = (output ** 2).sum()
            loss.backward()
            grad = inputs['node_feats'].grad
            
            # Store results
            if config_name == "tf32":
                tf32_results[block_idx] = {
                    'output': output.detach().cpu().double().numpy(),
                    'grad': grad.detach().cpu().double().numpy(),
                    'time_ms': time_ms,
                    'memory_bytes': memory_bytes
                }
            else:
                fp32_results[block_idx] = {
                    'output': output.detach().cpu().double().numpy(),
                    'grad': grad.detach().cpu().double().numpy(),
                    'time_ms': time_ms,
                    'memory_bytes': memory_bytes
                }
            
            metrics.append({
                'block': block_idx,
                'precision': config_name,
                'time_ms': time_ms,
                'memory_bytes': memory_bytes,
            })
            
            print(f"  Block {block_idx}: {time_ms:.2f}ms, {memory_bytes/1024**2:.1f}MB")
            
            # Prepare for next block
            if block_idx == 0:
                product = m.products[0]
                output_prod = product(node_feats=output, sc=None, node_attrs=inputs['node_attrs'])
                node_feats_next = output_prod.clone().detach().to(device=device, dtype=torch.float32).requires_grad_(True)
                node_feats_next.retain_grad()
                inputs['node_feats'] = node_feats_next
    
    # Create DataFrame for plotting
    df_perf = pd.DataFrame(metrics)
    df_perf.to_excel('Experiments/precision_3/xlsx/tf32_vs_fp32_metrics.xlsx', index=False)
    
    # Plot the comparison
    ns_plots.plot_precision_comparison(df_perf, 'Experiments/precision_3/figs/tf32_vs_fp32_comparison.png')
    
    # Compare results
    print("\n=== TF32 vs FP32 Comparison Results ===")
    for block_idx in [0, 1]:
        print(f"\nBlock {block_idx}:")
        
        # Forward error
        err_fwd = np.abs(tf32_results[block_idx]['output'] - fp32_results[block_idx]['output'])
        rel_fwd = np.abs(err_fwd / (np.abs(tf32_results[block_idx]['output']) + 1e-12))
        
        # Backward error
        err_bwd = np.abs(tf32_results[block_idx]['grad'] - fp32_results[block_idx]['grad'])
        rel_bwd = np.abs(err_bwd / (np.abs(tf32_results[block_idx]['grad']) + 1e-12))
        
        # Performance
        time_speedup = fp32_results[block_idx]['time_ms'] / tf32_results[block_idx]['time_ms']
        memory_ratio = tf32_results[block_idx]['memory_bytes'] / fp32_results[block_idx]['memory_bytes']
        
        print(f"  Forward max abs error: {err_fwd.max():.2e}")
        print(f"  Forward max rel error: {rel_fwd.max():.2e}")
        print(f"  Backward max abs error: {err_bwd.max():.2e}")
        print(f"  Backward max rel error: {rel_bwd.max():.2e}")
        print(f"  TF32 speedup: {time_speedup:.2f}x")
        print(f"  Memory ratio (TF32/FP32): {memory_ratio:.2f}")
    
    return tf32_results, fp32_results

def comprehensive_precision_test(device='cpu'):
    """Comprehensive precision testing including TF32, FP32, FP64"""
    print("\n=== Comprehensive Precision Testing ===")
    
    # Prepare data first to get z_table, then model config
    batch, lengths, vectors, z_table = data_prep()
    default_model_config = get_default_model_config(z_table)
    
    # Build master FP64 model
    torch.set_default_dtype(torch.float64)
    master = modules.MACE(**default_model_config).to(device=device, dtype=torch.float64)
    master_state = master.state_dict()
    
    # Define all precision configurations to test
    precision_configs = [
        ("fp64", torch.float64, False, False),
        ("fp32", torch.float32, False, False),
    ]
    
    # Add CUDA-specific precisions
    """
    TF32 vs FP32 on NVIDIA GPUs:
    TF32 is a precision mode available on NVIDIA Ampere and newer GPUs. It allows matrix multiplications 
    (like those in deep learning) to run much faster, with a small loss in precision compared to full FP32.
    FP32 (no TF32) disables TF32, forcing the use of full 32-bit floating point math, which is slower but more precise.
    """
    if device.startswith('cuda'):
        precision_configs.extend([
            ("tf32", torch.float32, True, True),
        ])
        
        # Check for BF16 support
        try:
            torch.zeros(1, device='cuda', dtype=torch.bfloat16)
            precision_configs.append(("bf16", torch.bfloat16, False, False))
            print("✓ BF16 will be tested")
        except Exception:
            print("✗ BF16 not supported, skipping")
        
        # Check for FP16 support
        try:
            torch.zeros(1, device='cuda', dtype=torch.float16)
            precision_configs.append(("fp16", torch.float16, False, False))
            print("✓ FP16 will be tested")
        except Exception:
            print("✗ FP16 not supported, skipping")
    
    results = {0: {}, 1: {}}
    metrics = []
    
    for config_name, dtype, allow_matmul_tf32, allow_cudnn_tf32 in precision_configs:
        print(f"\n--- Testing {config_name} ---")
        
        # Set TF32 flags if applicable
        """
        This controls whether pytorch is allowed to use tf32 precision for matrix multiplications and cuDNN operations (convolutions)
        """
        if device.startswith('cuda'):
            torch.backends.cuda.matmul.allow_tf32 = allow_matmul_tf32
            torch.backends.cudnn.allow_tf32 = allow_cudnn_tf32
            if allow_matmul_tf32 or allow_cudnn_tf32:
                print(f"  TF32 enabled: matmul={allow_matmul_tf32}, cudnn={allow_cudnn_tf32}")
        
        # Create model
        m = modules.MACE(**default_model_config)
        m.load_state_dict(master_state)
        m = m.to(device=device, dtype=dtype)
        
        # Prepare data
        batch_d = batch.to(device=device, dtype=dtype)
        lengths_d = lengths.to(device=device, dtype=dtype)
        vectors_d = vectors.to(device=device, dtype=dtype)
        
        x0 = m.node_embedding(batch_d.node_attrs)
        x0 = x0.requires_grad_()
        x0.retain_grad()
        e_feats, _ = m.radial_embedding(lengths_d, batch_d.node_attrs, batch_d.edge_index, z_table)
        e_attrs = m.spherical_harmonics(vectors_d)
        node_attrs_d = batch_d.node_attrs.to(dtype)
        
        inputs = {
            'node_feats': x0,
            'node_attrs': node_attrs_d,
            'edge_feats': e_feats,
            'edge_attrs': e_attrs,
            'edge_index': batch_d.edge_index.to(torch.long),
        }
        
        # Test both blocks
        for block_idx in [0, 1]:
            block = m.interactions[block_idx]
            
            # Time the forward pass
            if device.startswith('cuda'):
                torch.cuda.reset_peak_memory_stats(device)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
            
            output, _ = block(**inputs)
            
            if device.startswith('cuda'):
                end.record()
                torch.cuda.synchronize()
                time_ms = start.elapsed_time(end)
                memory_bytes = torch.cuda.max_memory_allocated(device)
            else:
                import time
                start_time = time.perf_counter()
                output, _ = block(**inputs)
                time_ms = (time.perf_counter() - start_time) * 1000
                memory_bytes = 0
            
            # Compute gradients
            loss = (output ** 2).sum()
            loss.backward()
            grad = inputs['node_feats'].grad
            
            # Store results
            results[block_idx][config_name] = {
                'output': output.detach().cpu().double().numpy(),
                'grad': grad.detach().cpu().double().numpy(),
                'time_ms': time_ms,
                'memory_bytes': memory_bytes
            }
            
            metrics.append({
                'block': block_idx,
                'precision': config_name,
                'time_ms': time_ms,
                'memory_bytes': memory_bytes,
            })
            
            print(f"  Block {block_idx}: {time_ms:.2f}ms, {memory_bytes/1024**2:.1f}MB")
            
            # Prepare for next block
            if block_idx == 0:
                product = m.products[0]
                output_prod = product(node_feats=output, sc=None, node_attrs=inputs['node_attrs'])
                node_feats_next = output_prod.clone().detach().to(device=device, dtype=dtype).requires_grad_(True)
                node_feats_next.retain_grad()
                inputs['node_feats'] = node_feats_next
    
    # Comprehensive comparison
    print("\n=== Comprehensive Precision Comparison ===")
    ref_config = "fp64"
    
    comparison_data = []
    for block_idx in [0, 1]:
        print(f"\nBlock {block_idx} (reference: {ref_config}):")
        ref_output = results[block_idx][ref_config]['output']
        ref_grad = results[block_idx][ref_config]['grad']
        
        for config_name in results[block_idx].keys():
            if config_name == ref_config:
                continue
                
            # Forward error
            err_fwd = np.abs(ref_output - results[block_idx][config_name]['output'])
            rel_fwd = np.abs(err_fwd / (np.abs(ref_output) + 1e-12))
            
            # Backward error
            err_bwd = np.abs(ref_grad - results[block_idx][config_name]['grad'])
            rel_bwd = np.abs(err_bwd / (np.abs(ref_grad) + 1e-12))
            
            # Performance relative to FP32
            fp32_time = results[block_idx]['fp32']['time_ms']
            current_time = results[block_idx][config_name]['time_ms']
            speedup = fp32_time / current_time if current_time > 0 else 0
            
            comparison_data.append({
                'block': block_idx,
                'precision': config_name,
                'fwd_max_abs': err_fwd.max(),
                'fwd_max_rel': rel_fwd.max(),
                'bwd_max_abs': err_bwd.max(),
                'bwd_max_rel': rel_bwd.max(),
                'speedup_vs_fp32': speedup,
                'time_ms': current_time,
                'memory_mb': results[block_idx][config_name]['memory_bytes'] / 1024**2,
            })
            
            print(f"  {config_name}:")
            print(f"    Forward error: {err_fwd.max():.2e} (abs), {rel_fwd.max():.2e} (rel)")
            print(f"    Backward error: {err_bwd.max():.2e} (abs), {rel_bwd.max():.2e} (rel)")
            print(f"    Speedup vs FP32: {speedup:.2f}x")
            print(f"    Time: {current_time:.2f}ms")
            print(f"    Memory: {results[block_idx][config_name]['memory_bytes']/1024**2:.1f}MB")
    
    # Save comprehensive results
    df_comprehensive = pd.DataFrame(comparison_data)
    df_comprehensive.to_excel('Experiments/numerical_stability/xlsx/comprehensive_precision_test.xlsx', index=False)
    print(f"\nComprehensive results saved to Experiments/numerical_stability/xlsx/comprehensive_precision_test.xlsx")
    
    # Create DataFrame for plotting performance metrics
    df_perf = pd.DataFrame(metrics)
    df_perf.to_excel('Experiments/numerical_stability/xlsx/comprehensive_performance_metrics.xlsx', index=False)
    
    # Plot the performance comparison
    ns_plots.plot_precision_comparison(df_perf, 'Experiments/numerical_stability/figs/comprehensive_performance_comparison.png')
    # Call error distribution plotting
    ns_plots.plot_error_distributions_2(results, batch, z_table, ref_precision="fp64", save_path='Experiments/numerical_stability/figs/comprehensive_error_distributions.png')
    return results, df_comprehensive, batch, z_table

def main():
    # check if cuda is available and set the device to cuda
    if torch.cuda.is_available():
        device = 'cuda'
    else:   
        device = 'cpu'

    print(f"Device: {device}")

    # empty pytorch cache
    if device.startswith('cuda'):
        torch.cuda.empty_cache()

    # check the default dtype
    print(f"Default dtype: {torch.get_default_dtype()}")
    if torch.get_default_dtype() != torch.float64:
        torch.set_default_dtype(torch.float64)
        print(f"Default dtype set to float64")
    else:
        print(f"Default dtype is already float64")
    
    # Choose which test to run
    import sys
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()
    else:
        test_type = "comprehensive"  # Default to comprehensive test
    
    if test_type == "basic":
        # Run the original precision comparison experiment
        run_precision_comparison(device=device)
    elif test_type == "tf32":
        # Run TF32 vs FP32 comparison if on CUDA
        if device.startswith('cuda'):
            test_tf32_vs_fp32(device=device)
        else:
            print("TF32 testing requires CUDA device")
    elif test_type == "comprehensive":
        # Run comprehensive precision testing
        comprehensive_precision_test(device=device)
    else:
        print(f"Unknown test type: {test_type}")
        print("Available options: basic, tf32, comprehensive")
        print("Running comprehensive test by default...")
        comprehensive_precision_test(device=device)

if __name__ == "__main__":
    main()