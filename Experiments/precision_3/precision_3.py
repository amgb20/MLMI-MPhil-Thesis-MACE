import torch
from mace import data, modules, tools
import numpy as np
import torch.nn.functional
from e3nn import o3
import ase.io
from mace.modules.wrapper_ops import CuEquivarianceConfig, OEQConfig


import warnings
import copy
import pandas as pd
warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")


try:
    import cuequivariance as cue
    cueq_available = True
    print("✓ cuEquivariance library is available")
except ImportError:
    cueq_available = False
    print("✗ cuEquivariance library is not available - cuEq will be disabled")

def get_default_model_config():
    # setup some default prameters
    z_table = tools.AtomicNumberTable([1, 6, 8])
    atomic_energies = np.array([-1.0, -3.0, -5.0], dtype=float)
    cutoff = 3

    # Add cuEq configuration as a CuEquivarianceConfig instance
    cueq_config = CuEquivarianceConfig(
        enabled=True,
        layout="ir_mul",
        group="O3_e3nn",
        optimize_all=True,
    )

    oeq_config = OEQConfig(
        enabled=True,
        optimize_all=True,
        conv_fusion="atomic",  # or another supported value
    )

    default_model_config = dict(
            num_elements=3,  # number of chemical elements
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
    single_molecule = ase.io.read('Experiments/Official MACE notebook/data/solvent_rotated.xyz', index='0')

    Rcut = 3.0 # cutoff radius
    z_table = tools.AtomicNumberTable([1, 6, 8])

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

def run_precision_comparison(device='cpu'):
    # Prepare data and model config
    default_model_config = get_default_model_config()
    batch, lengths, vectors, z_table = data_prep()
    
    # 1) Build one "master" FP64 model and capture its state
    torch.set_default_dtype(torch.float64)
    master = modules.MACE(**default_model_config).to(device=device, dtype=torch.float64)
    master_state = master.state_dict()

    precisions = [torch.float64, torch.float32]
    # if torch.cuda.is_available():
    #     # Try a test operation in FP16 to check support
    #     try:
    #         torch.zeros(1, device='cuda', dtype=torch.float16)
    #         precisions.append(torch.float16)
    #     except Exception:
    #         print("FP16 not supported on this device, skipping FP16.")
    # else:
    #     print("FP16 not supported on CPU, skipping FP16.")

    results = {0: {}, 1: {}}  # block_idx -> dtype -> {'output': ..., 'grad': ...}

    for dtype in precisions:
        print(f"\n=== Testing {dtype} ===")
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
        output0_prod = product0(node_feats=output0, sc=None, node_attrs=inputs0['node_attrs'])
        loss0 = (output0 ** 2).sum()
        loss0.backward()
        grad0 = inputs0['node_feats'].grad
        if grad0 is None:
            raise RuntimeError("Gradient for inputs0['node_feats'] is None. Ensure .retain_grad() is called on the tensor.")
        results[0][dtype] = {
            'output': output0.detach().cpu().double().numpy(),
            'grad': grad0.detach().cpu().double().numpy(),
        }
        # Prepare inputs for block 1
        inputs1 = dict(inputs0)
        node_feats1 = output0_prod.clone().detach().to(device=device, dtype=dtype).requires_grad_(True)
        node_feats1.retain_grad()  # Ensure gradient is retained for non-leaf tensor
        inputs1['node_feats'] = node_feats1
        # Forward block 1
        output1, _ = block1(**inputs1)
        loss1 = (output1 ** 2).sum()
        loss1.backward()
        grad1 = inputs1['node_feats'].grad
        if grad1 is None:
            raise RuntimeError("Gradient for inputs1['node_feats'] is None. Ensure .retain_grad() is called on the tensor.")
        results[1][dtype] = {
            'output': output1.detach().cpu().double().numpy(),
            'grad': grad1.detach().cpu().double().numpy(),
        }

    # Now compare for both blocks and output as DataFrame
    all_dfs = {}
    for block_idx in [0, 1]:
        print(f"\n=== Interaction Block {block_idx} ===")
        ref = results[block_idx][torch.float64]
        rows = []
        for dtype in [torch.float32, torch.float16] if torch.float16 in results[block_idx] else [torch.float32]:
            # Forward error
            err_fwd = np.abs(ref['output'] - results[block_idx][dtype]['output'])
            rel_fwd = np.abs(err_fwd / (np.abs(ref['output']) + 1e-12))
            # Backward error
            err_bwd = np.abs(ref['grad'] - results[block_idx][dtype]['grad'])
            rel_bwd = np.abs(err_bwd / (np.abs(ref['grad']) + 1e-12))
            rows.append({
                'block': block_idx,
                'precision': str(dtype),
                'fwd_max_abs': err_fwd.max(),
                'fwd_max_rel': rel_fwd.max(),
                'bwd_max_abs': err_bwd.max(),
                'bwd_max_rel': rel_bwd.max(),
            })

            df_sub = pd.DataFrame(rows)
            label = str(dtype).replace('torch.float','fp')
            print(f"\nComparison to FP64 for block {block_idx} (FP64 vs. {label}):")
            print(df_sub)

        df = pd.DataFrame(rows, columns=['block', 'precision', 'fwd_max_abs', 'fwd_max_rel', 'bwd_max_abs', 'bwd_max_rel'])
        print(f"\nComparison to FP64 for block {block_idx}:")
        print(df)
        all_dfs[f'block_{block_idx}'] = df

    with pd.ExcelWriter('precision_3.xlsx') as writer:
        for sheet, df in all_dfs.items():
            df.to_excel(writer, sheet_name=sheet, index=False)
    print(f"Excel file saved as precision_3.xlsx")

def main():
    # check if cuda is available and set the device to cuda
    if torch.cuda.is_available():
        device = 'cuda'
    else:   
        device = 'cpu'

    print(f"Device: {device}")

    # check the default dtype
    print(f"Default dtype: {torch.get_default_dtype()}")
    if torch.get_default_dtype() != torch.float64:
        torch.set_default_dtype(torch.float64)
        print(f"Default dtype set to float64")
    else:
        print(f"Default dtype is already float64")
        
    # Run the precision comparison experiment
    run_precision_comparison(device=device)

if __name__ == "__main__":
    main()