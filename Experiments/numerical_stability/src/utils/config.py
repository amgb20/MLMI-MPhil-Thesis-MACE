# 1. STANDARD LIBRARY IMPORTS (alphabetical)
from mace import data, modules, tools
from e3nn import o3
import torch
import numpy as np
import ase.io

# 2. THIRD-PARTY LIBRARY IMPORTS (alphabetical)
import sys
import os

# 3. LOCAL/APPLICATION IMPORTS (alphabetical)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.get_logging_profile import logger


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
    single_molecule = ase.io.read('Experiments/numerical_stability/data/md22_double-walled_nanotube.xyz', index='0')

    # Detect elements present in the dataset
    atomic_numbers = single_molecule.numbers
    unique_atomic_numbers = sorted(set(atomic_numbers))
    logger.info(f"Elements found in dataset: {unique_atomic_numbers}")
    logger.info(f"Element symbols: {single_molecule.get_chemical_symbols()[:10]}...")  # Show first 10 symbols
    
    Rcut = 3.0 # cutoff radius
    # z_table = tools.AtomicNumberTable([1, 6, 8])
    z_table = tools.AtomicNumberTable(unique_atomic_numbers)
    logger.info(f"Created z_table with {len(z_table.zs)} elements: {z_table.zs}")

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
    logger.info(f'there are {batch.positions.shape[0]} nodes and {len(lengths)} edges')
    logger.info(f'lengths is shape {lengths.shape}')
    logger.info(f'vectors is shape {vectors.shape}')

    return batch, lengths, vectors, z_table
