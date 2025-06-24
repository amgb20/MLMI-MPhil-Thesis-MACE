import mace
import torch
import copy
import logging
import numpy as np
from typing import Dict, List, Tuple

# Configure logging to show INFO level messages
logging.basicConfig(level=logging.INFO, format='%(message)s')

# TODO: check that cuEq is enabled
# TODO: check that you can enter conv_tp and conv_fusion

from mace.calculators import mace_off
from ase import build

def setup_model_and_data():
    """Setup MACE model and test data"""
    atoms = build.molecule('H2O')
    
    # check if cuda is available and set the device to cuda
    if torch.cuda.is_available():
        device = 'cuda'
    else:   
        device = 'cpu'
    
    calc = mace_off(model="medium", device=device)
    logging.info(f"Device: {device}")
    atoms.calc = calc
    
    ssmace = calc.models
    model = ssmace[0]
    
    logging.info("\n==== Model ====")
    logging.info(model)
    
    return model, atoms, calc, device

def get_fixed_fp64_embeddings(model, atoms, calc):
    """Get fixed fp64 embeddings that will be used as reference"""
    logging.info("==== GETTING FIXED FP64 EMBEDDINGS ====")
    
    # Convert model to fp64 for reference embeddings
    model_fp64 = model.double()
    
    # Check model parameters dtype
    first_param = next(model_fp64.parameters())
    logging.info(f"Model parameters dtype: {first_param.dtype}")
    
    # Get batch data
    if hasattr(calc, '_atoms_to_batch'):
        batch = calc._atoms_to_batch(atoms)
        logging.info(f"Batch keys: {list(batch.keys())}")
        
        # Convert batch to fp64
        batch_fp64 = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                batch_fp64[key] = value.double()
            else:
                batch_fp64[key] = value
        
        # Get embeddings in fp64
        with torch.no_grad():
            # Get node embeddings
            node_feats_fp64 = model_fp64.node_embedding(batch_fp64["node_attrs"])
            
            # Get edge embeddings
            vectors = batch_fp64["shifts"]  # displacement vectors
            edge_attrs_fp64 = model_fp64.spherical_harmonics(vectors)
            lengths = torch.norm(vectors, dim=-1)
            edge_feats_fp64, cutoff_fp64 = model_fp64.radial_embedding(
                lengths, batch_fp64["node_attrs"], batch_fp64["edge_index"], model_fp64.atomic_numbers
            )
            
            logging.info(f"Node features fp64 shape: {node_feats_fp64.shape}")
            logging.info(f"Edge attributes fp64 shape: {edge_attrs_fp64.shape}")
            logging.info(f"Edge features fp64 shape: {edge_feats_fp64.shape}")
            
            return {
                'node_feats': node_feats_fp64,
                'edge_attrs': edge_attrs_fp64,
                'edge_feats': edge_feats_fp64,
                'cutoff': cutoff_fp64,
                'batch': batch_fp64,
                'model_fp64': model_fp64
            }
    else:
        logging.error("Cannot access _atoms_to_batch method")
        return None

def test_interaction_precision(embeddings, model, target_dtype=torch.float32):
    """
    Test interaction precision with fixed fp64 embeddings
    Follows the actual ScaleShiftMACE forward pass structure
    """
    logging.info(f"==== TESTING INTERACTION PRECISION: {target_dtype} ====")
    
    # Convert model to target dtype
    model_target = model.to(target_dtype)
    
    # Convert embeddings to target dtype (except keep fp64 reference)
    node_feats = embeddings['node_feats'].to(target_dtype)
    edge_attrs = embeddings['edge_attrs'].to(target_dtype)
    edge_feats = embeddings['edge_feats'].to(target_dtype)
    cutoff = embeddings['cutoff'].to(target_dtype) if embeddings['cutoff'] is not None else None
    batch = embeddings['batch']
    
    # Convert batch tensors to target dtype
    batch_target = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            batch_target[key] = value.to(target_dtype)
        else:
            batch_target[key] = value
    
    results = {}
    
    # Get atomic energies (reference energies for each atom type)
    num_atoms_arange = torch.arange(len(batch_target["node_attrs"]))
    node_heads = batch_target.get("node_heads", torch.zeros(len(batch_target["node_attrs"]), dtype=torch.long))
    
    # Get atomic energies for each atom
    node_e0 = model_target.atomic_energies_fn(batch_target["node_attrs"])[num_atoms_arange, node_heads]
    e0 = torch.sum(node_e0)  # Sum over all atoms for total atomic energy
    
    logging.info(f"Atomic energies shape: {node_e0.shape}")
    logging.info(f"Total atomic energy: {e0.item()}")
    
    # Test each interaction layer following ScaleShiftMACE structure
    node_feats_current = node_feats.clone()
    node_es_list = []  # Store energy contributions from each layer
    
    for i, (interaction, product, readout) in enumerate(zip(
        model_target.interactions, model_target.products, model_target.readouts
    )):
        logging.info(f"Testing Interaction {i}")
        
        # Convert components to target dtype
        interaction = interaction.to(target_dtype)
        product = product.to(target_dtype)
        readout = readout.to(target_dtype)
        
        # Forward pass through interaction (following ScaleShiftMACE structure)
        node_feats_out, sc = interaction(
            node_attrs=batch_target["node_attrs"],
            node_feats=node_feats_current,
            edge_attrs=edge_attrs,
            edge_feats=edge_feats,
            edge_index=batch["edge_index"],  # Keep original for indexing
            cutoff=cutoff,
            first_layer=(i == 0),
            lammps_class=None,
            lammps_natoms=None,
        )
        
        # Apply product layer
        node_feats_out = product(
            node_feats=node_feats_out,
            sc=sc,
            node_attrs=batch_target["node_attrs"]
        )
        
        # Apply readout layer to get energy contribution
        node_es = readout(node_feats_out, node_heads)[num_atoms_arange, node_heads]
        node_es_list.append(node_es)
        
        results[f'interaction_{i}'] = {
            'node_feats': node_feats_out,
            'node_energy': node_es,
            'dtype': target_dtype
        }
        
        # Update node_feats for next interaction
        node_feats_current = node_feats_out
    
    # Apply scale/shift to interaction energies (ScaleShiftMACE specific)
    if len(node_es_list) > 0:
        node_inter_es = torch.sum(torch.stack(node_es_list, dim=0), dim=0)
        node_inter_es = model_target.scale_shift(node_inter_es, node_heads)
        inter_e = torch.sum(node_inter_es)  # Sum over all atoms
        
        # Total energy = atomic energy + interaction energy
        total_energy = e0 + inter_e
        
        results['final'] = {
            'total_energy': total_energy,
            'atomic_energy': e0,
            'interaction_energy': inter_e,
            'node_inter_energies': node_inter_es,
            'dtype': target_dtype
        }
    
    return results

def calculate_forward_error(results_fp64, results_fp32):
    """Calculate forward pass error between fp64 and fp32 results"""
    logging.info("==== CALCULATING FORWARD PASS ERROR ====")
    
    errors = {}
    
    for interaction_key in results_fp64.keys():
        if interaction_key in results_fp32:
            # Calculate energy differences
            if 'node_energy' in results_fp64[interaction_key]:
                energy_fp64 = results_fp64[interaction_key]['node_energy']
                energy_fp32 = results_fp32[interaction_key]['node_energy']
            elif 'total_energy' in results_fp64[interaction_key]:
                energy_fp64 = results_fp64[interaction_key]['total_energy']
                energy_fp32 = results_fp32[interaction_key]['total_energy']
            else:
                continue
            
            # Convert fp32 to fp64 for comparison
            energy_fp32_fp64 = energy_fp32.double()
            
            # Calculate absolute error
            abs_error = torch.abs(energy_fp64 - energy_fp32_fp64)
            rel_error = abs_error / (torch.abs(energy_fp64) + 1e-12)
            
            errors[interaction_key] = {
                'abs_error': abs_error,
                'rel_error': rel_error,
                'abs_error_mean': abs_error.mean().item(),
                'abs_error_max': abs_error.max().item(),
                'rel_error_mean': rel_error.mean().item(),
                'rel_error_max': rel_error.max().item(),
            }
            
            logging.info(f"{interaction_key}:")
            logging.info(f"  Abs Error - Mean: {errors[interaction_key]['abs_error_mean']:.2e}, Max: {errors[interaction_key]['abs_error_max']:.2e}")
            logging.info(f"  Rel Error - Mean: {errors[interaction_key]['rel_error_mean']:.2e}, Max: {errors[interaction_key]['rel_error_max']:.2e}")
    
    return errors

def calculate_backward_error(model, embeddings, target_dtype=torch.float32):
    """
    Calculate backward pass error: err_bwd = abs(Gradient_x L_fp64 - Gradient_x L_fp32)
    where L = sum(E_i^2) is a scalar loss function
    Follows the actual ScaleShiftMACE forward pass structure
    """
    logging.info(f"==== CALCULATING BACKWARD PASS ERROR: {target_dtype} ====")
    
    # Convert model to target dtype
    model_target = model.to(target_dtype)
    
    # Enable gradients
    model_target.zero_grad()
    
    # Get embeddings in target dtype
    node_feats = embeddings['node_feats'].to(target_dtype)
    edge_attrs = embeddings['edge_attrs'].to(target_dtype)
    edge_feats = embeddings['edge_feats'].to(target_dtype)
    cutoff = embeddings['cutoff'].to(target_dtype) if embeddings['cutoff'] is not None else None
    batch = embeddings['batch']
    
    # Convert batch tensors to target dtype
    batch_target = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            batch_target[key] = value.to(target_dtype)
        else:
            batch_target[key] = value
    
    # Forward pass following ScaleShiftMACE structure
    num_atoms_arange = torch.arange(len(batch_target["node_attrs"]))
    node_heads = batch_target.get("node_heads", torch.zeros(len(batch_target["node_attrs"]), dtype=torch.long))
    
    # Get atomic energies
    node_e0 = model_target.atomic_energies_fn(batch_target["node_attrs"])[num_atoms_arange, node_heads]
    e0 = torch.sum(node_e0)
    
    # Forward pass through all interactions
    node_feats_current = node_feats.clone()
    node_es_list = []
    
    for i, (interaction, product, readout) in enumerate(zip(
        model_target.interactions, model_target.products, model_target.readouts
    )):
        interaction = interaction.to(target_dtype)
        product = product.to(target_dtype)
        readout = readout.to(target_dtype)
        
        # Forward pass
        node_feats_out, sc = interaction(
            node_attrs=batch_target["node_attrs"],
            node_feats=node_feats_current,
            edge_attrs=edge_attrs,
            edge_feats=edge_feats,
            edge_index=batch["edge_index"],
            cutoff=cutoff,
            first_layer=(i == 0),
            lammps_class=None,
            lammps_natoms=None,
        )
        
        node_feats_out = product(
            node_feats=node_feats_out,
            sc=sc,
            node_attrs=batch_target["node_attrs"]
        )
        
        node_es = readout(node_feats_out, node_heads)[num_atoms_arange, node_heads]
        node_es_list.append(node_es)
        node_feats_current = node_feats_out
    
    # Apply scale/shift and get final energies
    if len(node_es_list) > 0:
        node_inter_es = torch.sum(torch.stack(node_es_list, dim=0), dim=0)
        node_inter_es = model_target.scale_shift(node_inter_es, node_heads)
        inter_e = torch.sum(node_inter_es)
        total_energy = e0 + inter_e
    else:
        total_energy = e0
    
    # Calculate loss: L = sum(E_i^2) - using total energy as the scalar
    loss = total_energy ** 2  # Since we have one total energy, square it
    
    # Backward pass
    loss.backward()
    
    # Collect gradients
    gradients = {}
    for name, param in model_target.named_parameters():
        if param.grad is not None:
            gradients[name] = param.grad.clone()
    
    return {
        'loss': loss.item(),
        'total_energy': total_energy.item(),
        'gradients': gradients
    }

def compare_backward_errors(results_fp64, results_fp32):
    """Compare backward pass errors between fp64 and fp32"""
    logging.info("==== COMPARING BACKWARD PASS ERRORS ====")
    
    gradients_fp64 = results_fp64['gradients']
    gradients_fp32 = results_fp32['gradients']
    
    gradient_errors = {}
    
    for param_name in gradients_fp64.keys():
        if param_name in gradients_fp32:
            grad_fp64 = gradients_fp64[param_name]
            grad_fp32 = gradients_fp32[param_name]
            
            # Convert fp32 to fp64 for comparison
            grad_fp32_fp64 = grad_fp32.double()
            
            # Calculate absolute error
            abs_error = torch.abs(grad_fp64 - grad_fp32_fp64)
            rel_error = abs_error / (torch.abs(grad_fp64) + 1e-12)
            
            gradient_errors[param_name] = {
                'abs_error_mean': abs_error.mean().item(),
                'abs_error_max': abs_error.max().item(),
                'rel_error_mean': rel_error.mean().item(),
                'rel_error_max': rel_error.max().item(),
            }
            
            logging.info(f"{param_name}:")
            logging.info(f"  Abs Error - Mean: {gradient_errors[param_name]['abs_error_mean']:.2e}, Max: {gradient_errors[param_name]['abs_error_max']:.2e}")
            logging.info(f"  Rel Error - Mean: {gradient_errors[param_name]['rel_error_mean']:.2e}, Max: {gradient_errors[param_name]['rel_error_max']:.2e}")
    
    return gradient_errors

def main():
    # Setup
    model, atoms, calc, device = setup_model_and_data()
    
    # Get fixed fp64 embeddings
    embeddings = get_fixed_fp64_embeddings(model, atoms, calc)
    if embeddings is None:
        logging.error("Failed to get embeddings")
        return
    
    # Test forward pass with different precisions
    logging.info("\n" + "="*50)
    logging.info("FORWARD PASS TESTING")
    logging.info("="*50)
    
    results_fp64 = test_interaction_precision(embeddings, model, torch.float64)
    results_fp32 = test_interaction_precision(embeddings, model, torch.float32)
    
    # Calculate forward pass errors
    forward_errors = calculate_forward_error(results_fp64, results_fp32)
    
    # Test backward pass with different precisions
    logging.info("\n" + "="*50)
    logging.info("BACKWARD PASS TESTING")
    logging.info("="*50)
    
    backward_fp64 = calculate_backward_error(model, embeddings, torch.float64)
    backward_fp32 = calculate_backward_error(model, embeddings, torch.float32)
    
    # Compare backward pass errors
    gradient_errors = compare_backward_errors(backward_fp64, backward_fp32)
    
    # Summary
    logging.info("\n" + "="*50)
    logging.info("TESTBENCH SUMMARY")
    logging.info("="*50)
    logging.info(f"Forward pass errors calculated for {len(forward_errors)} interactions")
    logging.info(f"Backward pass errors calculated for {len(gradient_errors)} parameters")
    logging.info(f"Loss fp64: {backward_fp64['loss']:.6e}")
    logging.info(f"Loss fp32: {backward_fp32['loss']:.6e}")
    logging.info("Testbench completed successfully!")

if __name__ == "__main__":
    main()