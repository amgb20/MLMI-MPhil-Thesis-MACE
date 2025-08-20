import time
import numpy as np
from ase import build
from mace.calculators import MACECalculator
import torch

def analyze_model_structure(model_path):
    """Analyze the MACE model structure to understand graph complexity"""
    print("=== MODEL STRUCTURE ANALYSIS ===\n")
    
    try:
        # Load the model
        model = torch.load(model_path, map_location='cpu')
        print(f"Model loaded successfully")
        
        # Known model parameters
        print(f"Known model parameters:")
        print(f"  Cutoff radius: 6.0 Å")
        print(f"  Chemical channels (k): 128")
        print(f"  MAC L: 1")
        print(f"  SPICE version: 2")
        
        # Get model info
        if hasattr(model, 'num_interactions'):
            print(f"Number of interaction layers: {model.num_interactions}")
        
        # Check for graph-related attributes
        if hasattr(model, 'max_ell'):
            print(f"Maximum angular momentum: {model.max_ell}")
        
        # Look for specific MACE attributes
        print("\nModel architecture details:")
        for attr in dir(model):
            if not attr.startswith('_') and not callable(getattr(model, attr)):
                try:
                    value = getattr(model, attr)
                    if hasattr(value, 'shape'):
                        print(f"  {attr}: {value.shape}")
                    elif hasattr(value, '__len__') and len(value) < 100:
                        print(f"  {attr}: {value}")
                    else:
                        print(f"  {attr}: {type(value)}")
                except:
                    print(f"  {attr}: {type(value)}")
        
        # Look for interaction blocks
        print("\nInteraction blocks:")
        if hasattr(model, 'interactions'):
            for i, interaction in enumerate(model.interactions):
                print(f"  Interaction {i}: {type(interaction).__name__}")
                if hasattr(interaction, 'conv_tp'):
                    print(f"    Has tensor product convolution")
                if hasattr(interaction, 'max_ell'):
                    print(f"    Max ell: {interaction.max_ell}")
        
        return model
        
    except Exception as e:
        print(f"Error analyzing model: {e}")
        return None

def create_large_test_system():
    """Create a larger test system to stress the GPU"""
    print("\n=== CREATING LARGE TEST SYSTEM ===\n")
    
    # Option 1: Large molecule
    try:
        # Try to create a larger molecule
        large_mol = build.molecule('C60')  # Buckminsterfullerene
        print(f"Created C60 molecule: {len(large_mol)} atoms")
        return large_mol
    except:
        pass
    
    # Option 2: Crystal structure
    try:
        # Create a crystal structure
        from ase.build import bulk
        crystal = bulk('Si', 'diamond', a=5.43, size=(4, 4, 4))  # 512 atoms
        print(f"Created Si crystal: {len(crystal)} atoms")
        return crystal
    except:
        pass
    
    # Option 3: Nanoparticle
    try:
        # Create a nanoparticle
        from ase.cluster import FaceCenteredCubic
        nanoparticle = FaceCenteredCubic('Au', [(1, 0, 0), (1, 1, 0), (1, 1, 1)], [6, 6, 6])
        print(f"Created Au nanoparticle: {len(nanoparticle)} atoms")
        return nanoparticle
    except:
        pass
    
    # Option 4: Custom large system
    try:
        # Create a large random system
        from ase import Atoms
        import numpy as np
        
        # Generate random positions for many atoms
        n_atoms = 1000
        positions = np.random.rand(n_atoms, 3) * 20.0  # 20Å box
        elements = ['C'] * (n_atoms // 2) + ['H'] * (n_atoms // 2)  # Mix of C and H
        
        large_system = Atoms(elements, positions=positions)
        print(f"Created random system: {len(large_system)} atoms")
        return large_system
    except Exception as e:
        print(f"Error creating large system: {e}")
        return None
    
    # Option 5: Ultra-large system for extreme GPU stress
    try:
        # Create an ultra-large system that will really stress the GPU
        from ase import Atoms
        import numpy as np
        
        # Generate random positions for many atoms
        n_atoms = 2000  # Double the size
        positions = np.random.rand(n_atoms, 3) * 30.0  # 30Å box for better spacing
        elements = ['C'] * (n_atoms // 3) + ['H'] * (n_atoms // 3) + ['O'] * (n_atoms // 3)  # Mix of C, H, O
        
        ultra_large_system = Atoms(elements, positions=positions)
        print(f"Created ultra-large system: {len(ultra_large_system)} atoms")
        return ultra_large_system
    except Exception as e:
        print(f"Error creating ultra-large system: {e}")
        return None

def estimate_graph_complexity(atoms, cutoff=6.0):
    """Estimate the number of edges and nodes in the molecular graph"""
    print(f"\n=== GRAPH COMPLEXITY ESTIMATION ===")
    print(f"System: {atoms.get_chemical_formula()} with {len(atoms)} atoms")
    
    # Model-specific parameters
    print(f"Model parameters:")
    print(f"  Cutoff radius: {cutoff} Å")
    print(f"  Chemical channels (k): 128")
    print(f"  MAC L: 1")
    print(f"  SPICE version: 2")
    
    # Estimate edges based on cutoff radius
    # For 6.0Å cutoff, each atom connects to more neighbors than 5.0Å
    # Typical molecular densities give ~20-40 neighbors per atom at 6.0Å
    estimated_edges_per_atom = 30  # More realistic for 6.0Å cutoff
    total_estimated_edges = len(atoms) * estimated_edges_per_atom
    
    print(f"\nGraph complexity:")
    print(f"  Nodes (atoms): {len(atoms)}")
    print(f"  Estimated edges (connections): ~{total_estimated_edges}")
    print(f"  Estimated edges per atom: ~{estimated_edges_per_atom}")
    
    # Calculate computational complexity
    # Each edge involves tensor operations with 128 chemical channels
    total_tensor_operations = total_estimated_edges * 128
    print(f"  Total tensor operations: ~{total_tensor_operations:,}")
    
    # Calculate system volume and density
    if hasattr(atoms, 'get_cell'):
        cell = atoms.get_cell()
        if cell.any():
            volume = abs(np.linalg.det(cell))
            density = len(atoms) / volume
            print(f"\nSystem properties:")
            print(f"  Volume: {volume:.2f} Å³")
            print(f"  Atomic density: {density:.2f} atoms/Å³")
    
    # Estimate memory usage
    # FP64: 8 bytes per value, FP32: 4 bytes per value
    fp64_memory_mb = (total_estimated_edges * 128 * 8) / (1024 * 1024)
    fp32_memory_mb = (total_estimated_edges * 128 * 4) / (1024 * 1024)
    print(f"\nEstimated memory usage:")
    print(f"  FP64: ~{fp64_memory_mb:.1f} MB")
    print(f"  FP32: ~{fp32_memory_mb:.1f} MB")
    print(f"  Memory savings with FP32: ~{fp64_memory_mb - fp32_memory_mb:.1f} MB")
    
    return len(atoms), total_estimated_edges

def compare_precision_fp64_vs_fp32(use_large_system=False):
    """Compare inference results between FP64 and FP32 precision using torch.cuda.Event"""
    
    print("=== PRECISION COMPARISON: FP64 vs FP32 ===\n")
    
    # Choose test system
    if use_large_system:
        atoms = create_large_test_system()
        if atoms is None:
            print("Failed to create large system, falling back to H2O")
            atoms = build.molecule('H2O')
    else:
        atoms = build.molecule('H2O')
    
    print(f"Testing molecule: {atoms.get_chemical_formula()} with {len(atoms)} atoms")
    
    # Estimate graph complexity
    estimate_graph_complexity(atoms)
    
    # Ensure positions are in the right format
    positions = atoms.get_positions()
    if positions.dtype != np.float64:
        print("Converting positions to float64...")
        atoms.set_positions(positions.astype(np.float64))
    
    # Test FP64 (double precision)
    print("\nTesting FP64 (double precision)...")
    try:
        print("Loading FP64 model...")
        start_load = time.time()
        mace_calc_fp64 = MACECalculator(
            model_paths="Experiments/numerical_stability/src/inference/model/MACE-OFF24_medium.model", 
            default_dtype="float64",
            enable_cueq=True,
            device='cuda:1'
        )
        fp64_load_time = (time.time() - start_load) * 1000
        print(f"FP64 model loaded in {fp64_load_time:.2f} ms")
        
        atoms.calc = mace_calc_fp64
        
        # Warm-up run (first run is usually slower due to compilation)
        print("Running warm-up...")
        _ = atoms.get_potential_energy()
        _ = atoms.get_forces()
        
        # Use torch.cuda.Event for precise timing
        if torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            # Multiple runs for more accurate timing
            times_fp64 = []
            for run in range(5):
                torch.cuda.synchronize()  # Ensure previous operations are complete
                start_event.record()
                energy_fp64 = atoms.get_potential_energy()
                forces_fp64 = atoms.get_forces()
                end_event.record()
                torch.cuda.synchronize()
                run_time = start_event.elapsed_time(end_event)
                times_fp64.append(run_time)
                print(f"  Run {run+1}: {run_time:.2f} ms")
            
            fp64_time_ms = np.mean(times_fp64)
            fp64_time_std = np.std(times_fp64)
        else:
            start_time = time.time()
            energy_fp64 = atoms.get_potential_energy()
            forces_fp64 = atoms.get_forces()
            fp64_time_ms = (time.time() - start_time) * 1000  # Convert to ms
            fp64_time_std = 0
        
        print(f"FP64 Energy: {energy_fp64:.12f} eV")
        print(f"FP64 Max Force: {np.max(np.linalg.norm(forces_fp64, axis=1)):.12f} eV/Å")
        print(f"FP64 Time: {fp64_time_ms:.2f} ± {fp64_time_std:.2f} ms (avg of 5 runs)")
        
    except Exception as e:
        print(f"FP64 calculation failed: {e}")
        return None
    
    # Test FP32 (single precision)
    print("\nTesting FP32 (single precision)...")
    try:
        print("Loading FP32 model...")
        start_load = time.time()
        mace_calc_fp32 = MACECalculator(
            model_paths="Experiments/numerical_stability/src/inference/model/MACE-OFF24_medium.model", 
            default_dtype="float32",
            enable_cueq=True,
            device='cuda:1'
        )
        fp32_load_time = (time.time() - start_load) * 1000
        print(f"FP32 model loaded in {fp32_load_time:.2f} ms")

        atoms.calc = mace_calc_fp32
        
        # Warm-up run
        print("Running warm-up...")
        _ = atoms.get_potential_energy()
        _ = atoms.get_forces()
        
        # Use torch.cuda.Event for precise timing
        if torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            # Multiple runs for more accurate timing
            times_fp32 = []
            for run in range(5):
                torch.cuda.synchronize()  # Ensure previous operations are complete
                start_event.record()
                energy_fp32 = atoms.get_potential_energy()
                forces_fp32 = atoms.get_forces()
                end_event.record()
                torch.cuda.synchronize()
                run_time = start_event.elapsed_time(end_event)
                times_fp32.append(run_time)
                print(f"  Run {run+1}: {run_time:.2f} ms")
            
            fp32_time_ms = np.mean(times_fp32)
            fp32_time_std = np.std(times_fp32)
        else:
            start_time = time.time()
            energy_fp32 = atoms.get_potential_energy()
            forces_fp32 = atoms.get_potential_energy()
            fp32_time_ms = (time.time() - start_time) * 1000  # Convert to ms
            fp32_time_std = 0
        
        print(f"FP32 Energy: {energy_fp32:.12f} eV")
        print(f"FP32 Max Force: {np.max(np.linalg.norm(forces_fp32, axis=1)):.12f} eV/Å")
        print(f"FP32 Time: {fp32_time_ms:.2f} ± {fp32_time_std:.2f} ms (avg of 5 runs)")
        
    except Exception as e:
        print(f"FP32 calculation failed: {e}")
        return None
    
    # Calculate differences
    energy_diff = energy_fp32 - energy_fp64
    energy_diff_percent = (energy_diff / abs(energy_fp64)) * 100 if energy_fp64 != 0 else 0
    
    max_force_fp64 = np.max(np.linalg.norm(forces_fp64, axis=1))
    max_force_fp32 = np.max(np.linalg.norm(forces_fp32, axis=1))
    force_diff = max_force_fp32 - max_force_fp64
    force_diff_percent = (force_diff / max_force_fp64) * 100 if max_force_fp64 != 0 else 0
    
    # Calculate force component differences
    force_component_diffs = forces_fp32 - forces_fp64
    max_force_component_diff = np.max(np.abs(force_component_diffs))
    
    # Performance comparison
    speedup = fp64_time_ms / fp32_time_ms
    
    print(f"\n=== PERFORMANCE ANALYSIS ===")
    print(f"FP64 load time: {fp64_load_time:.2f} ms")
    print(f"FP32 load time: {fp32_load_time:.2f} ms")
    print(f"FP64 inference time: {fp64_time_ms:.2f} ± {fp64_time_std:.2f} ms")
    print(f"FP32 inference time: {fp32_time_ms:.2f} ± {fp32_time_std:.2f} ms")
    
    if speedup > 1:
        print(f"FP32 is {speedup:.2f}x FASTER than FP64 ✓")
    else:
        print(f"FP32 is {1/speedup:.2f}x SLOWER than FP64 ⚠️")
        print("This is unusual - FP32 should typically be faster!")
    
    # Check if the difference is statistically significant
    if fp64_time_std > 0 and fp32_time_std > 0:
        # Simple t-test approximation
        pooled_std = np.sqrt((fp64_time_std**2 + fp32_time_std**2) / 2)
        t_stat = abs(fp64_time_ms - fp32_time_ms) / pooled_std
        if t_stat > 2:  # Rough threshold for significance
            print(f"Timing difference is statistically significant (t={t_stat:.2f})")
        else:
            print(f"Timing difference may not be statistically significant (t={t_stat:.2f})")
    
    print(f"\n=== PRECISION LOSS ANALYSIS ===")
    print(f"Energy difference: {energy_diff:.12f} eV ({energy_diff_percent:.6f}%)")
    print(f"Max force difference: {force_diff:.12f} eV/Å ({force_diff_percent:.6f}%)")
    print(f"Max force component difference: {max_force_component_diff:.12f} eV/Å")
    
    # Determine if precision loss is acceptable
    print(f"\n=== PRECISION ASSESSMENT ===")
    
    # Energy precision assessment
    if abs(energy_diff_percent) < 0.001:
        energy_assessment = "EXCELLENT - Negligible precision loss"
    elif abs(energy_diff_percent) < 0.01:
        energy_assessment = "GOOD - Very small precision loss"
    elif abs(energy_diff_percent) < 0.1:
        energy_assessment = "ACCEPTABLE - Small precision loss"
    else:
        energy_assessment = "POOR - Significant precision loss"
    
    print(f"Energy precision: {energy_assessment}")
    
    # Force precision assessment
    if max_force_component_diff < 0.001:
        force_assessment = "EXCELLENT - Negligible precision loss"
    elif max_force_component_diff < 0.01:
        force_assessment = "GOOD - Very small precision loss"
    elif max_force_component_diff < 0.1:
        force_assessment = "ACCEPTABLE - Small precision loss"
    else:
        force_assessment = "POOR - Significant precision loss"
    
    print(f"Force precision: {force_assessment}")
    
    # Detailed force comparison
    print(f"\n=== DETAILED FORCE COMPARISON ===")
    for i, (atom_fp64, atom_fp32) in enumerate(zip(forces_fp64, forces_fp32)):
        atom_diff = np.linalg.norm(atom_fp32 - atom_fp64)
        print(f"Atom {i+1}: FP64={np.linalg.norm(atom_fp64):.8f}, FP32={np.linalg.norm(atom_fp32):.8f}, Diff={atom_diff:.8f}")
    
    return {
        'fp64': {'energy': energy_fp64, 'forces': forces_fp64, 'time_ms': fp64_time_ms, 'load_time': fp64_load_time},
        'fp32': {'energy': energy_fp32, 'forces': forces_fp32, 'time_ms': fp32_time_ms, 'load_time': fp32_load_time},
        'differences': {
            'energy_diff': energy_diff,
            'energy_diff_percent': energy_diff_percent,
            'force_diff': force_diff,
            'force_diff_percent': force_diff_percent,
            'max_force_component_diff': max_force_component_diff
        },
        'speedup': speedup
    }

if __name__ == "__main__":
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA not available, using CPU timing")
    
    # First, analyze the model structure
    model_path = "Experiments/numerical_stability/src/inference/model/MACE-OFF24_medium.model"
    print(f"\n{'='*60}")
    analyze_model_structure(model_path)
    print(f"{'='*60}")
    
    # Test 1: Small system (H2O)
    print(f"\n{'='*60}")
    print("TEST 1: SMALL SYSTEM (H2O)")
    print(f"{'='*60}")
    results_small = compare_precision_fp64_vs_fp32(use_large_system=True)
    
    if results_small:
        print(f"\n=== SMALL SYSTEM SUMMARY ===")
        print(f"FP64 time: {results_small['fp64']['time_ms']:.2f} ms")
        print(f"FP32 time: {results_small['fp32']['time_ms']:.2f} ms")
        print(f"Speedup: {results_small['speedup']:.2f}x")
        print(f"Precision loss: {results_small['differences']['energy_diff_percent']:.6f}% (energy), {results_small['differences']['max_force_component_diff']:.6f} eV/Å (forces)")
    
    # Test 2: Large system to stress the GPU
    print(f"\n{'='*60}")
    print("TEST 2: LARGE SYSTEM (GPU STRESS TEST)")
    print(f"{'='*60}")
    results_large = compare_precision_fp64_vs_fp32(use_large_system=True)
    
    if results_large:
        print(f"\n=== LARGE SYSTEM SUMMARY ===")
        print(f"FP64 time: {results_large['fp64']['time_ms']:.2f} ms")
        print(f"FP32 time: {results_large['fp32']['time_ms']:.2f} ms")
        print(f"Speedup: {results_large['speedup']:.2f}x")
        print(f"Precision loss: {results_large['differences']['energy_diff_percent']:.6f}% (energy), {results_large['differences']['max_force_component_diff']:.6f} eV/Å (forces)")
        
        # Compare small vs large system performance
        if results_small:
            print(f"\n=== SCALING ANALYSIS ===")
            fp64_scaling = results_large['fp64']['time_ms'] / results_small['fp64']['time_ms']
            fp32_scaling = results_large['fp32']['time_ms'] / results_small['fp32']['time_ms']
            print(f"FP64 scaling factor: {fp64_scaling:.2f}x")
            print(f"FP32 scaling factor: {fp32_scaling:.2f}x")
            print(f"Scaling efficiency: {fp32_scaling/fp64_scaling:.2f}")
    
    # Investigate potential causes if FP32 is slower in either test
    for test_name, results in [("Small System", results_small), ("Large System", results_large)]:
        if results and results['speedup'] < 1:
            print(f"\n=== INVESTIGATION: Why is FP32 slower in {test_name}? ===")
            print("Possible causes:")
            print("1. Model compilation overhead (FP32 might need different kernels)")
            print("2. Memory bandwidth limitations (FP32 uses less memory but might hit other bottlenecks)")
            print("3. CUDA kernel optimization differences")
            print("4. Model architecture differences between precision modes")
            print("5. Timing measurement artifacts")
            
            # Check if load times explain the difference
            load_time_diff = results.get('fp64', {}).get('load_time', 0) - results.get('fp32', {}).get('load_time', 0)
            if abs(load_time_diff) > 10:  # If load time difference is significant
                print(f"Load time difference: {load_time_diff:.2f} ms (this might explain the performance difference)")