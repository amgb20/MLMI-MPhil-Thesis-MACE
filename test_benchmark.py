#!/usr/bin/env python3
"""
Test script to verify the benchmark functionality
"""

import sys
import os

# Add the mace directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'mace'))

def test_imports():
    """Test that all required imports work"""
    try:
        import torch
        print("✓ PyTorch imported successfully")
        
        from e3nn import o3
        print("✓ e3nn imported successfully")
        
        from mace.modules.blocks import RealAgnosticInteractionBlock
        print("✓ RealAgnosticInteractionBlock imported successfully")
        
        from mace.modules.wrapper_ops import CuEquivarianceConfig, OEQConfig
        print("✓ CUEQ configs imported successfully")
        
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_cueq_availability():
    """Test if CUEQ is available"""
    try:
        import cuequivariance as cue
        import cuequivariance_torch as cuet
        print("✓ CUEQ libraries available")
        return True
    except ImportError:
        print("⚠ CUEQ libraries not available - will use fallback")
        return False

def test_device():
    """Test device availability"""
    import torch
    
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        return "cuda"
    else:
        print("⚠ CUDA not available, using CPU")
        return "cpu"

def main():
    """Main test function"""
    print("Testing benchmark setup...")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("❌ Import test failed")
        return False
    
    # Test CUEQ availability
    cueq_available = test_cueq_availability()
    
    # Test device
    device = test_device()
    
    print("\n" + "=" * 50)
    print("Setup test completed!")
    
    if cueq_available:
        print("✅ Ready to run CUEQ vs non-CUEQ benchmarks")
    else:
        print("⚠️  CUEQ not available - benchmarks will only test non-CUEQ")
    
    print(f"📱 Device: {device}")
    
    return True

if __name__ == "__main__":
    main() 