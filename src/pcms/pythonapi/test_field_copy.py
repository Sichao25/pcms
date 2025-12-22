import py_pcms
import numpy as np

def test_copy(world, dim, order, num_components):
    """Test copying omega_h_field2 data"""
    nx = 100
    ny = 100 if dim > 1 else 0
    nz = 100 if dim > 2 else 0
    
    # Build mesh
    mesh = py_pcms.build_box(
        world, 
        py_pcms.OMEGA_H_SIMPLEX, 
        1.0, 1.0, 1.0, 
        nx, ny, nz, 
        False
    )
    
    # Create layout
    layout = py_pcms.CreateLagrangeLayout(
        mesh, 
        order, 
        num_components,
        py_pcms.CoordinateSystem.Cartesian
    )
    
    # Get number of data points
    ndata = layout.GetNumOwnedDofHolder() * num_components
    
    # Create sequential array of IDs
    ids = np.arange(ndata, dtype=np.float64)
    
    # Create original field and set data
    original = layout.CreateField()
    original.SetDOFHolderData(ids)
    
    # Create copied field and copy data
    copied = layout.CreateField()
    py_pcms.copy_field2(original, copied)
    
    # Get copied data
    copied_array = copied.GetDOFHolderData()
    
    # Verify the copy
    assert len(copied_array) == ndata, f"Expected {ndata} elements, got {len(copied_array)}"
    
    # Check that all values match
    differences = np.abs(ids - copied_array)
    num_matches = np.sum(differences < 1e-12)
    
    assert num_matches == ndata, f"Only {num_matches}/{ndata} elements matched"
    
    print(f"✓ Test passed: dim={dim}, order={order}, num_components={num_components}")

def main():
    """Run all test cases"""
    print("Testing copy omega_h_field2 data...")
    
    # Initialize Omega_h library
    lib = py_pcms.Library()
    world = lib.world()
    
    # Run test cases
    test_copy(world, 2, 1, 1)
    test_copy(world, 2, 2, 1)
    
    print("\nAll tests passed!")

if __name__ == "__main__":
    main()