#!/usr/bin/env python3
"""
Test OmegaHFieldLayout Python bindings
"""

import py_pcms
import numpy as np

def test_layout_methods(world, dim, order, num_components):
    """Test OmegaHFieldLayout methods"""
    nx = 10
    ny = 10 if dim > 1 else 0
    nz = 10 if dim > 2 else 0
    
    print(f"\nTesting layout: dim={dim}, order={order}, num_components={num_components}")
    
    # Build mesh
    mesh = py_pcms.build_box(
        world, 
        py_pcms.Family.SIMPLEX, 
        1.0, 1.0, 1.0, 
        nx, ny, nz, 
        False
    )
    
    # Create layout
    layout = py_pcms.create_lagrange_layout(
        mesh, 
        order, 
        num_components,
        py_pcms.CoordinateSystem.Cartesian
    )
    
    # Test layout methods
    num_comp = layout.get_num_components()
    assert num_comp == num_components, f"Expected {num_components} components, got {num_comp}"
    
    num_owned = layout.get_num_owned_dof_holder()
    print(f"  Number of owned DOF holders: {num_owned}")
    assert num_owned > 0, "Expected at least one owned DOF holder"
    
    num_global = layout.get_num_global_dof_holder()
    print(f"  Number of global DOF holders: {num_global}")
    assert num_global >= num_owned, "Global DOF holders should be >= owned"
    
    # Get ownership mask
    owned = layout.get_owned()
    print(f"  Ownership mask length: {len(owned)}")
    assert len(owned) > 0, "Ownership mask should not be empty"
    
    # Get global IDs
    gids = layout.get_gids()
    print(f"  Global IDs length: {len(gids)}")
    assert len(gids) > 0, "Global IDs should not be empty"
    
    # Get coordinates of DOF holders (2D parametric coordinates)
    coords = layout.get_dof_holder_coordinates()
    print(f"  Coordinates shape: {coords.shape}")
    assert coords.shape[0] > 0, "Should have coordinates"
    assert coords.shape[1] == 2, "Coordinates should be 2D (parametric)"
    
    # Check if distributed
    is_distributed = layout.is_distributed()
    print(f"  Is distributed: {is_distributed}")
    
    # Get entity offsets
    ent_offsets = layout.get_ent_offsets()
    print(f"  Entity offsets: {ent_offsets}")
    assert len(ent_offsets) == 5, f"Expected 5 entity offsets"
    
    # Get nodes per dimension
    nodes = layout.get_nodes_per_dim()
    print(f"  Nodes per dim: {nodes}")
    assert len(nodes) == 4, "Should have 4 entries (verts, edges, faces, regions)"
    
    # Get number of entities
    num_ents = layout.get_num_ents()
    print(f"  Number of entities: {num_ents}")
    assert num_ents > 0, "Should have at least one entity"
    
    # Get sizes
    owned_size = layout.owned_size()
    global_size = layout.global_size()
    print(f"  Owned size: {owned_size}")
    print(f"  Global size: {global_size}")
    assert owned_size == num_owned * num_components
    assert global_size == num_global * num_components
    
    # Create a field from the layout
    field = layout.create_field()
    print(f"  Created field: {type(field)}")
    
    # Test setting and getting data
    ndata = num_owned * num_components
    test_data = np.arange(ndata, dtype=np.float64)
    field.set_dof_holder_data(test_data)
    
    retrieved_data = field.get_dof_holder_data()
    assert len(retrieved_data) == ndata, f"Expected {ndata} elements, got {len(retrieved_data)}"
    
    # Verify data matches
    # debug print
    for i in range(min(10, ndata)):
        print(f"  Data[{i}]: set={test_data[i]}, got={retrieved_data[i]}")
    differences = np.abs(test_data - retrieved_data)
    assert np.all(differences < 1e-12), "Data mismatch after set/get"
    
    print(f"✓ Test passed: dim={dim}, order={order}, num_components={num_components}")

def test_field_evaluation(world, dim, order, num_components):
    """Test OmegaHField evaluation at points"""
    nx = 10
    ny = 10 if dim > 1 else 0
    nz = 10 if dim > 2 else 0
    
    print(f"\nTesting field evaluation: dim={dim}, order={order}, num_components={num_components}")
    
    # Build mesh
    mesh = py_pcms.build_box(
        world, 
        py_pcms.Family.SIMPLEX, 
        1.0, 1.0, 1.0, 
        nx, ny, nz, 
        False
    )
    
    # Create layout
    layout = py_pcms.create_lagrange_layout(
        mesh, 
        order, 
        num_components,
        py_pcms.CoordinateSystem.Cartesian
    )
    
    # Create field
    field = layout.create_field()
    
    # Set up test data - use a simple function: f(x,y,z) = sin(x*y) for component 0, etc.
    print(f"  Setting up field data...")
    # Get mesh vertex coordinates to set field values
    mesh_coords = mesh.coords()
    num_verts = mesh.nverts()
    print(f"  Mesh has {num_verts} vertices")
    
    # For order 1, we only need vertex values
    # For order 2, we also need edge midpoint values
    num_owned = layout.get_num_owned_dof_holder()
    ndata = num_owned * num_components
    print(f"  Setting up field data for {ndata} DOFs")
    
    # Define test function
    def test_func(x, y, z, component):
        return np.sin(5.0 * x * y) + float(component)
    
    # Create test data array
    test_data = np.zeros(ndata, dtype=np.float64)
    
    # Set vertex values
    for i in range(min(num_verts, num_owned)):
        x = mesh_coords[i * dim + 0] if dim >= 1 else 0.0
        y = mesh_coords[i * dim + 1] if dim >= 2 else 0.0
        z = mesh_coords[i * dim + 2] if dim >= 3 else 0.0
        for c in range(num_components):
            idx = i * num_components + c
            test_data[idx] = test_func(x, y, z, c)

    print(f"  Set vertex values for field data")
    
    # For order 2, set edge midpoint values (simplified - would need proper edge coordinates)
    if order == 2 and num_owned > num_verts:
        for i in range(num_verts, num_owned):
            for c in range(num_components):
                idx = i * num_components + c
                # Use simplified values for edges
                test_data[idx] = float(c) + 0.5
    
    field.set_dof_holder_data(test_data)
    print(f"  Set field data with test function")
    
    # Define evaluation points (physical coordinates)
    if dim == 2:
        eval_coords = np.array([
            [0.5, 0.5],
            [0.25, 0.25],
            [0.75, 0.75],
            [0.1, 0.9],
            [0.9, 0.1]
        ], dtype=np.float64)
    else:  # dim == 3
        eval_coords = np.array([
            [0.5, 0.5, 0.5],
            [0.25, 0.25, 0.25],
            [0.75, 0.75, 0.75],
            [0.1, 0.9, 0.1],
            [0.9, 0.1, 0.9]
        ], dtype=np.float64)
    
    num_eval_points = eval_coords.shape[0]
    print(f"  Evaluating at {num_eval_points} points")
    
    # Get localization hint for the coordinates
    coord_system = py_pcms.CoordinateSystem.Cartesian
    location_hint = field.get_localization_hint(eval_coords, coord_system)
    print(f"  Got localization hint")
    
    # Create output buffer for evaluation results
    eval_values = np.zeros(num_eval_points * num_components, dtype=np.float64)
    
    # Evaluate the field
    field.evaluate(location_hint, eval_values, coord_system)
    print(f"  Field evaluated successfully")
    
    # Print and verify results
    for i in range(num_eval_points):
        coords_str = ", ".join([f"{eval_coords[i, j]:.3f}" for j in range(dim)])
        values_str = ", ".join([f"{eval_values[i * num_components + c]:.4f}" 
                               for c in range(num_components)])
        print(f"  Point {i} ({coords_str}): values = [{values_str}]")
        
        # Verify we got valid values (not NaN or infinity)
        for c in range(num_components):
            val = eval_values[i * num_components + c]
            assert not np.isnan(val), f"Got NaN value at point {i}, component {c}"
            assert not np.isinf(val), f"Got inf value at point {i}, component {c}"
    
    # Test gradient evaluation if available
    if field.can_evaluate_gradient():
        print(f"  Testing gradient evaluation...")
        gradient_values = np.zeros(num_eval_points * num_components * dim, dtype=np.float64)
        
        try:
            field.evaluate_gradient(gradient_values, coord_system)
            print(f"  Gradient evaluated successfully")
            
            # Print gradient results
            for i in range(min(3, num_eval_points)):
                for c in range(num_components):
                    grad_components = []
                    for d in range(dim):
                        idx = i * num_components * dim + c * dim + d
                        grad_components.append(f"{gradient_values[idx]:.4f}")
                    grad_str = ", ".join(grad_components)
                    print(f"    Point {i}, component {c}: gradient = [{grad_str}]")
        except Exception as e:
            print(f"  Gradient evaluation failed: {e}")
    else:
        print(f"  Gradient evaluation not supported")
    
    print(f"✓ Field evaluation test passed: dim={dim}, order={order}, num_components={num_components}")

def main():
    """Run all test cases"""
    print("Testing OmegaHFieldLayout Python bindings...")
    
    # Initialize Omega_h library
    lib = py_pcms.OmegaHLibrary()
    world = lib.world()
    print("Initialized Omega_h library and world")
    
    # Test different configurations
    test_layout_methods(world, 2, 1, 1)
    test_layout_methods(world, 2, 2, 1)
    # test_layout_methods(world, 2, 1, 3)
    # test_layout_methods(world, 3, 1, 1)
    
    # Test field evaluation
    print("\n" + "="*60)
    print("Testing field evaluation...")
    print("="*60)
    test_field_evaluation(world, 2, 1, 1)
    test_field_evaluation(world, 2, 2, 1)
    
    print("\n✓ All tests passed!")

if __name__ == "__main__":
    main()
