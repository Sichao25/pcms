# Unstructured mesh field transfer to structured mesh field workflow guide

This guide demonstrates how to transfer unstructured mesh fields to uniform grid fields in PCMS Python API, including mesh creation, field interpolation, and data serialization.

## Overview

The workflow allows you to:
1. Create a structured uniform grid as the bounding box as an unstructured Omega_h mesh
2. Create binary mask fields indicating which cells are inside the mesh
3. Interpolate field data from Omega_h mesh fields to uniform grid fields
4. Serialize and deserialize field data for I/O operations

## Complete Workflow Example

### Step 1: Initialize Omega_h Library

Initialize the Omega_h library which manages parallel communication.

```python
import py_pcms
import numpy as np

# Create library and world communicator
lib = py_pcms.OmegaHLibrary()
world = lib.world()
```


---

### Step 2: Create an Omega_h Mesh

You can either create a mesh or read an existing mesh from a file.

#### Option A: Create Mesh Programmatically

```python
# Create a 2D box mesh: 1.0 x 1.0 domain with 4x4 elements
mesh = py_pcms.build_box(
    world,                          # Communicator
    py_pcms.Family.SIMPLEX,        # Element type
    1.0, 1.0, 0.0,                 # Domain size: x, y, z
    4, 4, 0,                       # Number of elements: nx, ny, nz
    False                          # Symmetric flag
)
```

#### Option B: Read Mesh from File

```python
# Auto-detect file format and read
mesh = py_pcms.read_mesh_file("my_mesh.osh", world)

# Or use format-specific readers for explicit control:

# Binary format (.osh)
mesh = py_pcms.read_mesh_binary("mesh.osh", lib)

# Gmsh format (.msh)
mesh = py_pcms.read_mesh_gmsh("mesh.msh", world)

# VTK format (.pvtu for parallel files)
mesh = py_pcms.read_mesh_parallel_vtk("mesh.pvtu", world)

# Exodus format (.exo) - if SEACAS support is enabled
# Using file handle API
exo_handle = py_pcms.exodus_open("mesh.exo", verbose=False)
num_steps = py_pcms.exodus_get_num_time_steps(exo_handle)
mesh = py_pcms.OmegaHMesh(lib)
mesh.set_comm(world)
py_pcms.read_mesh_exodus(exo_handle, mesh, verbose=False)
py_pcms.exodus_close(exo_handle)

# ADIOS2 format (.bp) - if ADIOS2 support is enabled
mesh = py_pcms.read_mesh_adios2("mesh.bp", lib, prefix="")

# MESHB format (.mesh) - if LIBMESHB support is enabled
mesh = py_pcms.OmegaHMesh(lib)
mesh.set_comm(world)
py_pcms.read_mesh_meshb(mesh, "mesh.mesh")
# Read solution/resolution data if available
py_pcms.read_meshb_sol(mesh, "mesh.sol", sol_name="resolution")
```

**PS**: File formats such as Exodus or libMeshB require that PCMS be built with the respective libraries enabled.

Checking Mesh Properties:

```python
print(f"Mesh dimension: {mesh.dim()}")
print(f"Number of vertices: {mesh.nverts()}")
print(f"Number of elements: {mesh.nelems()}")
print(f"Mesh family: {mesh.family()}")
```

---

### Step 3: Create Uniform Grid from Mesh

Generate a structured uniform grid that covers the bounding box of the mesh.

```python
# Create uniform grid with 4x4 cell divisions
grid = py_pcms.create_uniform_grid_from_mesh(mesh, [4, 4])
print(f"Created uniform grid with {grid.get_num_cells()} cells")
```

---

### Step 4: Create Binary Mask Field

Create a mask where each uniform grid vertex is marked as 1 (inside mesh) or 0 (outside mesh).

```python
# Create binary mask indicating which vertices are inside the mesh
# Returns tuple of (layout, field) - layout must be kept alive while using field
mask_layout, mask_field = py_pcms.create_uniform_grid_binary_field(mesh, [4, 4])
print(f"Created mask field with {mask_layout.get_num_vertices()} vertices")
```

**Accessing Values**:
```python
# Access mask value for vertex (i, j)
vertex_id = j * (grid.divisions[0] + 1) + i
mask_data = mask_field.get_dof_holder_data()
mask_value = mask_data[vertex_id]  # Returns 0.0 or 1.0
```

---

### Step 5: Create and Initialize Omega_h Field

Create a field on the unstructured mesh and initialize it with data.

```python
# Create field layout with linear (order=1) elements and 1 component (scalar)
omega_h_layout = py_pcms.create_lagrange_layout(
    mesh, 
    1,
    1,
    py_pcms.CoordinateSystem.Cartesian
)
omega_h_field = omega_h_layout.create_field()

# Initialize field with f(x,y) = x + 2*y
coords = omega_h_layout.get_dof_holder_coordinates()
num_nodes = omega_h_layout.get_num_owned_dof_holder()
omega_h_data = np.zeros(num_nodes)

for i in range(num_nodes):
    x = coords[i, 0]
    y = coords[i, 1]
    omega_h_data[i] = x + 2.0 * y

omega_h_field.set_dof_holder_data(omega_h_data)
```

---

### Step 6: Create Uniform Grid Field Layout

Set up the data structure for storing field values on the uniform grid vertices.

```python
# Create uniform grid field layout
ug_layout = py_pcms.UniformGridFieldLayout2D(
    grid, 
    1,                                      # Number of components
    py_pcms.CoordinateSystem.Cartesian
)
ug_field = ug_layout.create_field()
```

---

### Step 7: Interpolate Field Data

Interpolate field values from the unstructured mesh to the structured uniform grid vertices.

```python
# Transfer field from Omega_h mesh to uniform grid
py_pcms.interpolate_field(omega_h_field, ug_field)
```

---

### Step 8: Access Field Data

Retrieve interpolated values and verify correctness.

```python
# Get field data and coordinates
ug_field_data = ug_field.get_dof_holder_data()
ug_coords = ug_layout.get_dof_holder_coordinates()

# Access data at vertex (i, j)
vertex_id = j * (grid.divisions[0] + 1) + i
value = ug_field_data[vertex_id]
x, y = ug_coords[vertex_id, 0], ug_coords[vertex_id, 1]
```

---

### Step 9: Export Field Data with `to_mdspan`

Convert field data to a structured $x \times y$ (or $x \times y \times z$) array
for downstream analyses.

```python
# Convert field data to a structured array
grid_values = ug_field.to_mdspan()  # 2D: (nx+1, ny+1), 3D: (nx+1, ny+1, nz+1)
mask_values = mask_field.to_mdspan()  # 2D: (nx+1, ny+1), 3D: (nx+1, ny+1, nz+1)

# Save to file (example)
np.save('field_data.npy', grid_values)
```

---

## API Reference Summary

### Grid Creation
- `create_uniform_grid_from_mesh(mesh, divisions)` - Create grid from mesh
- `create_uniform_grid_binary_field(mesh, divisions)` - Create inside/outside mask

### Field Layout
- `create_lagrange_layout(mesh, order, num_components, coord_system)` - Omega_h field layout
- `UniformGridFieldLayout2D(grid, num_components, coord_system)` - 2D grid layout
- `UniformGridFieldLayout3D(grid, num_components, coord_system)` - 3D grid layout

### Field Operations
- `layout.create_field()` - Create field from layout
- `field.set_dof_holder_data(data)` - Set field values
- `field.get_dof_holder_data()` - Get field values
- `field.to_mdspan()` - Get field values as a structured array
- `interpolate_field(source_field, target_field)` - Interpolate between fields

### Exodus I/O (if OMEGA_H_USE_SEACASEXODUS enabled)
- `exodus_open(filepath, verbose=False)` - Open Exodus file, returns file handle
- `exodus_close(exodus_file)` - Close Exodus file handle
- `exodus_get_num_time_steps(exodus_file)` - Get number of time steps in file
- `read_mesh_exodus(exodus_file, mesh, verbose=False, classify_with=...)` - Read mesh from file handle
- `read_exodus_nodal_fields(exodus_file, mesh, time_step, prefix="", postfix="", verbose=False)` - Read nodal fields
- `read_exodus_element_fields(exodus_file, mesh, time_step, prefix="", postfix="", verbose=False)` - Read element fields
- `read_mesh_exodus_sliced(filepath, comm, verbose=False, classify_with=..., time_step=-1)` - Read mesh in parallel
- `write_mesh_exodus(filepath, mesh, verbose=False, classify_with=...)` - Write mesh to Exodus file
- `ExodusClassifyWith.NODE_SETS` - Classify using node sets
- `ExodusClassifyWith.SIDE_SETS` - Classify using side sets

### MESHB I/O (if OMEGA_H_USE_LIBMESHB enabled)
- `read_mesh_meshb(mesh, filepath)` - Read mesh from MESHB file
- `write_mesh_meshb(mesh, filepath, version)` - Write mesh to MESHB file
- `read_meshb_sol(mesh, filepath, sol_name)` - Read solution/resolution from .sol file
- `write_meshb_sol(mesh, filepath, sol_name, version)` - Write solution/resolution to .sol file

### Grid Properties
- `grid.get_num_cells()` - Total number of cells
- `grid.divisions` - Cell divisions in each dimension
- `layout.get_num_vertices()` - Total number of vertices
- `layout.get_dof_holder_coordinates()` - Vertex coordinates

---

For more examples and advanced usage, see the test files in the `pcms/pythonapi/` directory.
