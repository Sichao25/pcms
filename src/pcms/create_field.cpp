#include "create_field.h"
#include "adapter/omega_h/omega_h_field2.h"
#include "adapter/omega_h/omega_h_field_layout.h"
#include "uniform_grid.h"
#include "point_search.h"

#include <Kokkos_Core.hpp>
#include <utility>

namespace pcms
{

std::unique_ptr<FieldLayout> CreateLagrangeLayout(
  Omega_h::Mesh& mesh, int order, int num_components,
  CoordinateSystem coordinate_system)
{

  std::array<int, 4> nodes_per_dim;

  switch (order) {
    case 1: nodes_per_dim = {1, 0, 0, 0}; break;
    case 2: nodes_per_dim = {1, 1, 0, 0}; break;
    default: throw std::runtime_error("Unimplemented order");
  }

  return std::make_unique<OmegaHFieldLayout>(mesh, nodes_per_dim,
                                             num_components, coordinate_system);
}

template <>
std::vector<int> CreateUniformGridBinaryFieldFromGrid<2>(
  Omega_h::Mesh& mesh, const UniformGrid<2>& grid)
{
  constexpr unsigned dim = 2;

  // Get total number of cells
  const LO num_cells = grid.GetNumCells();

  // Initialize result vector
  std::vector<int> binary_field(num_cells, 0);

  // Create GridPointSearch for point-in-mesh queries
  GridPointSearch point_search(mesh, grid.divisions[0], grid.divisions[1]);

  // Create array of grid cell center points
  Kokkos::View<Real* [dim]> cell_centers("cell centers", num_cells);
  auto cell_centers_h = Kokkos::create_mirror_view(cell_centers);

  // Fill cell center coordinates
  for (LO i = 0; i < num_cells; ++i) {
    auto bbox = grid.GetCellBBOX(i);
    for (unsigned d = 0; d < dim; ++d) {
      cell_centers_h(i, d) = bbox.center[d];
    }
  }

  // Copy to device
  Kokkos::deep_copy(cell_centers, cell_centers_h);

  // Perform point search
  auto results = point_search(cell_centers);

  // Copy results back to host
  auto results_h = Kokkos::create_mirror_view(results);
  Kokkos::deep_copy(results_h, results);

  // Process results: set field to 1 if point is inside mesh (tri_id >= 0)
  for (LO i = 0; i < num_cells; ++i) {
    binary_field[i] = (results_h(i).tri_id >= 0) ? 1 : 0;
  }

  return binary_field;
}

template <>
std::vector<int> CreateUniformGridBinaryField<2>(
  Omega_h::Mesh& mesh, const std::array<LO, 2>& divisions)
{
  constexpr unsigned dim = 2;

  // Create the uniform grid from the mesh
  auto grid = CreateUniformGridFromMesh<dim>(mesh, divisions);

  // Delegate to the grid-based implementation
  return CreateUniformGridBinaryFieldFromGrid<2>(mesh, grid);
}

template <>
std::vector<int> CreateUniformGridBinaryField<2>(Omega_h::Mesh& mesh,
                                                 LO cells_per_dim)
{
  std::array<LO, 2> divisions;
  divisions.fill(cells_per_dim);
  return CreateUniformGridBinaryField<2>(mesh, divisions);
}

} // namespace pcms
