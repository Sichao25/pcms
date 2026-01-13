/**
 * @file mesh_to_uniform_grid_workflow.cpp
 * @brief Complete workflow: Transfer field data from Omega_h mesh to uniform
 * grid with mask
 *
 * This example demonstrates a complete workflow starting from:
 * 1. An Omega_h mesh with field data
 * 2. Creating a uniform grid that covers the mesh domain
 * 3. Creating a binary mask field (inside/outside mesh)
 * 4. Transferring field data from mesh to uniform grid via interpolation
 * 5. Using the mask to identify valid grid regions
 * 6. Evaluating the grid field at arbitrary points
 */

#include <pcms/create_field.h>
#include <pcms/uniform_grid.h>
#include <pcms/adapter/uniform_grid/uniform_grid_field_layout.h>
#include <pcms/adapter/omega_h/omega_h_field_layout.h>
#include <pcms/adapter/omega_h/omega_h_field2.h>
#include <pcms/arrays.h>
#include <Omega_h_build.hpp>
#include <Omega_h_library.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <numeric>

// Helper function to print field statistics
void print_field_stats(const std::string& name,
                       const std::vector<pcms::Real>& data)
{
  auto min = *std::min_element(data.begin(), data.end());
  auto max = *std::max_element(data.begin(), data.end());
  auto sum = std::accumulate(data.begin(), data.end(), 0.0);

  std::cout << name << " statistics:\n";
  std::cout << "  Size:    " << data.size() << "\n";
  std::cout << "  Min:     " << min << "\n";
  std::cout << "  Max:     " << max << "\n";
}

// Helper function to visualize mask field (2D only)
void print_mask_visualization(const pcms::UniformGrid<2>& grid,
                              const std::vector<int>& mask,
                              int sample_size = 20)
{
  // Only visualize if grid is reasonable size
  if (grid.divisions[0] > sample_size || grid.divisions[1] > sample_size) {
    std::cout << "(Grid too large for visualization, skipping...)\n";
    return;
  }

  std::cout << "\nMask field visualization (1=inside, 0=outside):\n";
  std::cout << std::string(grid.divisions[0] * 2 + 1, '-') << "\n";

  // Print from top to bottom
  for (int j = grid.divisions[1] - 1; j >= 0; --j) {
    std::cout << "|";
    for (int i = 0; i < grid.divisions[0]; ++i) {
      auto cell_id = grid.GetCellIndex({i, j});
      std::cout << mask[cell_id] << "|";
    }
    std::cout << "\n";
  }
  std::cout << std::string(grid.divisions[0] * 2 + 1, '-') << "\n";
}

int main(int argc, char** argv)
{
  // Initialize Omega_h
  auto lib = Omega_h::Library(&argc, &argv);
  auto world = lib.world();

  std::cout << "========================================================\n";
  std::cout << "Mesh-to-UniformGrid Field Transfer Workflow\n";
  std::cout << "========================================================\n\n";

  // ============================================================================
  // STEP 1: Create an Omega_h mesh
  // ============================================================================
  std::cout << "STEP 1: Creating Omega_h mesh\n";
  std::cout << "------------------------------\n";

  // Create a 2D mesh: domain [0, 2] x [0, 2] with triangular elements
  auto mesh =
    Omega_h::build_box(world, OMEGA_H_SIMPLEX, 2.0, 2.0, 0.0, 8, 8, 0, false);

  std::cout << "Created 2D mesh:\n";
  std::cout << "  Domain:   [0, 2] x [0, 2]\n";
  std::cout << "  Elements: " << mesh.nelems() << "\n";
  std::cout << "  Vertices: " << mesh.nverts() << "\n\n";

  // ============================================================================
  // STEP 2: Create an Omega_h field with mathematical data
  // ============================================================================
  std::cout << "STEP 2: Creating Omega_h field with data\n";
  std::cout << "-----------------------------------------\n";

  // Create a field layout on the mesh (P1 linear elements)
  auto omega_h_layout =
    pcms::CreateLagrangeLayout(mesh, 1, 1, pcms::CoordinateSystem::Cartesian);
  auto omega_h_field = omega_h_layout->CreateField();

  // Get mesh vertex coordinates
  auto mesh_coords = omega_h_layout->GetDOFHolderCoordinates();
  auto mesh_coords_data = mesh_coords.GetCoordinates();
  int num_mesh_nodes = omega_h_layout->GetNumOwnedDofHolder();

  // Initialize field with a mathematical function: f(x,y) = sin(πx) * cos(πy)
  std::cout << "Setting field values: f(x,y) = sin(πx) * cos(πy)\n";
  std::vector<pcms::Real> mesh_field_data(num_mesh_nodes);
  for (int i = 0; i < num_mesh_nodes; ++i) {
    pcms::Real x = mesh_coords_data(i, 0);
    pcms::Real y = mesh_coords_data(i, 1);
    mesh_field_data[i] = std::sin(M_PI * x) * std::cos(M_PI * y);
  }

  auto mesh_data_view =
    pcms::Rank1View<const pcms::Real, pcms::HostMemorySpace>(
      mesh_field_data.data(), mesh_field_data.size());
  omega_h_field->SetDOFHolderData(mesh_data_view);

  print_field_stats("Omega_h field", mesh_field_data);
  std::cout << "\n";

  // ============================================================================
  // STEP 3: Create uniform grid from mesh
  // ============================================================================
  std::cout << "STEP 3: Creating uniform grid from mesh\n";
  std::cout << "----------------------------------------\n";

  // Create a 10x10 uniform grid covering the mesh bounding box
  auto grid = pcms::CreateUniformGridFromMesh<2>(mesh, {10, 10});

  std::cout << "Created uniform grid:\n";
  std::cout << "  Number of cells:     " << grid.GetNumCells() << "\n";
  std::cout << "  Number of vertices:  "
            << (grid.divisions[0] + 1) * (grid.divisions[1] + 1) << "\n";
  std::cout << "  Domain:              [" << grid.bot_left[0] << ", "
            << grid.bot_left[0] + grid.edge_length[0] << "] x ["
            << grid.bot_left[1] << ", "
            << grid.bot_left[1] + grid.edge_length[1] << "]\n";
  std::cout << "  Cell size:           "
            << grid.edge_length[0] / grid.divisions[0] << " x "
            << grid.edge_length[1] / grid.divisions[1] << "\n\n";

  // ============================================================================
  // STEP 4: Create binary mask field (inside/outside mesh)
  // ============================================================================
  std::cout << "STEP 4: Creating binary mask field\n";
  std::cout << "-----------------------------------\n";

  auto mask_field = pcms::CreateUniformGridBinaryField<2>(mesh, {10, 10});

  int inside_count = std::accumulate(mask_field.begin(), mask_field.end(), 0);
  int outside_count = mask_field.size() - inside_count;

  std::cout << "Mask field created:\n";
  std::cout << "  Total cells:   " << mask_field.size() << "\n";
  std::cout << "  Inside mesh:   " << inside_count << " cells ("
            << 100.0 * inside_count / mask_field.size() << "%)\n";
  std::cout << "  Outside mesh:  " << outside_count << " cells ("
            << 100.0 * outside_count / mask_field.size() << "%)\n";

  print_mask_visualization(grid, mask_field);
  std::cout << "\n";

  // ============================================================================
  // STEP 5: Create uniform grid field and transfer data
  // ============================================================================
  std::cout << "STEP 5: Transferring field data to uniform grid\n";
  std::cout << "------------------------------------------------\n";

  // Create uniform grid field layout
  pcms::UniformGridFieldLayout<2> ug_layout(grid, 1,
                                            pcms::CoordinateSystem::Cartesian);
  auto ug_field = ug_layout.CreateField();

  std::cout << "Grid field layout:\n";
  std::cout << "  DOF holders (vertices): " << ug_layout.GetNumOwnedDofHolder()
            << "\n";
  std::cout << "  Components:             " << ug_layout.GetNumComponents()
            << "\n\n";

  // Get uniform grid vertex coordinates for evaluation
  auto ug_coords = ug_layout.GetDOFHolderCoordinates();

  // Evaluate omega_h field at uniform grid vertices
  std::cout << "Interpolating mesh field to grid vertices...\n";
  std::vector<pcms::Real> ug_field_data(ug_layout.GetNumOwnedDofHolder());
  auto ug_data_view = pcms::Rank1View<pcms::Real, pcms::HostMemorySpace>(
    ug_field_data.data(), ug_field_data.size());
  pcms::FieldDataView<pcms::Real, pcms::HostMemorySpace> field_data_view{
    ug_data_view, omega_h_field->GetCoordinateSystem()};

  auto localization_hint = omega_h_field->GetLocalizationHint(ug_coords);
  omega_h_field->Evaluate(localization_hint, field_data_view);

  // Set the evaluated data on the uniform grid field
  auto ug_data_const_view =
    pcms::Rank1View<const pcms::Real, pcms::HostMemorySpace>(
      ug_field_data.data(), ug_field_data.size());
  ug_field->SetDOFHolderData(ug_data_const_view);

  print_field_stats("Uniform grid field", ug_field_data);
  std::cout << "\n";

  // ============================================================================
  // STEP 6: Verify the transferred data
  // ============================================================================
  std::cout << "STEP 6: Verifying data transfer accuracy\n";
  std::cout << "-----------------------------------------\n";

  auto transferred_data = ug_field->GetDOFHolderData();
  auto ug_coords_data = ug_coords.GetCoordinates();

  // Check several sample points
  std::vector<int> sample_indices = {0, 5, 10, 50, 100, 120}; // Sample vertices
  double max_error = 0.0;

  std::cout << "Sample verification points:\n";
  std::cout << std::fixed << std::setprecision(6);
  for (int idx : sample_indices) {
    if (idx >= ug_layout.GetNumOwnedDofHolder())
      continue;

    pcms::Real x = ug_coords_data(idx, 0);
    pcms::Real y = ug_coords_data(idx, 1);
    pcms::Real expected = std::sin(M_PI * x) * std::cos(M_PI * y);
    pcms::Real actual = transferred_data[idx];
    double error = std::abs(actual - expected);
    max_error = std::max(max_error, error);

    std::cout << "  Vertex " << std::setw(3) << idx << " at (" << std::setw(8)
              << x << ", " << std::setw(8) << y
              << "): " << "expected=" << std::setw(10) << expected
              << ", actual=" << std::setw(10) << actual
              << ", error=" << std::scientific << error << std::fixed << "\n";
  }

  std::cout << "\nMaximum interpolation error: " << std::scientific << max_error
            << std::fixed << "\n\n";

  // ============================================================================
  // STEP 7: Use mask field to process only valid grid regions
  // ============================================================================
  std::cout << "STEP 7: Processing valid grid regions using mask\n";
  std::cout << "-------------------------------------------------\n";

  double sum_inside = 0.0;
  int count_inside = 0;

  for (int j = 0; j < grid.divisions[1]; ++j) {
    for (int i = 0; i < grid.divisions[0]; ++i) {
      pcms::LO cell_id = grid.GetCellIndex({i, j});

      if (mask_field[cell_id] == 1) {
        // This cell is inside the mesh - we can safely work with it
        auto cell_bbox = grid.GetCellBBOX(cell_id);

        // For demonstration, accumulate field values at cell centers
        // In practice, you would evaluate the field or perform other operations
        count_inside++;

        // Get the four vertices of this cell (bottom-left, bottom-right,
        // top-left, top-right)
        int v_bl = j * (grid.divisions[0] + 1) + i;
        int v_br = j * (grid.divisions[0] + 1) + (i + 1);
        int v_tl = (j + 1) * (grid.divisions[0] + 1) + i;
        int v_tr = (j + 1) * (grid.divisions[0] + 1) + (i + 1);

        // Average the field values at the four corners
        double cell_avg = (transferred_data[v_bl] + transferred_data[v_br] +
                           transferred_data[v_tl] + transferred_data[v_tr]) /
                          4.0;
        sum_inside += cell_avg;
      }
    }
  }

  std::cout << "Processed " << count_inside << " valid cells (inside mesh)\n";
  std::cout << "Average field value in valid region: "
            << sum_inside / count_inside << "\n\n";

  // ============================================================================
  // STEP 8: Evaluate grid field at arbitrary points
  // ============================================================================
  std::cout << "STEP 8: Evaluating grid field at arbitrary points\n";
  std::cout << "--------------------------------------------------\n";

  // Define some test points
  std::vector<pcms::Real> eval_points = {
    0.5,  0.5, // Point 1
    1.0,  1.0, // Point 2 (center)
    1.5,  1.5, // Point 3
    0.25, 1.75 // Point 4
  };

  auto eval_coords_view =
    pcms::Rank2View<const pcms::Real, pcms::HostMemorySpace>(eval_points.data(),
                                                             4, 2);
  auto eval_coord_view = pcms::CoordinateView<pcms::HostMemorySpace>(
    pcms::CoordinateSystem::Cartesian, eval_coords_view);

  auto eval_hint = ug_field->GetLocalizationHint(eval_coord_view);

  std::vector<pcms::Real> eval_results(4);
  auto eval_results_view =
    pcms::Rank1View<pcms::Real, pcms::HostMemorySpace>(eval_results.data(), 4);
  auto eval_field_view = pcms::FieldDataView<pcms::Real, pcms::HostMemorySpace>(
    eval_results_view, pcms::CoordinateSystem::Cartesian);

  ug_field->Evaluate(eval_hint, eval_field_view);

  std::cout << "Evaluation results:\n";
  for (int i = 0; i < 4; ++i) {
    pcms::Real x = eval_points[2 * i];
    pcms::Real y = eval_points[2 * i + 1];
    pcms::Real expected = std::sin(M_PI * x) * std::cos(M_PI * y);
    pcms::Real actual = eval_results[i];

    std::cout << "  Point " << i + 1 << " (" << std::setw(8) << x << ", "
              << std::setw(8) << y << "): " << "interpolated=" << std::setw(10)
              << actual << ", exact=" << std::setw(10) << expected
              << ", error=" << std::scientific << std::abs(actual - expected)
              << std::fixed << "\n";
  }

  std::cout << "\n";

  // ============================================================================
  // End
  // ============================================================================
  std::cout << "Workflow Complete!\n";

  return 0;
}
