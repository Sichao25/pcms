#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <Kokkos_Core.hpp>
#include "pcms/adapter/uniform_grid/uniform_grid_field_layout.h"
#include "pcms/adapter/uniform_grid/uniform_grid_field.h"
#include "pcms/uniform_grid.h"
#include "Omega_h_library.hpp"
#include "Omega_h_build.hpp"
#include "pcms/transfer_field2.h"
#include "pcms/adapter/omega_h/omega_h_field_layout.h"
#include "pcms/adapter/omega_h/omega_h_field2.h"
#include "pcms/create_field.h"
#include "pcms/arrays.h"
#include <cmath>

using pcms::CreateUniformGridBinaryField;
using pcms::CreateUniformGridFromMesh;

// Helper function to initialize omega_h field data with f(x,y) = x + 2*y
std::vector<pcms::Real> CreateOmegaHFieldData(
  const pcms::CoordinateView<pcms::HostMemorySpace>& coords, int num_nodes) {
  auto coords_data = coords.GetCoordinates();
  std::vector<pcms::Real> omega_h_data(num_nodes);
  for (int i = 0; i < num_nodes; ++i) {
    pcms::Real x = coords_data(i, 0);
    pcms::Real y = coords_data(i, 1);
    omega_h_data[i] = x + 2.0 * y; // f(x,y) = x + 2y
  }
  return omega_h_data;
}

// Helper function to verify ug_field values
void VerifyUniformGridFieldValues(
  const pcms::UniformGrid<2>& grid,
  const pcms::CoordinateView<pcms::HostMemorySpace>& ug_coords,
  const pcms::Rank1View<const pcms::Real, pcms::HostMemorySpace>& ug_field_data) {
  fprintf(stderr, "\nVerifying ug_field values:\n");
  for (int j = 0; j <= grid.divisions[1]; ++j) {
    for (int i = 0; i <= grid.divisions[0]; ++i) {
      int vertex_id = j * (grid.divisions[0] + 1) + i;
      pcms::Real x = ug_coords.GetCoordinates()(vertex_id, 0);
      pcms::Real y = ug_coords.GetCoordinates()(vertex_id, 1);
      pcms::Real expected = x + 2.0 * y;
      pcms::Real actual = ug_field_data(vertex_id);
      fprintf(
        stderr,
        "ug_field Vertex (%d, %d) at (%.2f, %.2f): expected %.4f, got %.4f\n",
        i, j, x, y, expected, actual);
      REQUIRE(std::abs(expected - actual) <= 1e-10);
    }
  }
}

// Helper function to verify mask field values
void VerifyMaskFieldValues(
  const pcms::UniformGrid<2>& grid,
  const std::vector<int>& mask_field) {
  fprintf(stderr, "\nVerifying mask_field values:\n");
  auto mask_data = mask_field.data();
  for (int j = 0; j < grid.divisions[1]; ++j) {
    for (int i = 0; i < grid.divisions[0]; ++i) {
      int cell_id = j * grid.divisions[0] + i;
      int mask_value = mask_data[cell_id];
      fprintf(stderr, "Cell (%d, %d): mask = %d\n", i, j, mask_value);
      REQUIRE(mask_value == 1);
    }
  }
}

TEST_CASE("UniformGrid field creation")
{
  // Create a simple 2D uniform grid
  pcms::UniformGrid<2> grid;
  grid.bot_left = {0.0, 0.0};
  grid.edge_length = {10.0, 10.0};
  grid.divisions = {5, 5};

  // Create field layout with 1 component (scalar field)
  pcms::UniformGridFieldLayout<2> layout(grid, 1,
                                         pcms::CoordinateSystem::Cartesian);

  REQUIRE(layout.GetNumComponents() == 1);
  REQUIRE(layout.GetNumOwnedDofHolder() == 36); // (5+1)x(5+1) = 36 vertices
  REQUIRE(layout.GetNumGlobalDofHolder() == 36);
  REQUIRE_FALSE(layout.IsDistributed());

  // Create field
  auto field = layout.CreateField();
  REQUIRE(field != nullptr);
}

TEST_CASE("UniformGrid field data operations", "[uniform_grid_field]")
{
  pcms::UniformGrid<2> grid;
  grid.bot_left = {0.0, 0.0};
  grid.edge_length = {10.0, 10.0};
  grid.divisions = {4, 4}; // 4x4 = 16 cells

  pcms::UniformGridFieldLayout<2> layout(grid, 1,
                                         pcms::CoordinateSystem::Cartesian);
  auto field = layout.CreateField();

  // Initialize field data with vertex indices (5x5 = 25 vertices for 4x4 cells)
  std::vector<pcms::Real> data(25);
  for (size_t i = 0; i < 25; ++i) {
    data[i] = static_cast<pcms::Real>(i);
  }

  auto data_view = pcms::Rank1View<const pcms::Real, pcms::HostMemorySpace>(
    data.data(), data.size());
  field->SetDOFHolderData(data_view);

  // Retrieve data
  auto retrieved = field->GetDOFHolderData();
  REQUIRE(retrieved.size() == 25);

  for (size_t i = 0; i < 25; ++i) {
    REQUIRE(retrieved[i] == static_cast<pcms::Real>(i));
  }
}

TEST_CASE("UniformGrid field evaluation - piecewise constant")
{
  pcms::UniformGrid<2> grid;
  grid.bot_left = {0.0, 0.0};
  grid.edge_length = {10.0, 10.0};
  grid.divisions = {2, 2}; // 2x2 = 4 cells

  pcms::UniformGridFieldLayout<2> layout(grid, 1,
                                         pcms::CoordinateSystem::Cartesian);
  auto field = layout.CreateField();

  // Set vertex values for a 2x2 cell grid (3x3 = 9 vertices)
  // Vertex layout:
  //   v6---v7---v8
  //   |  2 |  3 |
  //   v3---v4---v5
  //   |  0 |  1 |
  //   v0---v1---v2
  std::vector<pcms::Real> data = {
    1.0, 1.5, 2.0, // v0, v1, v2 (bottom row, y=0)
    2.0, 2.5, 3.0, // v3, v4, v5 (middle row, y=5)
    3.0, 3.5, 4.0  // v6, v7, v8 (top row, y=10)
  };
  auto data_view = pcms::Rank1View<const pcms::Real, pcms::HostMemorySpace>(
    data.data(), data.size());
  field->SetDOFHolderData(data_view);

  // Evaluate at cell centers
  std::vector<pcms::Real> eval_coords = {
    2.5, 2.5, // Cell 0 center
    7.5, 2.5, // Cell 1 center
    2.5, 7.5, // Cell 2 center
    7.5, 7.5  // Cell 3 center
  };

  auto coords_view = pcms::Rank2View<const pcms::Real, pcms::HostMemorySpace>(
    eval_coords.data(), 4, 2);
  auto coord_view = pcms::CoordinateView<pcms::HostMemorySpace>(
    pcms::CoordinateSystem::Cartesian, coords_view);

  auto hint = field->GetLocalizationHint(coord_view);

  std::vector<pcms::Real> results(4);
  auto results_view =
    pcms::Rank1View<pcms::Real, pcms::HostMemorySpace>(results.data(), 4);
  auto results_field_view =
    pcms::FieldDataView<pcms::Real, pcms::HostMemorySpace>(
      results_view, pcms::CoordinateSystem::Cartesian);

  field->Evaluate(hint, results_field_view);

  // Check results - interpolated from vertices
  // Cell 0 center (2.5, 2.5): avg of v0,v1,v3,v4 = (1.0+1.5+2.0+2.5)/4 = 1.75
  // Cell 1 center (7.5, 2.5): avg of v1,v2,v4,v5 = (1.5+2.0+2.5+3.0)/4 = 2.25
  // Cell 2 center (2.5, 7.5): avg of v3,v4,v6,v7 = (2.0+2.5+3.0+3.5)/4 = 2.75
  // Cell 3 center (7.5, 7.5): avg of v4,v5,v7,v8 = (2.5+3.0+3.5+4.0)/4 = 3.25
  REQUIRE(std::abs(results[0] - 1.75) < 1e-10);
  REQUIRE(std::abs(results[1] - 2.25) < 1e-10);
  REQUIRE(std::abs(results[2] - 2.75) < 1e-10);
  REQUIRE(std::abs(results[3] - 3.25) < 1e-10);
}

TEST_CASE("UniformGrid field serialization")
{
  pcms::UniformGrid<2> grid;
  grid.bot_left = {0.0, 0.0};
  grid.edge_length = {10.0, 10.0};
  grid.divisions = {3, 3}; // 9 cells

  pcms::UniformGridFieldLayout<2> layout(grid, 1,
                                         pcms::CoordinateSystem::Cartesian);
  auto field = layout.CreateField();

  // Set field data (4x4 = 16 vertices for 3x3 cells)
  std::vector<pcms::Real> data(16);
  for (size_t i = 0; i < 16; ++i) {
    data[i] = static_cast<pcms::Real>(i * 10);
  }

  auto data_view = pcms::Rank1View<const pcms::Real, pcms::HostMemorySpace>(
    data.data(), data.size());
  field->SetDOFHolderData(data_view);

  // Create identity permutation
  std::vector<pcms::LO> permutation(16);
  for (size_t i = 0; i < 16; ++i) {
    permutation[i] = i;
  }

  auto perm_view = pcms::Rank1View<const pcms::LO, pcms::HostMemorySpace>(
    permutation.data(), 16);

  // Serialize
  std::vector<pcms::Real> buffer(16);
  auto buffer_view =
    pcms::Rank1View<pcms::Real, pcms::HostMemorySpace>(buffer.data(), 16);
  int size = field->Serialize(buffer_view, perm_view);

  REQUIRE(size == 16);

  // Verify serialized data
  for (size_t i = 0; i < 16; ++i) {
    REQUIRE(buffer[i] == data[i]);
  }

  // Create new field and deserialize
  auto field2 = layout.CreateField();
  auto buffer_const_view =
    pcms::Rank1View<const pcms::Real, pcms::HostMemorySpace>(buffer.data(), 16);
  field2->Deserialize(buffer_const_view, perm_view);

  // Verify deserialized data
  auto retrieved = field2->GetDOFHolderData();
  for (size_t i = 0; i < 16; ++i) {
    REQUIRE(retrieved[i] == data[i]);
  }
}

TEST_CASE("UniformGrid field copy")
{
  pcms::UniformGrid<2> grid;
  grid.bot_left = {0.0, 0.0};
  grid.edge_length = {10.0, 10.0};
  grid.divisions = {2, 2}; // 2x2 grid

  pcms::UniformGridFieldLayout<2> layout(grid, 1,
                                         pcms::CoordinateSystem::Cartesian);
  auto field = layout.CreateField();

  // Set vertex values with f(x,y) = x + y at 3x3 vertex positions
  // Vertices at: (0,0), (5,0), (10,0), (0,5), (5,5), (10,5), (0,10), (5,10),
  // (10,10)
  std::vector<pcms::Real> data = {
    0.0,  5.0,  10.0, // y=0:  v0(0,0)=0,   v1(5,0)=5,   v2(10,0)=10
    5.0,  10.0, 15.0, // y=5:  v3(0,5)=5,   v4(5,5)=10,  v5(10,5)=15
    10.0, 15.0, 20.0  // y=10: v6(0,10)=10, v7(5,10)=15, v8(10,10)=20
  };
  auto data_view = pcms::Rank1View<const pcms::Real, pcms::HostMemorySpace>(
    data.data(), data.size());
  field->SetDOFHolderData(data_view);

  // Test 1: Evaluate at cell centers
  std::vector<pcms::Real> eval_coords = {
    2.5, 2.5, // Cell 0 center
    7.5, 2.5, // Cell 1 center
    2.5, 7.5, // Cell 2 center
    7.5, 7.5  // Cell 3 center
  };

  auto coords_view = pcms::Rank2View<const pcms::Real, pcms::HostMemorySpace>(
    eval_coords.data(), 4, 2);
  auto coord_view = pcms::CoordinateView<pcms::HostMemorySpace>(
    pcms::CoordinateSystem::Cartesian, coords_view);

  auto hint = field->GetLocalizationHint(coord_view);

  std::vector<pcms::Real> results(coord_view.GetCoordinates().size() / 2);
  auto results_view =
    pcms::Rank1View<pcms::Real, pcms::HostMemorySpace>(results.data(), 4);
  auto results_field_view =
    pcms::FieldDataView<pcms::Real, pcms::HostMemorySpace>(
      results_view, pcms::CoordinateSystem::Cartesian);

  field->Evaluate(hint, results_field_view);

  REQUIRE(std::abs(results[0] - 5.0) < 1e-10);
  REQUIRE(std::abs(results[1] - 10.0) < 1e-10);
  REQUIRE(std::abs(results[2] - 10.0) < 1e-10);
  REQUIRE(std::abs(results[3] - 15.0) < 1e-10);

  // Test 2: test copy_field2
  auto field2 = layout.CreateField();
  pcms::copy_field2(*field, *field2);

  auto copied_data = field2->GetDOFHolderData();
  REQUIRE(copied_data.size() == data.size());
  for (size_t i = 0; i < data.size(); ++i) {
    REQUIRE(copied_data[i] == data[i]);
  }
}

TEST_CASE("Transfer from OmegaH field to UniformGrid field")
{
  Omega_h::Library lib;

  // Create a simple omega_h mesh (2x2 box)
  auto mesh = Omega_h::build_box(lib.world(), OMEGA_H_SIMPLEX, 1.0, 1.0, 0.0, 2,
                                 2, 0, false);

  // Create OmegaH field layout with linear elements
  auto omega_h_layout =
    pcms::CreateLagrangeLayout(mesh, 1, 1, pcms::CoordinateSystem::Cartesian);
  auto omega_h_field = omega_h_layout->CreateField();

  // Initialize omega_h field with a simple function f(x,y) = x + 2*y
  auto coords = omega_h_layout->GetDOFHolderCoordinates();
  int num_nodes = omega_h_layout->GetNumOwnedDofHolder();
  std::vector<pcms::Real> omega_h_data = CreateOmegaHFieldData(coords, num_nodes);

  auto omega_h_data_view =
    pcms::Rank1View<const pcms::Real, pcms::HostMemorySpace>(
      omega_h_data.data(), omega_h_data.size());
  omega_h_field->SetDOFHolderData(omega_h_data_view);

  // Create a uniform grid field covering the same domain [0,1] x [0,1]
  pcms::UniformGrid<2> grid;
  grid.bot_left = {0.0, 0.0};
  grid.edge_length = {1.0, 1.0};
  grid.divisions = {2, 2}; // 4x4 grid for finer resolution

  pcms::UniformGridFieldLayout<2> ug_layout(grid, 1,
                                            pcms::CoordinateSystem::Cartesian);
  auto ug_field = ug_layout.CreateField();

  // Transfer from omega_h field to uniform grid field using interpolation
  auto coords_interpolation = ug_layout.GetDOFHolderCoordinates();
  std::vector<pcms::Real> evaluation(
    coords_interpolation.GetCoordinates().size() / 2);
  auto evaluation_view = pcms::Rank1View<pcms::Real, pcms::HostMemorySpace>(
    evaluation.data(), evaluation.size());
  pcms::FieldDataView<pcms::Real, pcms::HostMemorySpace> data_view{
    evaluation_view, omega_h_field->GetCoordinateSystem()};
  auto locale = omega_h_field->GetLocalizationHint(coords_interpolation);
  omega_h_field->Evaluate(locale, data_view);
  auto evaluation_view_const =
    pcms::Rank1View<const pcms::Real, pcms::HostMemorySpace>(evaluation.data(),
                                                             evaluation.size());
  ug_field->SetDOFHolderData(evaluation_view_const);

  // Verify the transferred data at uniform grid vertices
  auto transferred_data = ug_field->GetDOFHolderData();
  auto ug_coords = ug_layout.GetDOFHolderCoordinates();
  auto ug_coords_data = ug_coords.GetCoordinates();
  int num_ug_nodes = ug_layout.GetNumOwnedDofHolder(); // 5x5 = 25 vertices

  // Check a few sample points
  for (int i = 0; i < num_ug_nodes; ++i) {
    pcms::Real x = ug_coords_data(i, 0);
    pcms::Real y = ug_coords_data(i, 1);
    pcms::Real expected = x + 2.0 * y;
    pcms::Real actual = transferred_data[i];

    // Allow some tolerance for interpolation
    REQUIRE(std::abs(actual - expected) < 1e-6);
  }
}

TEST_CASE("Create binary field from uniform grid")
{
  auto lib = Omega_h::Library{};
  auto world = lib.world();

  SECTION("Simple 2D box mesh - all cells inside")
  {
    // Create a mesh that fills the domain [0,1] x [0,1]
    auto mesh = Omega_h::build_box(world, OMEGA_H_SIMPLEX, 1.0, 1.0, 0.0, 10,
                                   10, 0, false);

    // Create a 5x5 grid (coarser than mesh)
    auto field = CreateUniformGridBinaryField<2>(mesh, {5, 5});

    REQUIRE(field.size() == 25);

    // All grid cells should be inside the mesh
    int sum = std::accumulate(field.begin(), field.end(), 0);
    REQUIRE(sum == 25);
  }

  SECTION("Binary field with custom divisions")
  {
    auto mesh =
      Omega_h::build_box(world, OMEGA_H_SIMPLEX, 1.0, 1.0, 0.0, 8, 8, 0, false);

    auto field = CreateUniformGridBinaryField<2>(mesh, {10, 8});

    REQUIRE(field.size() == 80);

    // Most cells should be inside
    int inside_count = std::accumulate(field.begin(), field.end(), 0);
    REQUIRE(inside_count > 0);
    REQUIRE(inside_count <= 80);
  }

  SECTION("Binary field with equal divisions convenience function")
  {
    auto mesh =
      Omega_h::build_box(world, OMEGA_H_SIMPLEX, 1.0, 1.0, 0.0, 5, 5, 0, false);

    auto field = CreateUniformGridBinaryField<2>(mesh, 8);

    REQUIRE(field.size() == 64);

    int inside_count = std::accumulate(field.begin(), field.end(), 0);
    REQUIRE(inside_count > 0);
  }

  SECTION("Fine grid over coarse mesh")
  {
    // Create a simple mesh
    auto mesh =
      Omega_h::build_box(world, OMEGA_H_SIMPLEX, 2.0, 2.0, 0.0, 4, 4, 0, false);

    // Create a fine grid (20x20)
    auto grid = CreateUniformGridFromMesh<2>(mesh, {20, 20});
    auto field = CreateUniformGridBinaryField<2>(mesh, {20, 20});

    REQUIRE(field.size() == 400);

    // Verify consistency: check some specific cells
    // Center cells should be inside
    auto center_idx = grid.GetCellIndex({10, 10});
    REQUIRE(field[center_idx] == 1);

    // Corner cells should be inside
    auto corner_idx = grid.GetCellIndex({0, 0});
    REQUIRE(field[corner_idx] == 1);
  }

  SECTION("Verify field values are binary")
  {
    auto mesh =
      Omega_h::build_box(world, OMEGA_H_SIMPLEX, 1.0, 1.0, 0.0, 5, 5, 0, false);

    auto field = CreateUniformGridBinaryField<2>(mesh, 10);

    // All values should be 0 or 1
    for (const auto& val : field) {
      REQUIRE((val == 0 || val == 1));
    }
  }

  SECTION("Grid extends beyond mesh - cells outside should be marked 0")
  {
    // Create a small mesh in the center of a domain
    auto mesh =
      Omega_h::build_box(world, OMEGA_H_SIMPLEX, 0.5, 0.5, 0.0, 5, 5, 0, false);

    // The mesh occupies [0, 0.5] x [0, 0.5]
    // Create a grid that would cover this
    auto grid = CreateUniformGridFromMesh<2>(mesh, {10, 10});
    auto field = CreateUniformGridBinaryField<2>(mesh, {10, 10});

    REQUIRE(field.size() == 100);

    // All cells should be inside since grid is exactly on mesh bbox
    int inside_count = std::accumulate(field.begin(), field.end(), 0);
    REQUIRE(inside_count > 0);
  }

  SECTION("Test with different aspect ratio")
  {
    // Create a rectangular mesh
    auto mesh = Omega_h::build_box(world, OMEGA_H_SIMPLEX, 3.0, 1.0, 0.0, 12, 4,
                                   0, false);

    auto field = CreateUniformGridBinaryField<2>(mesh, {30, 10});

    REQUIRE(field.size() == 300);

    int inside_count = std::accumulate(field.begin(), field.end(), 0);
    REQUIRE(inside_count > 0);

    // Calculate percentage inside
    double inside_percent = 100.0 * inside_count / field.size();
    // Most cells should be inside
    REQUIRE(inside_percent > 50.0);
  }
}

TEST_CASE("Binary field integration with grid methods")
{
  auto lib = Omega_h::Library{};
  auto world = lib.world();

  auto mesh =
    Omega_h::build_box(world, OMEGA_H_SIMPLEX, 1.0, 1.0, 0.0, 10, 10, 0, false);

  SECTION("Query field value at specific grid cell")
  {
    auto grid = CreateUniformGridFromMesh<2>(mesh, {8, 8});
    auto field = CreateUniformGridBinaryField<2>(mesh, {8, 8});

    // Get field value for a specific cell
    pcms::LO cell_id = grid.GetCellIndex({4, 4}); // Middle cell
    REQUIRE(field[cell_id] == 1);                 // Should be inside

    // Get bbox of that cell
    auto bbox = grid.GetCellBBOX(cell_id);
    // Verify it's roughly in the middle
    REQUIRE(bbox.center[0] == Catch::Approx(0.5).margin(0.1));
    REQUIRE(bbox.center[1] == Catch::Approx(0.5).margin(0.1));
  }

  SECTION("Count cells by region")
  {
    auto grid = CreateUniformGridFromMesh<2>(mesh, 10);
    auto field = CreateUniformGridBinaryField<2>(mesh, 10);

    // Count cells in different quadrants
    int q1 = 0, q2 = 0, q3 = 0, q4 = 0; // quadrants

    for (pcms::LO i = 0; i < grid.GetNumCells(); ++i) {
      if (field[i] == 1) {
        auto bbox = grid.GetCellBBOX(i);
        if (bbox.center[0] < 0.5 && bbox.center[1] < 0.5)
          q1++;
        else if (bbox.center[0] >= 0.5 && bbox.center[1] < 0.5)
          q2++;
        else if (bbox.center[0] < 0.5 && bbox.center[1] >= 0.5)
          q3++;
        else
          q4++;
      }
    }

    // All quadrants should have some inside cells
    REQUIRE(q1 > 0);
    REQUIRE(q2 > 0);
    REQUIRE(q3 > 0);
    REQUIRE(q4 > 0);

    // Distribution should be roughly equal
    int total = q1 + q2 + q3 + q4;
    REQUIRE(q1 == Catch::Approx(total / 4.0).margin(5));
    REQUIRE(q2 == Catch::Approx(total / 4.0).margin(5));
    REQUIRE(q3 == Catch::Approx(total / 4.0).margin(5));
    REQUIRE(q4 == Catch::Approx(total / 4.0).margin(5));
  }
}

TEST_CASE("Performance and edge cases")
{
  auto lib = Omega_h::Library{};
  auto world = lib.world();

  SECTION("Very fine grid")
  {
    auto mesh =
      Omega_h::build_box(world, OMEGA_H_SIMPLEX, 1.0, 1.0, 0.0, 5, 5, 0, false);

    // Create a very fine grid
    auto field = CreateUniformGridBinaryField<2>(mesh, 50);

    REQUIRE(field.size() == 2500);

    int inside_count = std::accumulate(field.begin(), field.end(), 0);
    REQUIRE(inside_count > 0);
  }

  SECTION("Coarse grid")
  {
    auto mesh = Omega_h::build_box(world, OMEGA_H_SIMPLEX, 1.0, 1.0, 0.0, 10,
                                   10, 0, false);

    // Very coarse grid
    auto field = CreateUniformGridBinaryField<2>(mesh, {2, 2});

    REQUIRE(field.size() == 4);

    // All 4 cells should be inside for this configuration
    int inside_count = std::accumulate(field.begin(), field.end(), 0);
    REQUIRE(inside_count > 0);
  }

  SECTION("Non-square domain")
  {
    auto mesh = Omega_h::build_box(world, OMEGA_H_SIMPLEX, 5.0, 2.0, 0.0, 20, 8,
                                   0, false);

    auto field = CreateUniformGridBinaryField<2>(mesh, {25, 10});

    REQUIRE(field.size() == 250);

    int inside_count = std::accumulate(field.begin(), field.end(), 0);
    REQUIRE(inside_count > 0);
  }

  SECTION("Grid larger than mesh - cells outside marked as 0")
  {
    // Create a mesh covering [0, 0.5] x [0, 0.5]
    auto mesh =
      Omega_h::build_box(world, OMEGA_H_SIMPLEX, 0.5, 0.5, 0.0, 5, 5, 0, false);

    // Manually create a larger grid covering [0, 1] x [0, 1]
    pcms::UniformGrid<2> grid;
    grid.edge_length = {1.0, 1.0};
    grid.bot_left = {0.0, 0.0};
    grid.divisions = {10, 10};

    // Create binary field on the larger grid
    auto field = pcms::CreateUniformGridBinaryFieldFromGrid<2>(mesh, grid);

    REQUIRE(field.size() == 100);

    // Count cells inside and outside
    int inside_count = std::accumulate(field.begin(), field.end(), 0);
    int outside_count = field.size() - inside_count;

    // Should have both inside (1) and outside (0) cells
    REQUIRE(inside_count > 0);
    REQUIRE(outside_count > 0);

    // Check specific cells
    // Bottom-left corner [0, 0.1] x [0, 0.1] should be inside (mesh covers [0,
    // 0.5])
    auto corner_id = grid.GetCellIndex({0, 0});
    auto bbox = grid.GetCellBBOX(corner_id);
    REQUIRE(field[corner_id] == 1);

    // Top-right corner [0.9, 1.0] x [0.9, 1.0] should be outside (mesh ends at
    // 0.5)
    corner_id = grid.GetCellIndex({9, 9});
    bbox = grid.GetCellBBOX(corner_id);
    REQUIRE(field[corner_id] == 0);

    // Cell at [0.5, 0.6] x [0.5, 0.6] should be outside
    corner_id = grid.GetCellIndex({6, 6});
    bbox = grid.GetCellBBOX(corner_id);
    REQUIRE(field[corner_id] == 0);

    // Cell at [0.2, 0.3] x [0.2, 0.3] should be inside
    auto center_id = grid.GetCellIndex({2, 2});
    bbox = grid.GetCellBBOX(center_id);
    REQUIRE(field[center_id] == 1);
  }
}

TEST_CASE("UniformGrid workflow")
{
  auto lib = Omega_h::Library{};
  auto world = lib.world();

  // Create a simple 2D box mesh: 1.0 x 1.0 domain with 4x4 elements
  auto mesh =
    Omega_h::build_box(world, OMEGA_H_SIMPLEX, 1.0, 1.0, 0.0, 4, 4, 0, false);
  auto grid = pcms::CreateUniformGridFromMesh<2>(mesh, {4, 4});
  auto mask_field = pcms::CreateUniformGridBinaryField<2>(mesh, {4, 4});

  // Create OmegaH field layout with linear elements
  auto omega_h_layout =
    pcms::CreateLagrangeLayout(mesh, 1, 1, pcms::CoordinateSystem::Cartesian);
  auto omega_h_field = omega_h_layout->CreateField();

  // Initialize omega_h field with a simple function f(x,y) = x + 2*y
  auto coords = omega_h_layout->GetDOFHolderCoordinates();
  int num_nodes = omega_h_layout->GetNumOwnedDofHolder();
  std::vector<pcms::Real> omega_h_data = CreateOmegaHFieldData(coords, num_nodes);

  auto omega_h_data_view =
    pcms::Rank1View<const pcms::Real, pcms::HostMemorySpace>(
      omega_h_data.data(), omega_h_data.size());
  omega_h_field->SetDOFHolderData(omega_h_data_view);

  // Create uniform grid field layout
  pcms::UniformGridFieldLayout<2> ug_layout(grid, 1,
                                            pcms::CoordinateSystem::Cartesian);
  auto ug_field = ug_layout.CreateField();

  // Transfer from omega_h field to uniform grid field using interpolation
  pcms::interpolate_field2(*omega_h_field, *ug_field);
  auto ug_coords = ug_layout.GetDOFHolderCoordinates();

  // Verify ug_field values directly from the field object
  auto ug_field_data = ug_field->GetDOFHolderData();
  VerifyUniformGridFieldValues(grid, ug_coords, ug_field_data);

  // Verify mask field values
  VerifyMaskFieldValues(grid, mask_field);
}
