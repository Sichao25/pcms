#include <catch2/catch_test_macros.hpp>
#include <Omega_h_build.hpp>
#include <Omega_h_mesh.hpp>
#include <stdexcept>

#include "pcms/field/function_space/lagrange.h"
#include "pcms/field/function_space/nodal.h"
#include "pcms/field/function_space/spline.h"
#include "pcms/field/field_data.h"
#include "pcms/field/point_evaluator.h"
#include "pcms/field/out_of_bounds_policy.h"
#include "pcms/field/field_metadata.h"
#include "pcms/field/coordinate_system.h"
#include "pcms/utility/arrays.h"
#include "field_test_utils.h"

using pcms::Real;
using pcms::CoordinateSystem;

// ============================================================================
// OmegaH order-1 — basic evaluation via new API
// ============================================================================

TEST_CASE("PointEvaluator: OmegaH order-1 linear evaluation")
{
  auto lib  = Omega_h::Library{};
  auto mesh = Omega_h::build_box(
    lib.world(), OMEGA_H_SIMPLEX, 1, 1, 0, 100, 100, 0, false);

  auto factory =
    pcms::LagrangeFunctionSpace::FromMesh(mesh, 1, 1, CoordinateSystem::Cartesian,
                                         "global", pcms::LagrangeFunctionSpace::Backend::OmegaH);

  auto field_data = factory.CreateField<Real>();
  pcms::test::SetField(field_data.GetData(), *factory.GetLayout(), pcms::test::linear_f);

  auto pts = pcms::test::StandardEvalCoords2D();
  int n    = static_cast<int>(pts.size()) / 2;
  pcms::Rank2View<const Real, pcms::HostMemorySpace> coords_view(
    pts.data(), n, 2);
  pcms::CoordinateView<pcms::HostMemorySpace> cv{
    CoordinateSystem::Cartesian, coords_view};
  auto evaluator = factory.CreatePointEvaluator<Real>(cv);
  pcms::test::CheckEvaluation(
    *evaluator, field_data,
    pts, pcms::test::linear_f);
}

// ============================================================================
// OmegaH order-1 — repeated evaluation: same PointEvaluator, two FieldDatas
// ============================================================================

TEST_CASE("PointEvaluator: same evaluator reused for two FieldData objects")
{
  auto lib  = Omega_h::Library{};
  auto mesh = Omega_h::build_box(
    lib.world(), OMEGA_H_SIMPLEX, 1, 1, 0, 50, 50, 0, false);

  auto factory =
    pcms::LagrangeFunctionSpace::FromMesh(mesh, 1, 1, CoordinateSystem::Cartesian,
                                         "global", pcms::LagrangeFunctionSpace::Backend::OmegaH);

  auto field_a = factory.CreateField<Real>();
  auto field_b = factory.CreateField<Real>();

  // field_a: linear_f;  field_b: constant 42
  pcms::test::SetField(field_a.GetData(), *factory.GetLayout(), pcms::test::linear_f);
  pcms::test::SetField(field_b.GetData(), *factory.GetLayout(),
                       [](Real, Real) { return Real(42); });

  auto pts = pcms::test::StandardEvalCoords2D();
  int n    = static_cast<int>(pts.size()) / 2;

  pcms::Rank2View<const Real, pcms::HostMemorySpace> coords_view(
    pts.data(), n, 2);
  pcms::CoordinateView<pcms::HostMemorySpace> cv{
    CoordinateSystem::Cartesian, coords_view};

  // Create the PointEvaluator once
  auto evaluator = factory.CreatePointEvaluator<Real>(cv);

  std::vector<Real> out_a(n), out_b(n);
  pcms::Rank2View<Real, pcms::HostMemorySpace> view_a(out_a.data(), n, 1);
  pcms::Rank2View<Real, pcms::HostMemorySpace> view_b(out_b.data(), n, 1);

  // Evaluate field_a then field_b with the same evaluator
  evaluator->Evaluate(field_a, view_a);
  evaluator->Evaluate(field_b, view_b);

  for (int i = 0; i < n; ++i) {
    Real x = pts[2 * i], y = pts[2 * i + 1];
    REQUIRE(out_a[i] == Catch::Approx(pcms::test::linear_f(x, y)).margin(1e-10));
    REQUIRE(out_b[i] == Catch::Approx(42.0).margin(1e-10));
  }
}

// ============================================================================
// OmegaH order-1 — OutOfBoundsPolicy::FILL
// ============================================================================

TEST_CASE("PointEvaluator: OmegaH order-1 out-of-bounds fill")
{
  auto lib  = Omega_h::Library{};
  auto mesh = Omega_h::build_box(
    lib.world(), OMEGA_H_SIMPLEX, 1, 1, 0, 20, 20, 0, false);

  auto factory =
    pcms::LagrangeFunctionSpace::FromMesh(mesh, 1, 1, CoordinateSystem::Cartesian,
                                         "global", pcms::LagrangeFunctionSpace::Backend::OmegaH);

  auto field_data = factory.CreateField<Real>();
  pcms::test::SetField(field_data.GetData(), *factory.GetLayout(), pcms::test::linear_f);

  // Points clearly outside [0,1]^2
  const std::vector<Real> outside_pts = {-0.5, 0.5, 1.5, 0.5, 0.5, -0.5,
                                         0.5, 1.5};
  int n = static_cast<int>(outside_pts.size()) / 2;
  pcms::Rank2View<const Real, pcms::HostMemorySpace> coords_view(
    outside_pts.data(), n, 2);
  pcms::CoordinateView<pcms::HostMemorySpace> cv{
    CoordinateSystem::Cartesian, coords_view};
  pcms::OutOfBoundsPolicy policy{pcms::OutOfBoundsMode::FILL, -999.0};
  auto evaluator = factory.CreatePointEvaluator<Real>(cv, policy);
  pcms::test::CheckFillMode(*evaluator, field_data, -999.0, outside_pts);
}

// ============================================================================
// UniformGrid — basic evaluation via new API
// ============================================================================

TEST_CASE("PointEvaluator: UniformGrid order-1 linear evaluation")
{
  // 2D grid: [0,1]^2 with 10x10 divisions
  const int N = 10;
  pcms::UniformGrid<2> grid;
  grid.bot_left = {0.0, 0.0};
  grid.edge_length = {1.0, 1.0};
  grid.divisions = {N, N};

  auto factory = pcms::LagrangeFunctionSpace::FromUniformGrid(
    grid, 1, CoordinateSystem::Cartesian, 1);

  auto field_data = factory.CreateField<Real>();
  pcms::test::SetField(field_data.GetData(), *factory.GetLayout(), pcms::test::linear_f);
  auto pts = pcms::test::StandardEvalCoords2D();
  int n    = static_cast<int>(pts.size()) / 2;
  pcms::Rank2View<const Real, pcms::HostMemorySpace> coords_view(
    pts.data(), n, 2);
  pcms::CoordinateView<pcms::HostMemorySpace> cv{
    CoordinateSystem::Cartesian, coords_view};
  auto evaluator = factory.CreatePointEvaluator<Real>(cv);
  pcms::test::CheckEvaluation(
    *evaluator, field_data, pts, pcms::test::linear_f, 1e-8);
}

TEST_CASE("PointEvaluator: SplineFunctionSpace uniform-grid evaluation")
{
  const int N = 10;
  pcms::UniformGrid<2> grid;
  grid.bot_left = {0.0, 0.0};
  grid.edge_length = {1.0, 1.0};
  grid.divisions = {N, N};

  auto factory = pcms::SplineFunctionSpace::FromUniformGrid(
    grid, CoordinateSystem::Cartesian);

  auto field_data = factory.CreateField<Real>();
  pcms::test::SetField(field_data.GetData(), *factory.GetLayout(), pcms::test::linear_f);
  auto pts = pcms::test::StandardEvalCoords2D();
  int n    = static_cast<int>(pts.size()) / 2;
  pcms::Rank2View<const Real, pcms::HostMemorySpace> coords_view(
    pts.data(), n, 2);
  pcms::CoordinateView<pcms::HostMemorySpace> cv{
    CoordinateSystem::Cartesian, coords_view};
  auto evaluator = factory.CreatePointEvaluator<Real>(cv);
  pcms::test::CheckEvaluation(
    *evaluator, field_data, pts, pcms::test::linear_f, 1e-8);
}

// ============================================================================
// FieldLayout — metadata interface
// ============================================================================

TEST_CASE("FieldLayout: metadata queries")
{
  auto lib  = Omega_h::Library{};
  auto mesh = Omega_h::build_box(
    lib.world(), OMEGA_H_SIMPLEX, 1, 1, 0, 10, 10, 0, false);

  auto factory =
    pcms::LagrangeFunctionSpace::FromMesh(mesh, 1, 1, CoordinateSystem::Cartesian,
                                         "global", pcms::LagrangeFunctionSpace::Backend::OmegaH);
  auto layout = factory.GetLayout();

  auto coords = layout->GetDOFHolderCoordinates();
  REQUIRE(coords.GetCoordinateSystem() == CoordinateSystem::Cartesian);
  REQUIRE(coords.GetCoordinates().extent(0) > 0);
  REQUIRE(coords.GetCoordinates().extent(1) == 2);
}

// ============================================================================
// FieldData / FieldLayout metadata queries
// ============================================================================

TEST_CASE("FieldData: layout metadata queries")
{
  auto lib  = Omega_h::Library{};
  auto mesh = Omega_h::build_box(
    lib.world(), OMEGA_H_SIMPLEX, 1, 1, 0, 10, 10, 0, false);

  auto factory =
    pcms::LagrangeFunctionSpace::FromMesh(mesh, 1, 1, CoordinateSystem::Cartesian,
                                         "global", pcms::LagrangeFunctionSpace::Backend::OmegaH);
  auto field_data = factory.CreateField<Real>();

  auto coords = factory.GetLayout()->GetDOFHolderCoordinates();
  REQUIRE(coords.GetCoordinateSystem() == CoordinateSystem::Cartesian);
  REQUIRE(coords.GetCoordinates().extent(0) > 0);
  REQUIRE(coords.GetCoordinates().extent(1) == 2);
}

// ============================================================================
// CreateFieldData / SimpleFieldData round-trip
// ============================================================================

TEST_CASE("SimpleFieldData: set and get DOF holder data round-trip")
{
  auto lib  = Omega_h::Library{};
  auto mesh = Omega_h::build_box(
    lib.world(), OMEGA_H_SIMPLEX, 1, 1, 0, 10, 10, 0, false);

  auto factory =
    pcms::LagrangeFunctionSpace::FromMesh(mesh, 1, 1, CoordinateSystem::Cartesian,
                                         "global", pcms::LagrangeFunctionSpace::Backend::OmegaH);
  auto field_data = factory.CreateField<Real>();

  auto& layout = *factory.GetLayout();
  int n = layout.GetNumOwnedDofHolder();
  REQUIRE(n > 0);

  // Write sequential values
  std::vector<Real> data_in(n);
  for (int i = 0; i < n; ++i)
    data_in[i] = static_cast<Real>(i) * 0.5;

  field_data.GetData().SetDOFHolderDataHost(
    pcms::Rank1View<const Real, pcms::HostMemorySpace>(data_in.data(), n));

  auto data_out = field_data.GetData().GetDOFHolderDataHost();
  REQUIRE(static_cast<int>(data_out.size()) == n);
  for (int i = 0; i < n; ++i) {
    REQUIRE(data_out[i] == Catch::Approx(data_in[i]));
  }
}

// ============================================================================
// MeshFields FieldEvaluatorFactory metadata (only when MeshFields is enabled)
// ============================================================================

#ifdef PCMS_ENABLE_MESHFIELDS
TEST_CASE("FieldLayout: MeshFields metadata queries")
{
  auto lib  = Omega_h::Library{};
  auto mesh = Omega_h::build_box(
    lib.world(), OMEGA_H_SIMPLEX, 1, 1, 0, 10, 10, 0, false);

  auto factory = pcms::LagrangeFunctionSpace::FromMesh(
    mesh, 1, 1, CoordinateSystem::Cartesian, "global",
    pcms::LagrangeFunctionSpace::Backend::MeshFields);

  auto layout = factory.GetLayout();
  auto coords = layout->GetDOFHolderCoordinates();
  REQUIRE(coords.GetCoordinateSystem() == CoordinateSystem::Cartesian);
  REQUIRE(coords.GetCoordinates().extent(0) > 0);
  REQUIRE(coords.GetCoordinates().extent(1) == 2);
}

TEST_CASE("PointEvaluator: MeshFields order-1 linear evaluation")
{
  auto lib = Omega_h::Library{};
  auto mesh = Omega_h::build_box(lib.world(), OMEGA_H_SIMPLEX, 1, 1, 0, 100,
                                 100, 0, false);

  auto factory = pcms::LagrangeFunctionSpace::FromMesh(
    mesh, 1, 1, CoordinateSystem::Cartesian, "global",
    pcms::LagrangeFunctionSpace::Backend::MeshFields);

  auto field_data = factory.CreateField<Real>();
  pcms::test::SetField(field_data.GetData(), *factory.GetLayout(), pcms::test::linear_f);

  auto pts = pcms::test::StandardEvalCoords2D();
  int n = static_cast<int>(pts.size()) / 2;
  pcms::Rank2View<const Real, pcms::HostMemorySpace> coords_view(pts.data(), n,
                                                                 2);
  pcms::CoordinateView<pcms::HostMemorySpace> cv{CoordinateSystem::Cartesian,
                                                 coords_view};
  auto evaluator = factory.CreatePointEvaluator<Real>(cv);
  pcms::test::CheckEvaluation(*evaluator, field_data, pts,
                              pcms::test::linear_f);
}

TEST_CASE("PointEvaluator: MeshFields out-of-bounds fill")
{
  auto lib = Omega_h::Library{};
  auto mesh =
    Omega_h::build_box(lib.world(), OMEGA_H_SIMPLEX, 1, 1, 0, 20, 20, 0, false);

  auto factory = pcms::LagrangeFunctionSpace::FromMesh(
    mesh, 1, 1, CoordinateSystem::Cartesian, "global",
    pcms::LagrangeFunctionSpace::Backend::MeshFields);

  auto field_data = factory.CreateField<Real>();
  pcms::test::SetField(field_data.GetData(), *factory.GetLayout(), pcms::test::linear_f);

  const std::vector<Real> outside_pts = {-0.5, 0.5,  1.5, 0.5,
                                         0.5,  -0.5, 0.5, 1.5};
  int n = static_cast<int>(outside_pts.size()) / 2;
  pcms::Rank2View<const Real, pcms::HostMemorySpace> coords_view(
    outside_pts.data(), n, 2);
  pcms::CoordinateView<pcms::HostMemorySpace> cv{CoordinateSystem::Cartesian,
                                                 coords_view};
  pcms::OutOfBoundsPolicy policy{pcms::OutOfBoundsMode::FILL, -999.0};
  auto evaluator = factory.CreatePointEvaluator<Real>(cv, policy);
  pcms::test::CheckFillMode(*evaluator, field_data, -999.0, outside_pts);
}

TEST_CASE(
  "PointEvaluator: MeshFields same evaluator reused for two FieldData objects")
{
  auto lib = Omega_h::Library{};
  auto mesh =
    Omega_h::build_box(lib.world(), OMEGA_H_SIMPLEX, 1, 1, 0, 50, 50, 0, false);

  auto factory = pcms::LagrangeFunctionSpace::FromMesh(
    mesh, 1, 1, CoordinateSystem::Cartesian, "global",
    pcms::LagrangeFunctionSpace::Backend::MeshFields);

  auto field_a = factory.CreateField<Real>();
  auto field_b = factory.CreateField<Real>();
  pcms::test::SetField(field_a.GetData(), *factory.GetLayout(), pcms::test::linear_f);
  pcms::test::SetField(field_b.GetData(), *factory.GetLayout(),
                       [](Real, Real) { return Real(42); });

  auto pts = pcms::test::StandardEvalCoords2D();
  int n = static_cast<int>(pts.size()) / 2;
  pcms::Rank2View<const Real, pcms::HostMemorySpace> coords_view(pts.data(), n,
                                                                 2);
  pcms::CoordinateView<pcms::HostMemorySpace> cv{CoordinateSystem::Cartesian,
                                                 coords_view};

  auto evaluator = factory.CreatePointEvaluator<Real>(cv);

  std::vector<Real> out_a(n), out_b(n);
  pcms::Rank2View<Real, pcms::HostMemorySpace> view_a(out_a.data(), n, 1);
  pcms::Rank2View<Real, pcms::HostMemorySpace> view_b(out_b.data(), n, 1);

  evaluator->Evaluate(field_a, view_a);
  evaluator->Evaluate(field_b, view_b);

  for (int i = 0; i < n; ++i) {
    Real x = pts[2 * i], y = pts[2 * i + 1];
    REQUIRE(out_a[i] ==
            Catch::Approx(pcms::test::linear_f(x, y)).margin(1e-10));
    REQUIRE(out_b[i] == Catch::Approx(42.0).margin(1e-10));
  }
}

TEST_CASE("LagrangeFunctionSpace: MeshFields rejects multi-component fields")
{
  auto lib = Omega_h::Library{};
  auto mesh =
    Omega_h::build_box(lib.world(), OMEGA_H_SIMPLEX, 1, 1, 0, 10, 10, 0, false);

  REQUIRE_THROWS_AS(pcms::LagrangeFunctionSpace::FromMesh(
                      mesh, 1, 2, CoordinateSystem::Cartesian, "global",
                      pcms::LagrangeFunctionSpace::Backend::MeshFields),
                    pcms::pcms_error);
}
#endif // PCMS_ENABLE_MESHFIELDS
