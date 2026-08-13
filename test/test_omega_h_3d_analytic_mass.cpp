#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <Omega_h_build.hpp>
#include <Omega_h_library.hpp>
#include <Omega_h_mesh.hpp>
#include <pcms/field/function_space/lagrange.h>
#include <pcms/field/layout/omega_h_lagrange.h>
#include <pcms/transfer/omega_h_mass_integrator.hpp>
#include "field_test_utils.h"
#include <petscksp.h>

namespace
{

// Reference corner tet:
//   v0=(0,0,0), v1=(1,0,0), v2=(0,1,0), v3=(0,0,1)
// Volume V = 1/6.
//
// P1 consistent mass on a tet:
//   M_ij = V/20 * (1 + delta_ij)
//        = 1/60 on diagonal, 1/120 off-diagonal.
Omega_h::Mesh BuildReferenceTet(Omega_h::Library& lib)
{
  const Omega_h::Reals coords({
    0.0, 0.0, 0.0, // v0
    1.0, 0.0, 0.0, // v1
    0.0, 1.0, 0.0, // v2
    0.0, 0.0, 1.0  // v3
  });
  // One tet: vertices 0,1,2,3
  const Omega_h::LOs ev2v({0, 1, 2, 3});

  Omega_h::Mesh mesh(&lib);
  Omega_h::build_from_elems_and_coords(&mesh, OMEGA_H_SIMPLEX, 3, ev2v, coords);

  // Classification tags required by OmegaHLagrangeLayout / function space.
  for (Omega_h::Int dim = 0; dim <= 3; ++dim) {
    mesh.add_tag<Omega_h::I8>(
      dim, "class_dim", 1,
      Omega_h::Read<Omega_h::I8>(mesh.nents(dim), Omega_h::I8(dim)));
    mesh.add_tag<Omega_h::ClassId>(
      dim, "class_id", 1,
      Omega_h::Read<Omega_h::ClassId>(mesh.nents(dim), Omega_h::ClassId(0)));
  }
  return mesh;
}

} // namespace

TEST_CASE("OmegaHMassIntegrator (3D): P1 mass on reference tet matches "
          "analytic V/20*(1+delta)",
          "[mass_integrator][3d][analytic]")
{
  Omega_h::Library lib;
  auto mesh = BuildReferenceTet(lib);
  REQUIRE(mesh.dim() == 3);
  REQUIRE(mesh.nverts() == 4);
  REQUIRE(mesh.nelems() == 1);

  constexpr pcms::Real V = 1.0 / 6.0;
  constexpr pcms::Real M_diag = V / 10.0; // 1/60
  constexpr pcms::Real M_off = V / 20.0;  // 1/120

  auto space = pcms::test::MakeP1Space(mesh);
  auto integrator = pcms::BuildOmegaHMassIntegrator(*space);
  Mat mat = integrator->GetMatrix();

  const auto layout =
    std::dynamic_pointer_cast<const pcms::OmegaHLagrangeLayout>(
      space->GetLayout());
  REQUIRE(layout != nullptr);

  // PETSc rows are active indices = global_to_local(local_vertex).
  const auto perm = layout->GetGlobalToLocalPermutationHost();
  REQUIRE(static_cast<int>(perm.extent(0)) == 4);

  pcms::Real grand_total = 0.0;
  for (int i = 0; i < 4; ++i) {
    pcms::Real row_sum = 0.0;
    for (int j = 0; j < 4; ++j) {
      PetscInt r = static_cast<PetscInt>(perm(i));
      PetscInt c = static_cast<PetscInt>(perm(j));
      PetscScalar got = 0.0;
      MatGetValues(mat, 1, &r, 1, &c, &got);
      const pcms::Real expected = (i == j) ? M_diag : M_off;
      CAPTURE(i, j, r, c, expected, got);
      CHECK(static_cast<pcms::Real>(got) ==
            Catch::Approx(expected).epsilon(1e-12));
      row_sum += static_cast<pcms::Real>(got);
    }
    // Row sum = ∫ λ_i = V/4
    CHECK(row_sum == Catch::Approx(V / 4.0).epsilon(1e-12));
    grand_total += row_sum;
  }
  // Sum of row sums = volume
  CHECK(grand_total == Catch::Approx(V).epsilon(1e-12));
}

TEST_CASE("OmegaHMassIntegrator (3D): P0 mass on reference tet equals volume",
          "[mass_integrator][3d][analytic]")
{
  Omega_h::Library lib;
  auto mesh = BuildReferenceTet(lib);

  constexpr pcms::Real V = 1.0 / 6.0;

  auto space = pcms::test::MakeP0Space(mesh);
  auto integrator = pcms::BuildOmegaHMassIntegrator(*space);
  Mat mat = integrator->GetMatrix();

  PetscInt r = 0;
  PetscInt c = 0;
  PetscScalar got = 0.0;
  MatGetValues(mat, 1, &r, 1, &c, &got);
  CHECK(static_cast<pcms::Real>(got) == Catch::Approx(V).epsilon(1e-12));
}
