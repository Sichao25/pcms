#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <Omega_h_mesh.hpp>
#include <Omega_h_build.hpp>
#include <Omega_h_for.hpp>
#include <pcms/transfer_field2.h>
#include "pcms/adapter/meshfields/mesh_fields_adapter.h"
#include "pcms/adapter/meshfields/mesh_fields_adapter2.h"
#include "pcms/lagrange_field_factory.h"
#include "pcms/utility/assert.h"
#include <Kokkos_Core.hpp>

using pcms::Real;

void test_copy(Omega_h::CommPtr world, int dim, int order, int num_components)
{
  int nx = 100;
  int ny = dim > 1 ? 100 : 0;
  int nz = dim > 2 ? 100 : 0;

  auto mesh =
    Omega_h::build_box(world, OMEGA_H_SIMPLEX, 1, 1, 1, nx, ny, nz, false);
  auto factory = pcms::LagrangeFieldFactory::FromMesh(
    mesh, order, num_components, pcms::CoordinateSystem::Cartesian, "global",
    pcms::LagrangeFieldFactory::Backend::OmegaH);
  // FIXME this indicates that we are not exposing the right API from the fields to be able
  // to allocate buffers for getting/setting the field data.
  auto layout = factory.GetLayout();
  int ndata = layout->GetNumOwnedDofHolder() * num_components;
  Omega_h::HostWrite<Real> ids(ndata);
  Kokkos::parallel_for(
    Kokkos::RangePolicy<pcms::HostMemorySpace::execution_space>(0, ndata),
    [=](int i) { ids[i] = i; });

  auto original = factory.CreateFieldReal();
  original->SetDOFHolderData(pcms::make_const_array_view(ids));

  auto copied = factory.CreateFieldReal();
  pcms::copy_field2(*original, *copied);
  auto copied_array = copied->GetDOFHolderData();

  REQUIRE(copied_array.size() == ndata);
  int sum = 0;
  Kokkos::parallel_reduce(
    Kokkos::RangePolicy<pcms::HostMemorySpace::execution_space>(0, ndata),
    [=](int i, int& local_sum) {
      local_sum += std::abs(ids[i] - copied_array[i]) < 1e-12;
    },
    sum);
  REQUIRE(sum == ndata);
}

TEST_CASE("copy omega_h_field2 data")
{
  auto lib = Omega_h::Library{};
  test_copy(lib.world(), 2, 0, 1);
  test_copy(lib.world(), 2, 1, 1);
  auto mesh =
    Omega_h::build_box(lib.world(), OMEGA_H_SIMPLEX, 1, 1, 1, 100, 100, 0, false);
  REQUIRE_THROWS_AS(
    pcms::LagrangeFieldFactory::FromMesh(
      mesh,
      2, 1, pcms::CoordinateSystem::Cartesian, "global",
      pcms::LagrangeFieldFactory::Backend::OmegaH),
    pcms::pcms_error);
}
