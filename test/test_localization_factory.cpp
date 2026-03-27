#include <catch2/catch_test_macros.hpp>

#include <Omega_h_build.hpp>
#include <Omega_h_library.hpp>

#include "pcms/field/layout/point_cloud.h"
#include "pcms/field/evaluator/mls_options.h"
#include "pcms/localization/adj_search.hpp"
#include "pcms/localization/mesh_localization.h"
#include "pcms/localization/mls_support_helpers.h"
#include "pcms/localization/point_cloud_localization.h"
#include "field_test_utils.h"

TEST_CASE("LocalizationFactory: point-cloud Build matches BuildPointCloudSupports")
{
  auto lib = Omega_h::Library{};
  auto mesh =
    Omega_h::build_box(lib.world(), OMEGA_H_SIMPLEX, 1, 1, 1, 8, 8, 0, false);

  pcms::MLSOptions options;
  options.radius = 0.2;
  options.min_req_supports = 10;
  options.adapt_radius = true;

  const int dim = mesh.dim();
  auto source_coords = mesh.coords();
  auto target_coords = pcms::test::CopyOmegaHRealsToVector(source_coords);
  pcms::Rank2View<const pcms::Real, pcms::HostMemorySpace> target_view(
    target_coords.data(), mesh.nverts(), dim);
  pcms::CoordinateView<pcms::HostMemorySpace> target_cv{
    pcms::CoordinateSystem::Cartesian, target_view};

  auto coords_read = Omega_h::HostRead<Omega_h::Real>(source_coords);
  Kokkos::View<pcms::Real**, Kokkos::HostSpace> coords_host(
    "point_cloud_coords", mesh.nverts(), dim);
  for (int i = 0; i < mesh.nverts(); ++i)
    for (int d = 0; d < dim; ++d)
      coords_host(i, d) = coords_read[i * dim + d];
  auto coords_dev = Kokkos::create_mirror_view_and_copy(
    Kokkos::DefaultExecutionSpace{}, coords_host);

  auto layout = std::make_shared<pcms::PointCloudLayout>(
    dim, coords_dev, pcms::CoordinateSystem::Cartesian);
  pcms::PointCloudLocalizationFactory factory(layout, options);

  auto actual = factory.Build(target_cv);
  auto expected = pcms::BuildPointCloudSupports(
    source_coords, source_coords, dim, options.radius, options.min_req_supports,
    options.adapt_radius);

  pcms::test::CheckSupportResultsEquivalent(actual, expected);
}

TEST_CASE("LocalizationFactory: vertex adjacency Build matches two-mesh searchNeighbors")
{
  auto lib = Omega_h::Library{};
  auto world = lib.world();
  auto source_mesh =
    Omega_h::build_box(world, OMEGA_H_SIMPLEX, 1, 1, 1, 12, 12, 0, false);
  auto target_mesh =
    Omega_h::build_box(world, OMEGA_H_SIMPLEX, 1, 1, 1, 10, 10, 0, false);

  pcms::MLSOptions options;
  options.radius = 0.12;
  options.min_req_supports = 15;
  options.adapt_radius = true;

  pcms::AdjacencyLocalizationFactory factory(source_mesh, Omega_h::VERT,
                                             options);
  auto target_coords = pcms::test::CopyOmegaHRealsToVector(target_mesh.coords());
  pcms::Rank2View<const pcms::Real, pcms::HostMemorySpace> target_view(
    target_coords.data(), target_mesh.nverts(), target_mesh.dim());
  pcms::CoordinateView<pcms::HostMemorySpace> target_cv{
    pcms::CoordinateSystem::Cartesian, target_view};

  auto actual = factory.Build(target_cv);

  Omega_h::Real radius_sq = options.radius * options.radius;
  auto expected = pcms::searchNeighbors(
    source_mesh, target_mesh, radius_sq,
    static_cast<Omega_h::LO>(options.min_req_supports),
    static_cast<Omega_h::LO>(3 * options.min_req_supports),
    options.adapt_radius);

  pcms::test::CheckSupportResultsEquivalent(actual, expected);
}
