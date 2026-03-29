#include "pcms/localization/mesh_localization.h"
#include "pcms/localization/adj_search.hpp"
#include "pcms/localization/mls_support_helpers.h"
#include "pcms/field/coordinate_system.h"
#include "pcms/utility/assert.h"
#include "pcms/utility/mesh_geometry.h"
#include "pcms/utility/omega_h_array_utils.h"
#include "pcms/discretization/discretization/omega_h.hpp"

#include <Omega_h_array.hpp>

namespace pcms
{

SupportResults AdjacencyLocalizationFactory::Build(
  CoordinateView<HostMemorySpace> target_coords) const
{
  if (target_coords.GetCoordinateSystem() != CoordinateSystem::Cartesian) {
    throw pcms_error(
      "AdjacencyLocalizationFactory: only Cartesian coordinates are supported");
  }

  const auto tgt_view = target_coords.GetCoordinates();
  const int dim = source_mesh_.dim();
  PCMS_ALWAYS_ASSERT(static_cast<int>(tgt_view.extent(1)) == dim);
  const int n_tgt = static_cast<int>(tgt_view.extent(0));

  // Pass radius^2 as the cutoff. searchNeighbors stores it in a "radii2" array
  // and compares it against squared distances, so the squared value is correct.
  Omega_h::Real radius_sq = options_.radius * options_.radius;

  if (source_entity_dim_ == Omega_h::VERT) {
    Omega_h::Reals target_coords_oh =
      flatten_to_omega_h_reals(tgt_view, "tgt_coords");

    return searchNeighbors(source_mesh_, target_coords_oh, n_tgt, radius_sq,
                           static_cast<Omega_h::LO>(options_.min_req_supports),
                           static_cast<Omega_h::LO>(3 * options_.min_req_supports),
                           options_.adapt_radius);
  }
  else {
    // Non-vertex source entities without a matching target discretization
    // use generic point-cloud localization. When source and target share the
    // same mesh, call Build(coords, disc) to get the centroid-adjacency path.
    const Omega_h::Reals src_oh =
      get_entity_centroids(source_mesh_, source_entity_dim_);

    Omega_h::Reals target_coords_oh =
      flatten_to_omega_h_reals(tgt_view, "tgt_coords");

    return BuildPointCloudSupports(src_oh, target_coords_oh, dim,
                                   options_.radius, options_.min_req_supports,
                                   options_.adapt_radius);
  }
}

SupportResults AdjacencyLocalizationFactory::Build(
  CoordinateView<HostMemorySpace> target_coords,
  const Discretization& target_disc) const
{
  // Use the efficient centroid-to-vertex adjacency path when:
  //   1. Source entities are centroids (not vertices)
  //   2. The mesh is 2D (adjBasedSearchCentroidNodes is 2D only)
  //   3. Source and target share the same underlying mesh
  if (source_entity_dim_ != Omega_h::VERT && source_mesh_.dim() == 2 &&
      source_disc_.SameEntities(target_disc)) {
    PCMS_ALWAYS_ASSERT(
      static_cast<int>(target_coords.GetCoordinates().extent(0)) ==
      source_mesh_.nverts());
    Omega_h::Real radius_sq = options_.radius * options_.radius;
    return searchNeighbors(source_mesh_, radius_sq,
                           static_cast<Omega_h::LO>(options_.min_req_supports),
                           options_.adapt_radius);
  }
  return Build(target_coords);
}

} // namespace pcms
