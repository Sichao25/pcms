#ifndef PCMS_LOCALIZATION_ADJACENCY_LOCALIZATION_H
#define PCMS_LOCALIZATION_ADJACENCY_LOCALIZATION_H

#include "pcms/localization/localization_factory.h"
#include "pcms/field/evaluator/mls_options.h"
#include "pcms/discretization/discretization/omega_h.hpp"

#include <Omega_h_mesh.hpp>

namespace pcms
{

// LocalizationFactory implementation using mesh-adjacency BFS search.
//
// Stores a non-owning reference to the source mesh. The source mesh must
// outlive this factory.
//
// Build(CoordinateView) dispatch:
//   - source_entity_dim == VERT: uses two-mesh searchNeighbors (adjacency BFS).
//   - source_entity_dim != VERT: falls back to N^2 BuildPointCloudSupports.
//
// Build(CoordinateView, Discretization) dispatch additionally checks whether
// source and target share the same Omega_h mesh. When they do and the mesh is
// 2D, the efficient centroid-to-vertex searchNeighbors path is used instead of
// the N^2 fallback.
class AdjacencyLocalizationFactory : public LocalizationFactory
{
public:
  AdjacencyLocalizationFactory(Omega_h::Mesh& source_mesh,
                               int source_entity_dim,
                               MLSOptions options)
    : source_mesh_(source_mesh),
      source_entity_dim_(source_entity_dim),
      options_(options),
      source_disc_(source_mesh)
  {
  }

  SupportResults Build(CoordinateView<HostMemorySpace> target_coords) const override;

  SupportResults Build(CoordinateView<HostMemorySpace> target_coords,
                       const Discretization& target_disc) const override;

private:
  Omega_h::Mesh& source_mesh_;
  int source_entity_dim_;
  MLSOptions options_;
  OmegaHDiscretization source_disc_;
};

} // namespace pcms

#endif // PCMS_LOCALIZATION_ADJACENCY_LOCALIZATION_H
