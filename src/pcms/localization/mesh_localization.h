#ifndef PCMS_LOCALIZATION_ADJACENCY_LOCALIZATION_H
#define PCMS_LOCALIZATION_ADJACENCY_LOCALIZATION_H

#include "pcms/localization/localization_factory.h"
#include "pcms/field/evaluator/mls_options.h"

#include <Omega_h_mesh.hpp>

namespace pcms
{

// LocalizationFactory implementation using mesh-adjacency BFS search.
//
// Stores a non-owning reference to the source mesh. The source mesh must
// outlive this factory.
//
// Build() dispatch:
//   - source_entity_dim == VERT: builds a temporary vertex-only Omega_h mesh
//     from the target CoordinateView and calls the two-mesh searchNeighbors.
//   - source_entity_dim != VERT: falls back to N^2 BuildPointCloudSupports
//     using entity centroids as the source.
//
// Centroid adjacency traversal is intentionally deferred to a follow-on change.
// The current CreatePointEvaluator API only accepts points, so selecting a
// centroid-to-vertex adjacency path would require stronger target provenance
// than is currently available.
class AdjacencyLocalizationFactory : public LocalizationFactory
{
public:
  AdjacencyLocalizationFactory(Omega_h::Mesh& source_mesh,
                               int source_entity_dim,
                               MLSOptions options)
    : source_mesh_(source_mesh),
      source_entity_dim_(source_entity_dim),
      options_(options)
  {
  }

  SupportResults Build(CoordinateView<HostMemorySpace> target_coords) const override;

private:
  Omega_h::Mesh& source_mesh_;
  int source_entity_dim_;
  MLSOptions options_;
};

} // namespace pcms

#endif // PCMS_LOCALIZATION_ADJACENCY_LOCALIZATION_H
