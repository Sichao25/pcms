#ifndef PCMS_OMEGA_H_LAGRANGE_COMMON_H
#define PCMS_OMEGA_H_LAGRANGE_COMMON_H

// Shared types used by both OmegaHLagrangeField and
// OmegaHLagrangeEvaluatorFactory. Extracted here to break the circular include
// that would arise if the field header included the factory header.

#include <Omega_h_mesh.hpp>
#include <Kokkos_Core.hpp>
#include "pcms/field/field.h"         // OutOfBoundsMode
#include "pcms/field/coordinate_system.h"
#include "pcms/localization/point_search.h"
#include "pcms/utility/types.h"
#include "pcms/utility/arrays.h"
#include "pcms/utility/assert.h"
#include <memory>
#include <variant>
#include <stdexcept>

namespace pcms
{

struct OmegaHLagrangeLocHint
{
  // Per valid query point:
  Kokkos::View<LO*, HostMemorySpace> elem_ids;      // containing element
  Kokkos::View<Real**, HostMemorySpace> bary;        // [n_valid x (dim+1)]
  Kokkos::View<LO*, HostMemorySpace> orig_indices;   // original query index
  // Points that fell outside the mesh:
  Kokkos::View<LO*, HostMemorySpace> missing_indices;
  OutOfBoundsMode mode;
};

namespace detail
{
template <int Dim>
OmegaHLagrangeLocHint BuildLagrangeLocHint(
  int mesh_dim,
  Kokkos::View<typename PointLocalizationSearch<Dim>::Result*,
               HostMemorySpace> results_h,
  OutOfBoundsMode mode)
{
  LO n = static_cast<LO>(results_h.size());
  std::vector<LO> valid, missing;
  for (LO i = 0; i < n; ++i) {
    bool out = (static_cast<int>(results_h(i).dimensionality) != mesh_dim) ||
               (results_h(i).element_id < 0);
    if (out)
      missing.push_back(i);
    else
      valid.push_back(i);
  }

  if (mode == OutOfBoundsMode::ERROR) {
    PCMS_ALWAYS_ASSERT(missing.empty() && "Points found outside mesh domain");
  } else if (mode == OutOfBoundsMode::NEAREST_BOUNDARY) {
    PCMS_ALWAYS_ASSERT(false && "NEAREST_BOUNDARY mode not yet implemented");
  }

  LO nv = static_cast<LO>(valid.size());
  LO nm = static_cast<LO>(missing.size());

  Kokkos::View<LO*, HostMemorySpace> elem_ids("elem_ids", nv);
  Kokkos::View<Real**, HostMemorySpace> bary("bary", nv, mesh_dim + 1);
  Kokkos::View<LO*, HostMemorySpace> orig_indices("orig_indices", nv);
  Kokkos::View<LO*, HostMemorySpace> missing_indices("missing_indices", nm);

  for (LO k = 0; k < nv; ++k) {
    LO i = valid[k];
    elem_ids(k) = results_h(i).element_id;
    orig_indices(k) = i;
    for (int d = 0; d <= mesh_dim; ++d)
      bary(k, d) = results_h(i).parametric_coords[d];
  }
  for (LO k = 0; k < nm; ++k)
    missing_indices(k) = missing[k];

  return OmegaHLagrangeLocHint{elem_ids, bary, orig_indices, missing_indices,
                                mode};
}

inline std::variant<GridPointSearch2D, GridPointSearch3D> MakeSearch(
  Omega_h::Mesh& mesh)
{
  if (mesh.dim() == 2)
    return GridPointSearch2D(mesh, 10, 10);
  else if (mesh.dim() == 3)
    return GridPointSearch3D(mesh, 10, 10, 10);
  throw std::invalid_argument(
    "OmegaHLagrangeField: only 2D and 3D meshes are supported");
}
} // namespace detail

} // namespace pcms

#endif // PCMS_OMEGA_H_LAGRANGE_COMMON_H
