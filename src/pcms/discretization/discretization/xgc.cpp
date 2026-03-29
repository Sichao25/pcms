#include "pcms/discretization/discretization/xgc.hpp"

namespace pcms
{

XGCDiscretization::XGCDiscretization(
  const ReverseClassificationVertex& reverse_classification,
  LO num_plane_nodes)
  : reverse_classification_(&reverse_classification),
    num_plane_nodes_(num_plane_nodes),
    class_dims_("xgc_discretization_class_dims", num_plane_nodes),
    class_ids_("xgc_discretization_class_ids", num_plane_nodes)
{
  Kokkos::deep_copy(class_dims_, static_cast<ClassificationDimension>(-1));
  Kokkos::deep_copy(class_ids_, static_cast<ClassificationId>(-1));

  for (const auto& [geom, verts] : reverse_classification) {
    for (LO vert : verts) {
      if (vert >= 0 && vert < num_plane_nodes_) {
        class_dims_(vert) = geom.dim;
        class_ids_(vert) = geom.id;
      }
    }
  }
}

bool XGCDiscretization::SameEntities(const Discretization& other) const noexcept
{
  auto p = dynamic_cast<const XGCDiscretization*>(&other);
  return p != nullptr && p->reverse_classification_ == reverse_classification_ &&
         p->num_plane_nodes_ == num_plane_nodes_;
}

int XGCDiscretization::GetDimension() const
{
  return 2;
}

LO XGCDiscretization::GetNumEntities(int entity_dim) const
{
  return entity_dim == Vertex ? num_plane_nodes_ : 0;
}

Rank1View<const ClassificationDimension, HostMemorySpace>
XGCDiscretization::GetEntityClassificationDimensions(int entity_dim) const
{
  return entity_dim == Vertex
           ? make_const_array_view(class_dims_)
           : Rank1View<const ClassificationDimension, HostMemorySpace>(nullptr,
                                                                       0);
}

Rank1View<const ClassificationId, HostMemorySpace>
XGCDiscretization::GetEntityClassificationIds(int entity_dim) const
{
  return entity_dim == Vertex
           ? make_const_array_view(class_ids_)
           : Rank1View<const ClassificationId, HostMemorySpace>(nullptr, 0);
}

} // namespace pcms
