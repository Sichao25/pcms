#include "pcms/field/layout/xgc.h"
#include "pcms/utility/assert.h"

namespace pcms
{

XGCFieldLayout::XGCFieldLayout(
  const ReverseClassificationVertex& reverse_classification,
  std::function<int8_t(int, int)> in_overlap, LO num_plane_nodes)
  : owned_("xgc_owned", num_plane_nodes),
    gids_("xgc_gids", num_plane_nodes),
    class_dims_("xgc_class_dims", num_plane_nodes),
    class_ids_("xgc_class_ids", num_plane_nodes),
    coords_("xgc_coords", num_plane_nodes, 2),
    num_plane_nodes_(num_plane_nodes)
{
  PCMS_ALWAYS_ASSERT(static_cast<bool>(in_overlap));
  for (LO i = 0; i < num_plane_nodes_; ++i) {
    owned_(i) = false;
    gids_(i) = static_cast<GO>(i + 1);
    class_dims_(i) = -1;
    class_ids_(i) = -1;
  }

  for (const auto& [geom, verts] : reverse_classification) {
    if (!in_overlap(geom.dim, geom.id))
      continue;
    for (LO vert : verts) {
      PCMS_ALWAYS_ASSERT(vert >= 0 && vert < num_plane_nodes_);
      owned_(vert) = true;
      class_dims_(vert) = geom.dim;
      class_ids_(vert) = geom.id;
    }
  }
  // Copy coordinates to device
  Kokkos::deep_copy(coords_, 0.0);
  discretization_ =
    std::make_shared<XGCDiscretization>(reverse_classification, num_plane_nodes_);
}

std::shared_ptr<const Discretization>
XGCFieldLayout::GetDiscretization() const noexcept
{
  return discretization_;
}

int XGCFieldLayout::GetNumComponents() const
{
  return 1;
}

LO XGCFieldLayout::GetNumOwnedDofHolder() const
{
  return num_plane_nodes_;
}

GO XGCFieldLayout::GetNumGlobalDofHolder() const
{
  return num_plane_nodes_;
}

Rank1View<const bool, HostMemorySpace> XGCFieldLayout::GetOwned() const
{
  return make_const_array_view(owned_);
}

GlobalIDView<HostMemorySpace> XGCFieldLayout::GetGids() const
{
  return make_const_array_view(gids_);
}

bool XGCFieldLayout::IsDistributed() const
{
  return false;
}

EntOffsetsArray XGCFieldLayout::GetEntOffsets() const
{
  return {0, static_cast<size_t>(num_plane_nodes_),
          static_cast<size_t>(num_plane_nodes_),
          static_cast<size_t>(num_plane_nodes_),
          static_cast<size_t>(num_plane_nodes_)};
}

CoordinateView<DeviceMemorySpace> XGCFieldLayout::GetDOFHolderCoordinates() const
{
  using LayoutPolicy = detail::default_layout_for_memory_space_t<DeviceMemorySpace>;
  return CoordinateView<DeviceMemorySpace>{
    CoordinateSystem::XGC,
    Rank2View<const Real, DeviceMemorySpace, LayoutPolicy>(coords_.data(), num_plane_nodes_, 2)};
}

int XGCFieldLayout::GetDimension() const
{
  return 2;
}

Rank1View<const LO, HostMemorySpace>
XGCFieldLayout::GetDOFHolderClassificationDimensions() const
{
  return make_const_array_view(class_dims_);
}

Rank1View<const LO, HostMemorySpace>
XGCFieldLayout::GetDOFHolderClassificationIds() const
{
  return make_const_array_view(class_ids_);
}

LO XGCFieldLayout::GetFullDataSize() const noexcept
{
  return num_plane_nodes_;
}

} // namespace pcms
