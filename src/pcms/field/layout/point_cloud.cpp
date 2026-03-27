#include "pcms/field/layout/point_cloud.h"
#include <memory>
#include <Kokkos_StdAlgorithms.hpp>

namespace pcms
{

PointCloudLayout::PointCloudLayout(int dim, Kokkos::View<Real**> coords,
                                   CoordinateSystem coordinate_system)
  : dim_(dim),
    coordinate_system_(coordinate_system),
    coords_(coords),
    coords_host_(Kokkos::create_mirror_view_and_copy(HostMemorySpace(), coords_)),
    owned_("", coords.extent(0)),
    gids_("", coords.extent(0)),
    owned_host_("", coords.extent(0)),
    gids_host_("", coords.extent(0))
{
  components_ = 1;

  namespace KE = Kokkos::Experimental;
  KE::fill(Kokkos::DefaultExecutionSpace(), owned_, true);
  iota_view(gids_);

  LO n = static_cast<LO>(coords.extent(0));
  classification_dims_host_ =
    Kokkos::View<LO*, HostMemorySpace>("classification_dims", n);
  classification_ids_host_ =
    Kokkos::View<LO*, HostMemorySpace>("classification_ids", n);
  Kokkos::deep_copy(classification_dims_host_, static_cast<LO>(dim_));
  Kokkos::deep_copy(classification_ids_host_, LO{0});
}

int PointCloudLayout::GetNumComponents() const
{
  return components_;
}

LO PointCloudLayout::GetNumOwnedDofHolder() const
{
  return coords_.extent(0);
}

GO PointCloudLayout::GetNumGlobalDofHolder() const
{
  return coords_.extent(0);
}

Rank1View<const bool, HostMemorySpace> PointCloudLayout::GetOwned() const
{
  Kokkos::deep_copy(owned_host_, owned_);
  return make_const_array_view(owned_host_);
}

GlobalIDView<HostMemorySpace> PointCloudLayout::GetGids() const
{
  Kokkos::deep_copy(gids_host_, gids_);
  return GlobalIDView<HostMemorySpace>(gids_host_.data(), gids_host_.size());
}

CoordinateView<HostMemorySpace> PointCloudLayout::GetDOFHolderCoordinates()
  const
{
  Rank2View<const Real, HostMemorySpace> coords_view(
    coords_host_.data(), coords_host_.extent(0), dim_);
  return CoordinateView<HostMemorySpace>{coordinate_system_, coords_view};
}

bool PointCloudLayout::IsDistributed() const {
  return false;
}

size_t PointCloudLayout::GetNumEnts() const
{
  return coords_.extent(0);
}

EntOffsetsArray PointCloudLayout::GetEntOffsets() const
{
  EntOffsetsArray offsets{};
  for (size_t i = 0; i < offsets.size(); ++i)
    offsets[i] = coords_.extent(0);
  offsets[0] = 0;
  return offsets;
}

std::array<int, 4> PointCloudLayout::GetNodesPerDim() const
{
  std::array<int, 4> nodes{};
  for (size_t i = 0; i < nodes.size(); ++i)
    nodes[i] = 0;
  nodes[0] = 1;
  return nodes;
}

Kokkos::View<const Real**, HostMemorySpace>
PointCloudLayout::GetCoordinatesHost() const
{
  return coords_host_;
}

int PointCloudLayout::GetDimension() const
{
  return dim_;
}

Rank1View<const LO, HostMemorySpace>
PointCloudLayout::GetDOFHolderClassificationDimensions() const
{
  return make_const_array_view(classification_dims_host_);
}

Rank1View<const LO, HostMemorySpace>
PointCloudLayout::GetDOFHolderClassificationIds() const
{
  return make_const_array_view(classification_ids_host_);
}

} // namespace pcms
