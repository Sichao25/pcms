#include "pcms/field/layout/point_cloud.h"
#include "pcms/utility/arrays.h"
#include <memory>
#include <Kokkos_StdAlgorithms.hpp>

namespace pcms
{

namespace
{

std::shared_ptr<const Discretization> MakePointCloudDiscretization(
  int dim, Kokkos::View<Real**> coords)
{
  auto coords_mirror = Kokkos::create_mirror_view_and_copy(HostMemorySpace(), coords);
  // Create a view with the default layout for HostMemorySpace to avoid layout incompatibility
  Kokkos::View<Real**, HostMemorySpace> coords_host("coords_host", coords_mirror.extent(0), coords_mirror.extent(1));
  Kokkos::deep_copy(coords_host, coords_mirror);
  return std::make_shared<PointCloudDiscretization>(
    dim, coords_host, static_cast<const void*>(coords.data()));
}

void InitializePointCloudClassification(
  Kokkos::View<LO*, HostMemorySpace> classification_dims_host,
  Kokkos::View<LO*, HostMemorySpace> classification_ids_host,
  LO n, LO classification_entity_dim)
{
  Kokkos::deep_copy(classification_dims_host, classification_entity_dim);
  for (LO i = 0; i < n; ++i) {
    classification_ids_host(i) = i;
  }
}

} // namespace

PointCloudLayout::PointCloudLayout(
  int dim, Kokkos::View<Real**> coords, CoordinateSystem coordinate_system)
  : PointCloudLayout(
      dim, coords, coordinate_system, MakePointCloudDiscretization(dim, coords),
      Vertex)
{
}

PointCloudLayout::PointCloudLayout(
  int dim, Kokkos::View<Real**> coords, CoordinateSystem coordinate_system,
  std::shared_ptr<const Discretization> discretization,
  int classification_entity_dim)
  : dim_(dim),
    coordinate_system_(coordinate_system),
    coords_(coords),
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
  InitializePointCloudClassification(classification_dims_host_,
                                     classification_ids_host_, n,
                                     classification_entity_dim);
  discretization_ = std::move(discretization);
}

std::shared_ptr<const Discretization>
PointCloudLayout::GetDiscretization() const noexcept
{
  return discretization_;
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

CoordinateView<DeviceMemorySpace> PointCloudLayout::GetDOFHolderCoordinates()
  const
{
  auto coords_view = MakeConstRank2View(coords_);
  return CoordinateView<DeviceMemorySpace>{coordinate_system_, coords_view};
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

Kokkos::View<const Real**, DeviceMemorySpace>
PointCloudLayout::GetCoordinates() const
{
  return coords_;
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
