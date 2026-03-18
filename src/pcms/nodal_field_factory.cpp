#include "pcms/nodal_field_factory.h"
#include "pcms/adapter/point_cloud/point_cloud_layout.h"
#include "pcms/adapter/point_cloud/point_cloud.h"

#include <Kokkos_Core.hpp>

namespace pcms
{

NodalFieldFactory::NodalFieldFactory(
  std::shared_ptr<const FieldLayout> layout,
  std::function<std::unique_ptr<FieldT<Real>>()> create_fn) noexcept
  : layout_(std::move(layout)), create_fn_(std::move(create_fn))
{
}

NodalFieldFactory NodalFieldFactory::Create(
  Rank2View<Real, HostMemorySpace> coords,
  CoordinateSystem coordinate_system)
{
  int dim = static_cast<int>(coords.extent(1));
  Kokkos::View<Real**, Kokkos::HostSpace> host_view(
    coords.data_handle(), coords.extent(0), coords.extent(1));
  auto device_view =
    Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace{}, host_view);
  auto pc_layout =
    std::make_shared<PointCloudLayout>(dim, device_view, coordinate_system);
  return NodalFieldFactory(
    pc_layout,
    [pc_layout]() { return std::make_unique<PointCloud>(pc_layout); });
}

std::shared_ptr<const FieldLayout>
NodalFieldFactory::GetLayout() const noexcept
{
  return layout_;
}

std::unique_ptr<FieldT<Real>> NodalFieldFactory::CreateFieldReal() const
{
  return create_fn_();
}

} // namespace pcms
