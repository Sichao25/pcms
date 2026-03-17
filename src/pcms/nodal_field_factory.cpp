#include "pcms/nodal_field_factory.h"
#include "pcms/adapter/point_cloud/point_cloud_layout.h"

#include <Kokkos_Core.hpp>

namespace pcms
{

NodalFieldFactory::NodalFieldFactory(
  std::shared_ptr<const FieldLayout> layout) noexcept
  : layout_(std::move(layout))
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
  return NodalFieldFactory(
    std::make_shared<PointCloudLayout>(dim, device_view, coordinate_system));
}

std::shared_ptr<const FieldLayout>
NodalFieldFactory::GetLayout() const noexcept
{
  return layout_;
}

std::unique_ptr<FieldT<Real>> NodalFieldFactory::CreateFieldReal() const
{
  return layout_->CreateFieldReal();
}

} // namespace pcms
