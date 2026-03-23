#include "pcms/field/nodal_field_factory.h"
#include "pcms/field/adapter/point_cloud/point_cloud_layout.h"
#include "pcms/field/adapter/point_cloud/point_cloud_evaluator_factory.h"
#include "pcms/utility/assert.h"

#include <Kokkos_Core.hpp>

namespace pcms
{

NodalFieldFactory::NodalFieldFactory(
  std::shared_ptr<const FieldLayout> layout,
  std::shared_ptr<FieldEvaluatorFactory<Real>> evaluator_factory) noexcept
  : layout_(std::move(layout)),
    evaluator_factory_(std::move(evaluator_factory))
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
  auto eval_factory = std::make_shared<PointCloudEvaluatorFactory>(pc_layout);
  return NodalFieldFactory(pc_layout, std::move(eval_factory));
}

std::shared_ptr<const FieldLayout>
NodalFieldFactory::GetLayout() const noexcept
{
  return layout_;
}

std::unique_ptr<PointEvaluator<Real>> NodalFieldFactory::CreatePointEvaluator(
  CoordinateView<HostMemorySpace> coords,
  OutOfBoundsPolicy policy) const
{
  PCMS_ALWAYS_ASSERT(evaluator_factory_ != nullptr);
  return evaluator_factory_->CreatePointEvaluator(coords, policy);
}

std::unique_ptr<FieldData<Real>> NodalFieldFactory::CreateFieldData(
  FieldMetadata metadata) const
{
  return std::make_unique<SimpleFieldData<Real>>(layout_, metadata);
}

} // namespace pcms
