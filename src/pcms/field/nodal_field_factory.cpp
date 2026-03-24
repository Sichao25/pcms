#include "pcms/field/nodal_field_factory.h"
#include "pcms/field/adapter/point_cloud/point_cloud_layout.h"
#include "pcms/field/adapter/point_cloud/point_cloud_evaluator_factory.h"
#include "pcms/utility/assert.h"

#include <Kokkos_Core.hpp>

namespace pcms
{

NodalFunctionSpace::NodalFunctionSpace(
  std::shared_ptr<const FieldLayout> layout,
  std::shared_ptr<FieldEvaluatorFactory<Real>> evaluator_factory) noexcept
  : layout_(std::move(layout)),
    evaluator_factory_(std::move(evaluator_factory))
{
}

NodalFunctionSpace NodalFunctionSpace::Create(
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
  return NodalFunctionSpace(pc_layout, std::move(eval_factory));
}

std::shared_ptr<const FieldLayout>
NodalFunctionSpace::GetLayout() const noexcept
{
  return layout_;
}

CoordinateSystem NodalFunctionSpace::GetCoordinateSystem() const noexcept
{
  return evaluator_factory_->GetCoordinateSystem();
}

const FieldEvaluatorFactory<Real>&
NodalFunctionSpace::GetEvaluatorFactory() const
{
  return *evaluator_factory_;
}

std::unique_ptr<PointEvaluator<Real>> NodalFunctionSpace::CreatePointEvaluator(
  CoordinateView<HostMemorySpace> coords,
  OutOfBoundsPolicy policy) const
{
  PCMS_ALWAYS_ASSERT(evaluator_factory_ != nullptr);
  return evaluator_factory_->CreatePointEvaluator(coords, policy);
}

Field<Real> NodalFunctionSpace::CreateField(FieldMetadata metadata) const
{
  return Field<Real>(evaluator_factory_, CreateFieldData(metadata));
}

std::unique_ptr<FieldData<Real>> NodalFunctionSpace::CreateFieldData(
  FieldMetadata metadata) const
{
  return std::make_unique<SimpleFieldData<Real>>(layout_, metadata);
}

} // namespace pcms
