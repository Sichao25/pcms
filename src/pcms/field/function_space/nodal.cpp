#include "pcms/field/function_space/nodal.h"
#include "pcms/field/layout/point_cloud.h"
#include "pcms/field/evaluator/point_cloud.h"
#include "pcms/field/data/simple.h"
#include "pcms/utility/assert.h"
#include "pcms/utility/common.h"

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

FieldVariant NodalFunctionSpace::CreateFieldImpl(
  Type value_type, FieldMetadata metadata) const
{
  return apply_to_type(value_type, [&](auto tag) -> FieldVariant {
    using T = typename decltype(tag)::type;
    if constexpr (!std::is_same_v<T, double>) {
      throw pcms_error("NodalFunctionSpace: only double (Real) is supported");
    }
    else {
      return Field<double>(layout_, evaluator_factory_,
                           std::make_unique<SimpleFieldData<double>>(layout_,
                                                                      metadata));
    }
  });
}

FieldVariant NodalFunctionSpace::CreateFieldImpl(FieldDataVariant data) const
{
  if (!std::holds_alternative<std::unique_ptr<FieldData<double>>>(data)) {
    throw pcms_error("NodalFunctionSpace: only double (Real) is supported");
  }
  auto fd = std::move(std::get<std::unique_ptr<FieldData<double>>>(data));
  PCMS_ALWAYS_ASSERT(fd != nullptr);
  if (dynamic_cast<const SimpleFieldData<double>*>(fd.get()) == nullptr) {
    throw pcms_error(
      "NodalFunctionSpace::CreateField: requires SimpleFieldData<double>");
  }
  if (fd->GetDOFHolderDataHost().size() !=
      detail::ExpectedFlatFieldDataSize(*layout_)) {
    throw pcms_error(
      "NodalFunctionSpace::CreateField: field data size does not match "
      "layout");
  }
  return Field<double>(layout_, evaluator_factory_, std::move(fd));
}

PointEvaluatorVariant NodalFunctionSpace::CreatePointEvaluatorImpl(
  Type value_type,
  CoordinateView<HostMemorySpace> coords,
  OutOfBoundsPolicy policy) const
{
  if (value_type != Type::Real) {
    throw pcms_error(
      "NodalFunctionSpace: point evaluation only supports double (Real)");
  }
  PCMS_ALWAYS_ASSERT(evaluator_factory_ != nullptr);
  return evaluator_factory_->CreatePointEvaluator(coords, policy);
}

} // namespace pcms
