#include "pcms/field/function_space/spline.h"

#include "pcms/field/evaluator/uniform_grid_spline.h"
#include "pcms/field/layout/uniform_grid.h"

#include <stdexcept>

namespace pcms
{

SplineFunctionSpace::SplineFunctionSpace(
  std::shared_ptr<const FieldLayout> layout,
  std::shared_ptr<FieldEvaluatorFactory<Real>> evaluator_factory) noexcept
  : layout_(std::move(layout)),
    evaluator_factory_(std::move(evaluator_factory))
{
}

SplineFunctionSpace SplineFunctionSpace::FromUniformGrid(
  const UniformGrid<2>& grid,
  CoordinateSystem coordinate_system)
{
  auto layout = std::make_shared<UniformGridFieldLayout<2>>(
    grid, 1, coordinate_system, 1);
  auto evaluator_factory =
    std::make_shared<UniformGridSplineEvaluatorFactory2D>(layout);
  return SplineFunctionSpace(
    layout, std::move(evaluator_factory));
}

std::shared_ptr<const FieldLayout>
SplineFunctionSpace::GetLayout() const noexcept
{
  return layout_;
}

CoordinateSystem SplineFunctionSpace::GetCoordinateSystem() const noexcept
{
  return evaluator_factory_->GetCoordinateSystem();
}

const FieldEvaluatorFactory<Real>& SplineFunctionSpace::GetEvaluatorFactory()
  const
{
  return *evaluator_factory_;
}

std::unique_ptr<PointEvaluator<Real>> SplineFunctionSpace::CreatePointEvaluator(
  CoordinateView<HostMemorySpace> coords,
  OutOfBoundsPolicy policy) const
{
  if (!evaluator_factory_) {
    throw pcms_error(
      "SplineFunctionSpace::CreatePointEvaluator: evaluator construction is "
      "not available for this backend");
  }
  return evaluator_factory_->CreatePointEvaluator(coords, policy);
}

Field<Real> SplineFunctionSpace::CreateField(FieldMetadata metadata) const
{
  return Field<Real>(layout_, evaluator_factory_, CreateFieldData(metadata));
}

std::unique_ptr<FieldData<Real>> SplineFunctionSpace::CreateFieldData(
  FieldMetadata metadata) const
{
  return std::make_unique<SimpleFieldData<Real>>(layout_, metadata);
}

} // namespace pcms
