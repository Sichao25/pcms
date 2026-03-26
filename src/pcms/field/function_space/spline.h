#ifndef PCMS_SPLINE_FUNCTION_SPACE_H
#define PCMS_SPLINE_FUNCTION_SPACE_H

#include "pcms/field/data/simple.h"
#include "pcms/field/field.h"
#include "pcms/field/field_data.h"
#include "pcms/field/field_evaluator_factory.h"
#include "pcms/field/field_layout.h"
#include "pcms/field/field_metadata.h"
#include "pcms/field/function_space.h"
#include "pcms/field/out_of_bounds_policy.h"
#include "pcms/field/point_evaluator.h"
#include "pcms/utility/arrays.h"
#include "pcms/utility/memory_spaces.h"
#include "pcms/utility/uniform_grid.h"

#include <memory>

namespace pcms
{

class SplineFunctionSpace : public FunctionSpace
{
public:
  [[nodiscard]] static SplineFunctionSpace FromUniformGrid(
    const UniformGrid<2>& grid,
    CoordinateSystem coordinate_system);

  [[nodiscard]] std::shared_ptr<const FieldLayout> GetLayout() const noexcept override;

  [[nodiscard]] CoordinateSystem GetCoordinateSystem() const noexcept override;

  [[nodiscard]] const FieldEvaluatorFactory<Real>& GetEvaluatorFactory() const override;

  [[nodiscard]] std::unique_ptr<PointEvaluator<Real>> CreatePointEvaluator(
    CoordinateView<HostMemorySpace> coords,
    OutOfBoundsPolicy policy = {}) const;

  [[nodiscard]] Field<Real> CreateField(FieldMetadata metadata = {}) const;

  [[nodiscard]] std::unique_ptr<FieldData<Real>> CreateFieldData(
    FieldMetadata metadata = {}) const;

private:
  explicit SplineFunctionSpace(
    std::shared_ptr<const FieldLayout> layout,
    std::shared_ptr<FieldEvaluatorFactory<Real>> evaluator_factory) noexcept;

  std::shared_ptr<const FieldLayout> layout_;
  std::shared_ptr<FieldEvaluatorFactory<Real>> evaluator_factory_;
};

} // namespace pcms

#endif // PCMS_SPLINE_FUNCTION_SPACE_H
