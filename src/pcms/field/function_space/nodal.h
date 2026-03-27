#ifndef PCMS_NODAL_FIELD_FACTORY_H
#define PCMS_NODAL_FIELD_FACTORY_H

#include "pcms/field/field.h"
#include "pcms/field/field_layout.h"
#include "pcms/field/field_data.h"
#include "pcms/field/field_metadata.h"
#include "pcms/field/function_space.h"
#include "pcms/field/out_of_bounds_policy.h"
#include "pcms/field/coordinate_system.h"
#include "pcms/field/evaluator/mls_options.h"
#include "pcms/utility/arrays.h"
#include "pcms/utility/memory_spaces.h"

#include <memory>

namespace pcms
{

template <typename T>
class FieldEvaluatorFactory;

class NodalFunctionSpace : public FunctionSpace
{
public:
  [[nodiscard]] static NodalFunctionSpace Create(
    Rank2View<Real, HostMemorySpace> coords,
    CoordinateSystem coordinate_system,
    MLSOptions options = {});

  [[nodiscard]] std::shared_ptr<const FieldLayout> GetLayout() const noexcept override;

  [[nodiscard]] CoordinateSystem GetCoordinateSystem() const noexcept override;

protected:
  [[nodiscard]] FieldVariant CreateFieldImpl(
    Type value_type, FieldMetadata metadata) const override;

  [[nodiscard]] FieldVariant CreateFieldImpl(FieldDataVariant data) const override;

  [[nodiscard]] PointEvaluatorVariant CreatePointEvaluatorImpl(
    Type value_type,
    CoordinateView<HostMemorySpace> coords,
    OutOfBoundsPolicy policy) const override;

private:
  explicit NodalFunctionSpace(
    std::shared_ptr<const FieldLayout> layout,
    std::shared_ptr<FieldEvaluatorFactory<Real>> evaluator_factory) noexcept;

  std::shared_ptr<const FieldLayout> layout_;
  std::shared_ptr<FieldEvaluatorFactory<Real>> evaluator_factory_;
};

} // namespace pcms

#endif // PCMS_NODAL_FIELD_FACTORY_H
