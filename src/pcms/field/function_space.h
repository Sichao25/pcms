#ifndef PCMS_FUNCTION_SPACE_H
#define PCMS_FUNCTION_SPACE_H

#include "coordinate_system.h"
#include "field_evaluator_factory.h"
#include "field_layout.h"
#include "pcms/utility/types.h"
#include <memory>

namespace pcms
{

// FunctionSpace is an abstract interface representing an evaluatable field
// space: layout, evaluation rules, and coordinate interpretation.
//
// Concrete implementations (e.g. LagrangeFunctionSpace) provide backends for
// specific discretizations or mesh types.
//
// FunctionSpace is used as the parameter type for operation objects such as
// Interpolator<T>, so that operations are not coupled to a specific backend.
class FunctionSpace
{
public:
  virtual std::shared_ptr<const FieldLayout>
  GetLayout() const noexcept = 0;

  virtual const FieldEvaluatorFactory<Real>&
  GetEvaluatorFactory() const noexcept = 0;

  virtual CoordinateSystem GetCoordinateSystem() const noexcept = 0;

  virtual ~FunctionSpace() noexcept = default;
};

} // namespace pcms

#endif // PCMS_FUNCTION_SPACE_H
