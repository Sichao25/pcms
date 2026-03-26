#ifndef PCMS_NODAL_FIELD_FACTORY_H
#define PCMS_NODAL_FIELD_FACTORY_H

#include "pcms/field/field.h"
#include "pcms/field/field_layout.h"
#include "pcms/field/field_data.h"
#include "pcms/field/field_metadata.h"
#include "pcms/field/function_space.h"
#include "pcms/field/out_of_bounds_policy.h"
#include "pcms/field/data/simple.h"
#include "pcms/field/coordinate_system.h"
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
    CoordinateSystem coordinate_system);

  [[nodiscard]] std::shared_ptr<const FieldLayout> GetLayout() const noexcept override;

  [[nodiscard]] CoordinateSystem GetCoordinateSystem() const noexcept override;

  [[nodiscard]] const FieldEvaluatorFactory<Real>& GetEvaluatorFactory() const override;

  // Delegates PointEvaluator construction to the internal factory.
  [[nodiscard]] std::unique_ptr<PointEvaluator<Real>> CreatePointEvaluator(
    CoordinateView<HostMemorySpace> coords,
    OutOfBoundsPolicy policy = {}) const;

  // Returns a new Field<Real> for this function space.
  // Prefer this over CreateFieldData() in new code.
  [[nodiscard]] Field<Real> CreateField(FieldMetadata metadata = {}) const;

  // Low-level: bare FieldData without the evaluator factory reference.
  // Use when the communicator API requires a raw FieldData; prefer
  // CreateField() in user code.
  [[nodiscard]] std::unique_ptr<FieldData<Real>> CreateFieldData(
    FieldMetadata metadata = {}) const;

private:

  explicit NodalFunctionSpace(
    std::shared_ptr<const FieldLayout> layout,
    std::shared_ptr<FieldEvaluatorFactory<Real>> evaluator_factory) noexcept;

  std::shared_ptr<const FieldLayout> layout_;
  std::shared_ptr<FieldEvaluatorFactory<Real>> evaluator_factory_;
};

} // namespace pcms

#endif // PCMS_NODAL_FIELD_FACTORY_H
