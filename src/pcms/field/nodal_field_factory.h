#ifndef PCMS_NODAL_FIELD_FACTORY_H
#define PCMS_NODAL_FIELD_FACTORY_H

#include "field.h"
#include "pcms/field/field_layout.h"
#include "pcms/field/field_data.h"
#include "pcms/field/field_metadata.h"
#include "pcms/field/point_evaluator.h"
#include "pcms/field/out_of_bounds_policy.h"
#include "pcms/field/simple_field_data.h"
#include "coordinate_system.h"
#include "pcms/utility/arrays.h"
#include "pcms/utility/memory_spaces.h"

#include <functional>
#include <memory>

namespace pcms
{

template <typename T>
class FieldEvaluatorFactory;

class NodalFieldFactory
{
public:
  [[nodiscard]] static NodalFieldFactory Create(
    Rank2View<Real, HostMemorySpace> coords,
    CoordinateSystem coordinate_system);

  [[nodiscard]] std::shared_ptr<const FieldLayout> GetLayout() const noexcept;
  [[nodiscard]] std::unique_ptr<FieldT<Real>> CreateFieldReal() const;

  // New evaluation API — added alongside the existing API for incremental
  // migration. This factory is primarily a construction convenience; routine
  // data access should go through the resulting FieldData and FieldLayout
  // objects.
  [[nodiscard]] std::unique_ptr<PointEvaluator<Real>> CreatePointEvaluator(
    CoordinateView<HostMemorySpace> coords,
    OutOfBoundsPolicy policy = {}) const;
  [[nodiscard]] std::unique_ptr<FieldData<Real>> CreateFieldData(
    FieldMetadata metadata = {}) const;

private:
  explicit NodalFieldFactory(
    std::shared_ptr<const FieldLayout> layout,
    std::function<std::unique_ptr<FieldT<Real>>()> create_fn,
    std::shared_ptr<FieldEvaluatorFactory<Real>> evaluator_factory) noexcept;

  std::shared_ptr<const FieldLayout> layout_;
  std::function<std::unique_ptr<FieldT<Real>>()> create_fn_;
  std::shared_ptr<FieldEvaluatorFactory<Real>> evaluator_factory_;
};

} // namespace pcms

#endif // PCMS_NODAL_FIELD_FACTORY_H
