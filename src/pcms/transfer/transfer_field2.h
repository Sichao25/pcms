#ifndef PCMS_TRANSFER_FIELD2_H_
#define PCMS_TRANSFER_FIELD2_H_
#include "pcms/field/field_data.h"
#include "pcms/field/field_evaluator_factory.h"
#include "pcms/field/out_of_bounds_policy.h"
#include "pcms/field/point_evaluator.h"
#include "pcms/utility/assert.h"
#include "pcms/utility/arrays.h"
#include "pcms/utility/memory_spaces.h"
#include "pcms/utility/profile.h"
#include <Kokkos_Core.hpp>

namespace pcms
{

namespace detail
{

inline bool CompatibleMetadata(const FieldMetadata& source,
                               const FieldMetadata& target) noexcept
{
  return source.value_type == target.value_type &&
         source.value_coordinate_system == target.value_coordinate_system;
}

template <typename T>
void CheckCopyCompatible(const FieldData<T>& source, const FieldData<T>& target)
{
  if (&source.GetLayout() != &target.GetLayout()) {
    throw pcms_error("copy_field2: source and target layouts differ");
  }
  if (!CompatibleMetadata(source.GetMetadata(), target.GetMetadata())) {
    throw pcms_error("copy_field2: source and target metadata differ");
  }
}

template <typename T>
void CheckInterpolateCompatible(const FieldData<T>& source_data,
                                const FieldEvaluatorFactory<T>& source_factory,
                                const FieldData<T>& target_data)
{
  if (&source_data.GetLayout() != &source_factory.GetLayout()) {
    throw pcms_error(
      "interpolate_field2: source FieldData layout does not match the source "
      "evaluation factory");
  }
  if (source_data.GetLayout().GetNumComponents() !=
      target_data.GetLayout().GetNumComponents()) {
    throw pcms_error(
      "interpolate_field2: source and target component counts differ");
  }
  if (!CompatibleMetadata(source_data.GetMetadata(),
                          target_data.GetMetadata())) {
    throw pcms_error("interpolate_field2: source and target metadata differ");
  }
}

} // namespace detail

template <typename T>
void copy_field2(const FieldData<T>& source, FieldData<T>& target)
{
  PCMS_FUNCTION_TIMER;
  detail::CheckCopyCompatible(source, target);
  target.SetDOFHolderDataHost(source.GetDOFHolderDataHost());
}

template <typename T>
void interpolate_field2(const FieldData<T>& source_data,
                        const FieldEvaluatorFactory<T>& source_factory,
                        FieldData<T>& target_data,
                        OutOfBoundsPolicy policy = {})
{
  PCMS_FUNCTION_TIMER;
  detail::CheckInterpolateCompatible(source_data, source_factory, target_data);
  auto coords = target_data.GetLayout().GetDOFHolderCoordinates();
  auto evaluator = source_factory.CreatePointEvaluator(coords, policy);
  const LO num_points = coords.GetCoordinates().extent(0);
  const int n_comp = target_data.GetLayout().GetNumComponents();
  Kokkos::View<T**, HostMemorySpace> output("interpolated", num_points, n_comp);
  Rank2View<T, HostMemorySpace> output_view(output.data(), num_points, n_comp);
  evaluator->Evaluate(source_data, output_view);
  Kokkos::View<T*, HostMemorySpace> flat("flat",
                                         target_data.GetLayout().OwnedSize());
  Kokkos::parallel_for(
    Kokkos::RangePolicy<HostMemorySpace::execution_space>(0, num_points),
    KOKKOS_LAMBDA(LO i) {
      for (int c = 0; c < n_comp; ++c) {
        flat(i * n_comp + c) = output(i, c);
      }
    });
  target_data.SetDOFHolderDataHost(make_const_array_view(flat));
}

} // namespace pcms

#endif // PCMS_TRANSFER_FIELD2_H_
