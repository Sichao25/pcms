#ifndef PCMS_POINT_EVALUATOR_H
#define PCMS_POINT_EVALUATOR_H

#include "field_data.h"
#include "pcms/utility/arrays.h"
#include "pcms/utility/memory_spaces.h"

namespace pcms
{

// PointEvaluator<T> is bound to a specific set of query points. Created by
// calling CreatePointEvaluator(coords) on either a concrete field factory or a
// FieldEvaluatorFactory directly. Performs and caches all backend-specific
// localization work (spatial search, barycentric coordinate computation, etc.)
// at construction time. Evaluate may then be called repeatedly for different
// FieldData objects at zero additional localization cost.
//
// Replaces the previous design's FieldEvaluator + EvaluationCache pair.
// LocalizationHint from older code maps to the internal state of PointEvaluator.
//
// FieldData compatibility with a PointEvaluator is a precondition of Evaluate.
// Implementations should check this with a cheap backend-specific identity
// check rather than deep structural comparison and should throw pcms_error on
// mismatch.
//
// Evaluate writes results into a rank-2 output view with shape:
//   [num_query_points][num_components]
// The caller must provide the full output buffer. Successful Evaluate calls
// fill the entire buffer.
template <typename T>
class PointEvaluator
{
public:
  virtual void Evaluate(const FieldData<T>& field,
                        Rank2View<T, HostMemorySpace> values) const = 0;

#if defined(PCMS_HAS_DISTINCT_DEVICE_MEMORY_SPACE)
  virtual void Evaluate(const FieldData<T>& field,
                        Rank2View<T, DeviceMemorySpace> values) const = 0;
#endif

  virtual ~PointEvaluator() noexcept = default;
};

} // namespace pcms

#endif // PCMS_POINT_EVALUATOR_H
