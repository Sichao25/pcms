#ifndef PCMS_OUT_OF_BOUNDS_POLICY_H
#define PCMS_OUT_OF_BOUNDS_POLICY_H

#include "field.h"  // for OutOfBoundsMode (ERROR, FILL, NEAREST_BOUNDARY)
#include "pcms/utility/types.h"

namespace pcms
{

// OutOfBoundsPolicy wraps the existing OutOfBoundsMode enum (defined in
// field.h) with an optional fill value into a single construction-time policy
// object. This replaces the SetOutOfBoundsMode() / GetOutOfBoundsMode()
// pattern on FieldT<T>: out-of-bounds behavior is now a policy passed when
// creating a PointEvaluator rather than mutable state on the field object.
//
// NearestBoundary is an optional backend capability. Call
// FieldEvaluatorFactory::SupportsNearestBoundary() before requesting it.
// Backends that do not support it throw a descriptive error if it is requested.
// The policy is fixed when the PointEvaluator is created; it is not mutable
// after that evaluator has been constructed.
//
// NOTE: OutOfBoundsMode itself is defined in field.h with values ERROR, FILL,
// and NEAREST_BOUNDARY. Those names are retained for backward compatibility
// during the incremental migration to this API.

struct OutOfBoundsPolicy
{
  OutOfBoundsMode mode       = OutOfBoundsMode::ERROR;
  Real            fill_value = 0.0;  // used only when mode == FILL
};

} // namespace pcms

#endif // PCMS_OUT_OF_BOUNDS_POLICY_H
