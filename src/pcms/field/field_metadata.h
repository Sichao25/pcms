#ifndef PCMS_FIELD_METADATA_H
#define PCMS_FIELD_METADATA_H

#include "coordinate_system.h"

namespace pcms
{

enum class FieldValueType
{
  Scalar,
  Vector,
  Tensor
  // Tensor variance and transformation rules are intentionally deferred to a
  // future richer metadata model.
};

struct FieldMetadata
{
  FieldValueType value_type = FieldValueType::Scalar;
  // The coordinate system that field values are expressed in. Evaluate always
  // returns values in this system and does not transform them. Callers are
  // responsible for any value transformation after Evaluate (coordinate
  // transform wiring is deferred).
  CoordinateSystem value_coordinate_system = CoordinateSystem::Cartesian;
};

} // namespace pcms

#endif // PCMS_FIELD_METADATA_H
