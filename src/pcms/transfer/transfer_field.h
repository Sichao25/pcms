#ifndef PCMS_COUPLING_TRANSFER_FIELD_H
#define PCMS_COUPLING_TRANSFER_FIELD_H
#include <utility>
#include "pcms/utility/arrays.h"
#include "pcms/coupler/adapter/field_evaluation_methods.h"
#include "pcms/utility/profile.h"

namespace pcms
{

/**
 * Pointwise copy of data in source_field to target field. Source and target
 * fields must have the same size and implicit field iteration order
 */
template <typename Field>
void copy_field(const Field& source_field, Field& target_field)
{
  PCMS_FUNCTION_TIMER;
  const auto source_data = get_nodal_data(source_field);
  set_nodal_data(target_field, make_array_view(source_data));
}

template <typename SourceField, typename TargetField,
          typename EvaluationMethod = Lagrange<1>>
void interpolate_field(const SourceField& source_field,
                       TargetField& target_field, EvaluationMethod method = {})
{
  PCMS_FUNCTION_TIMER;
  auto coordinates = get_nodal_coordinates(target_field);
  auto coordinates_view = make_const_array_view(coordinates);
  const auto data = evaluate(source_field, method, coordinates_view);
  set_nodal_data(target_field, make_array_view(data));
}

} // namespace pcms

#endif // PCMS_COUPLING_TRANSFER_FIELD_H
