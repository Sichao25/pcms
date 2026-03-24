#ifndef PCMS_TRANSFER_FIELD2_H_
#define PCMS_TRANSFER_FIELD2_H_
#include "pcms/field/field_data.h"
#include "pcms/utility/assert.h"
#include "pcms/utility/profile.h"

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

} // namespace detail

template <typename T>
void copy_field2(const FieldData<T>& source, FieldData<T>& target)
{
  PCMS_FUNCTION_TIMER;
  detail::CheckCopyCompatible(source, target);
  target.SetDOFHolderDataHost(source.GetDOFHolderDataHost());
}

} // namespace pcms

#endif // PCMS_TRANSFER_FIELD2_H_
