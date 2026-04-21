#ifndef PCMS_FIELD_SERIALIZER_H
#define PCMS_FIELD_SERIALIZER_H

#include "pcms/field/field.h"
#include "pcms/utility/arrays.h"
#include "pcms/utility/memory_spaces.h"
#include "pcms/utility/types.h"
#include <Kokkos_Core.hpp>

namespace pcms
{

template <typename T>
class FieldSerializer
{
public:
  virtual int Serialize(const FieldData<T>& field,
                        const FieldLayout& layout,
                        Rank1View<T, HostMemorySpace> buffer,
                        Rank1View<const LO, HostMemorySpace> permutation) const
  {
    auto data = field.GetDOFHolderDataHost();
    auto owned = layout.GetOwnedHost();
    if (buffer.size() > 0) {
      for (LO i = 0; i < static_cast<LO>(data.size()); ++i) {
        if (owned[i])
          buffer[permutation[i]] = data[i];
      }
    }
    return static_cast<int>(data.size());
  }

  virtual void Deserialize(
    FieldData<T>& field, const FieldLayout& layout,
    Rank1View<const T, HostMemorySpace> buffer,
    Rank1View<const LO, HostMemorySpace> permutation) const
  {
    Kokkos::View<T*, HostMemorySpace> sorted("sorted", permutation.size());
    auto owned = layout.GetOwnedHost();
    for (LO i = 0; i < static_cast<LO>(sorted.size()); ++i) {
      if (owned[i])
        sorted[i] = buffer[permutation[i]];
    }
    field.SetDOFHolderDataHost(make_const_array_view(sorted));
  }

  virtual ~FieldSerializer() noexcept = default;
};

} // namespace pcms

#endif // PCMS_FIELD_SERIALIZER_H
