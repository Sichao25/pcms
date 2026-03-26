#ifndef PCMS_SIMPLE_FIELD_DATA_H
#define PCMS_SIMPLE_FIELD_DATA_H

#include "../field_data.h"
#include "../field_layout.h"
#include "../field_metadata.h"
#include "pcms/utility/arrays.h"
#include "pcms/utility/assert.h"
#include "pcms/utility/memory_spaces.h"
#include <Kokkos_Core.hpp>
#include <memory>
#include <string>

namespace pcms
{

// SimpleFieldData<T> is a generic concrete FieldData<T> backed by a flat
// Kokkos::View<T*, HostMemorySpace>. It works for any backend whose DOF data
// is a flat coefficient array (OmegaH, UniformGrid, PointCloud, etc.).
//
// Ownership of the layout is shared — the layout is typically held by the
// factory that created this field data object.
template <typename T>
class SimpleFieldData : public FieldData<T>
{
public:
  SimpleFieldData(std::shared_ptr<const FieldLayout> layout,
                  FieldMetadata metadata)
    : layout_(std::move(layout)),
      metadata_(metadata),
      data_("simple_field_data",
            static_cast<size_t>(layout_->OwnedSize()))
#if defined(PCMS_HAS_DISTINCT_DEVICE_MEMORY_SPACE)
      ,
      device_data_("simple_field_data_device",
                   static_cast<size_t>(layout_->OwnedSize()))
#endif
  {
  }

  const FieldLayout& GetLayout() const override { return *layout_; }

  const FieldMetadata& GetMetadata() const override { return metadata_; }

  Rank1View<const T, HostMemorySpace> GetDOFHolderDataHost() const override
  {
    return make_const_array_view(data_);
  }

  void SetDOFHolderDataHost(
    Rank1View<const T, HostMemorySpace> values) override
  {
    PCMS_ALWAYS_ASSERT(values.size() ==
                       static_cast<size_t>(layout_->OwnedSize()));
    for (size_t i = 0; i < values.size(); ++i) {
      data_(i) = values[i];
    }
  }

#if defined(PCMS_HAS_DISTINCT_DEVICE_MEMORY_SPACE)
  Rank1View<const T, DeviceMemorySpace> GetDOFHolderDataDevice() const override
  {
    // Keep a persistent device allocation so the returned view remains valid
    // until this object is mutated or destroyed.
    Kokkos::deep_copy(device_data_, data_);
    return make_const_array_view(device_data_);
  }

  void SetDOFHolderDataDevice(
    Rank1View<const T, DeviceMemorySpace> values) override
  {
    PCMS_ALWAYS_ASSERT(values.size() ==
                       static_cast<size_t>(layout_->OwnedSize()));
    Kokkos::deep_copy(device_data_, values);
    Kokkos::deep_copy(data_, values);
  }
#endif

private:
  std::shared_ptr<const FieldLayout> layout_;
  FieldMetadata metadata_;
  Kokkos::View<T*, HostMemorySpace> data_;
#if defined(PCMS_HAS_DISTINCT_DEVICE_MEMORY_SPACE)
  mutable Kokkos::View<T*, DeviceMemorySpace> device_data_;
#endif
};

} // namespace pcms

#endif // PCMS_SIMPLE_FIELD_DATA_H
