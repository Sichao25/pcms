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
      host_data_("simple_field_data",
                 static_cast<size_t>(layout_->OwnedSize())),
      device_data_("simple_field_data_device",
                   static_cast<size_t>(layout_->OwnedSize()))
  {
  }

  const FieldMetadata& GetMetadata() const override { return metadata_; }

  Rank1View<const T, HostMemorySpace> GetDOFHolderDataHost() const override
  {
    Kokkos::deep_copy(host_data_, device_data_);
    return make_const_array_view(host_data_);
  }

  void SetDOFHolderDataHost(Rank1View<const T, HostMemorySpace> values) override
  {
    PCMS_ALWAYS_ASSERT(values.size() ==
                       static_cast<size_t>(layout_->OwnedSize()));
    CopyHostRank1ViewToDeviceView(device_data_, values);
  }

  Rank1View<const T, DeviceMemorySpace> GetDOFHolderData() const override
  {
    return make_const_array_view(device_data_);
  }

  void SetDOFHolderData(Rank1View<const T, DeviceMemorySpace> values) override
  {
    PCMS_ALWAYS_ASSERT(values.size() ==
                       static_cast<size_t>(layout_->OwnedSize()));
    CopyDeviceRank1ViewToDeviceView(device_data_, values);
  }

private:
  std::shared_ptr<const FieldLayout> layout_;
  FieldMetadata metadata_;
  mutable Kokkos::View<T*, HostMemorySpace> host_data_;
  Kokkos::View<T*, DeviceMemorySpace> device_data_;
};

} // namespace pcms

#endif // PCMS_SIMPLE_FIELD_DATA_H
