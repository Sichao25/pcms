#ifndef PCMS_ADAPTER_MESHFIELDS_MESH_FIELDS_FIELD_DATA_H
#define PCMS_ADAPTER_MESHFIELDS_MESH_FIELDS_FIELD_DATA_H

#include "pcms/field/layout/mesh_fields.h"
#include "pcms/field/evaluator/mesh_fields_backend.h"
#include "pcms/field/field_data.h"
#include "pcms/field/field_metadata.h"
#include "pcms/utility/assert.h"
#include "pcms/utility/arrays.h"

#include <Kokkos_Core.hpp>
#include <memory>

namespace pcms
{

template <typename T>
class MeshFieldsFieldData : public FieldData<T>
{
public:
  MeshFieldsFieldData(std::shared_ptr<const MeshFieldsAdapterLayout> layout,
                      FieldMetadata metadata)
    : layout_(std::move(layout)),
      metadata_(metadata),
      mesh_field_(MakeMeshFieldBackend<T>(*layout_)),
      data_("meshfields_field_data", static_cast<size_t>(layout_->OwnedSize()))
#if defined(PCMS_HAS_DISTINCT_DEVICE_MEMORY_SPACE)
      ,
      device_data_("meshfields_field_data_device",
                   static_cast<size_t>(layout_->OwnedSize()))
#endif
  {
    if (!mesh_field_) {
      throw pcms_error(
        "MeshFieldsFieldData does not support this layout/order");
    }
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
    SyncBackend(make_const_array_view(data_));
  }

#if defined(PCMS_HAS_DISTINCT_DEVICE_MEMORY_SPACE)
  Rank1View<const T, DeviceMemorySpace> GetDOFHolderDataDevice() const override
  {
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
    SyncBackend(make_const_array_view(data_));
  }
#endif

  std::shared_ptr<MeshFieldBackend<T>> GetMeshFieldBackend() const
  {
    return mesh_field_;
  }

private:
  void SyncBackend(Rank1View<const T, HostMemorySpace> flat)
  {
    auto nodes_per_dim = layout_->GetNodesPerDim();
    auto num_components = layout_->GetNumComponents();
    auto& mesh = layout_->GetMesh();
    size_t offset = 0;
    for (int i = 0; i <= mesh.dim(); ++i) {
      if (nodes_per_dim[i]) {
        size_t len = static_cast<size_t>(mesh.nents(i)) *
                     static_cast<size_t>(nodes_per_dim[i]) *
                     static_cast<size_t>(num_components);
        Rank1View<const T, HostMemorySpace> subspan{flat.data_handle() + offset,
                                                    len};
        mesh_field_->SetData(subspan, nodes_per_dim[i], num_components, i);
        offset += len;
      }
    }
  }

  std::shared_ptr<const MeshFieldsAdapterLayout> layout_;
  FieldMetadata metadata_;
  std::shared_ptr<MeshFieldBackend<T>> mesh_field_;
  Kokkos::View<T*, HostMemorySpace> data_;
#if defined(PCMS_HAS_DISTINCT_DEVICE_MEMORY_SPACE)
  mutable Kokkos::View<T*, DeviceMemorySpace> device_data_;
#endif
};

} // namespace pcms

#endif // PCMS_ADAPTER_MESHFIELDS_MESH_FIELDS_FIELD_DATA_H
